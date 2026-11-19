"""Low-resolution calibrated EV3DS pilot. The first source camera defines the world."""
import torch
from torch import nn
import torch.nn.functional as F


def grid(height, width):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2).float()


class EventGaussians(nn.Module):
    def __init__(self, input_channels=4, scale=1, fusion_mode="none"):
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode != "none":
            self.register_buffer("fusion_id",torch.tensor({"geometry":1,"event":2}[fusion_mode]))
        self.scale = scale
        self.height, self.width = 6*scale, 8*scale
        self.encoder = nn.Sequential(nn.Conv2d(input_channels, 32, 5, 2, 2), nn.GELU(),
                                     nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())
        self.attributes = nn.Conv2d(64, 4, 1)
        if scale > 1:
            self.shape_head = nn.Conv2d(64,7,1)
            nn.init.zeros_(self.shape_head.weight)
            nn.init.zeros_(self.shape_head.bias)
            self.shape_head.bias.data[3] = 1
        self.camera = nn.Linear(64, 9)
        nn.init.zeros_(self.camera.weight)
        self.camera.bias.data.copy_(torch.tensor([1.,0,0,0,1,0,0,0,0]))
        self.distill = nn.Conv2d(64, 768, 1)
        self.register_buffer("rays", (grid(self.height,self.width) + .5 - torch.tensor([self.width,self.height])/2) / (888.8889*scale/80))
        self.register_buffer("pixels", grid(24*scale,32*scale))

    def fuse_gray(self, local, depth, gray, events, rotation, translation):
        batch = len(events)
        # For each local grid ray, locate the same estimated point in the other camera.
        left,right = local.reshape(batch,2,-1,3).unbind(1)
        xyz = torch.stack([(left-translation[:,None]) @ rotation,
                           right @ rotation.transpose(1,2)+translation[:,None]],1)
        z = xyz[...,2]
        focal = 888.8889*self.scale/80
        uv = xyz[...,:2]/z.clamp_min(.01)[...,None]
        norm = uv*uv.new_tensor([2*focal/self.width,2*focal/self.height])
        grid_xy = norm.reshape(batch*2,self.height,self.width,2)
        depth = depth.reshape(batch,2,self.height,self.width)
        gray = gray.sigmoid().reshape_as(depth)
        support = F.avg_pool2d(events[:,:,:4].expm1().sum(2),4).log1p()
        layers = torch.stack([depth,gray,support],2).flip(1).flatten(0,1)
        seen_depth,seen_gray,seen_support = F.grid_sample(layers,grid_xy,align_corners=False).reshape(
            batch,2,3,self.height,self.width).unbind(2)
        z = z.reshape_as(depth)
        valid = (z>.01) & (norm.abs().amax(-1).reshape_as(z)<1)
        consistency = (-(seen_depth-z).abs()/(.05*z.clamp_min(.01))).exp()*valid
        if self.fusion_mode == "event":
            consistency = consistency*(1+seen_support)/(1+support)
        return ((gray + consistency*seen_gray)/(1+consistency)).flatten(1)

    def view_features(self, events):
        left, right = self.encoder(events.flatten(0, 1)).reshape(-1,2,64,self.height,self.width).unbind(1)
        feature = self.fusion(torch.cat([left, right], 1))
        reverse = self.fusion(torch.cat([right, left], 1))
        return feature, reverse

    def forward(self,events):
        return self.scene_from_features(events,*self.view_features(events))

    def scene_from_features(self,events,feature,reverse):
        a, b, translation = self.camera(feature.mean((-2,-1))).chunk(3, -1)
        u = F.normalize(a, dim=-1)
        v = F.normalize(b - (b*u).sum(-1, keepdim=True)*u, dim=-1)
        rotation = torch.stack([u,v,torch.cross(u,v,dim=-1)], -1)
        fused = torch.stack([feature,reverse],1).flatten(0,1)
        attrs = self.attributes(fused)
        depth, radius, opacity, gray = attrs.flatten(2).transpose(1,2).unbind(-1)
        depth = F.softplus(depth) + .03
        centers = F.pad(self.rays,(0,1),value=1.) * depth[...,None]
        first, second = centers.reshape(len(events),2,-1,3).unbind(1)
        merged_gray = gray.sigmoid().reshape(len(events),-1)
        if self.fusion_mode != "none":
            merged_gray = self.fuse_gray(centers,depth,gray,events,rotation,translation)
        centers = torch.cat([first, second @ rotation.transpose(1,2) + translation[:,None]], 1)
        count = 2*self.height*self.width
        scene = dict(source_depth=depth.reshape(-1,2,self.height,self.width), centers=centers, radius=(F.softplus(radius)*.03+.001).reshape(-1,count),
                    opacity=opacity.sigmoid().reshape(-1,count), gray=merged_gray,
                    rotation=rotation, translation=translation, feature=feature)

        if self.scale > 1:
            shape = self.shape_head(fused).flatten(2).transpose(1,2).reshape(-1,count,7)
            # Rotation and scale are expressed in the shared first-camera frame.
            scene["scales"] = scene["radius"][...,None]*shape[...,:3].clamp(-3,3).exp()/self.scale
            scene["quats"] = F.normalize(shape[...,3:],dim=-1)
        return scene

    def render(self, scene, target, output_hw=None):
        if self.scale > 1:
            from gsplat import rasterization
            height,width = output_hw or (self.height*4,self.width*4)
            if height*4 != width*3:
                raise ValueError("Rendering uses the EV3DS native aspect ratio")
            focal = 888.8889*width/640
            # gsplat evaluates integer pixels at half-integer sample coordinates.
            k = target.new_tensor([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
            image,_,_ = rasterization(scene["centers"],scene["quats"],scene["scales"],
                                      scene["opacity"],scene["gray"][...,None],
                                      torch.linalg.inv(target)[:,None],k.expand(len(target),1,3,3),
                                      width,height,packed=False,eps2d=.25)
            return image[:,0,:,:,0]

        if output_hw not in (None,(24,32)):
            raise ValueError("The analytic pilot renderer uses a 24 by 32 grid")
        # target is the evaluation camera expressed relative to the left source.
        xyz = (scene["centers"] - target[:,None,:3,3]) @ target[:,:3,:3]
        z = xyz[...,2].clamp_min(.01)
        focal = 888.8889/20
        xy = xyz[...,:2]/z[...,None]*focal + xyz.new_tensor([15.5,11.5])
        delta = self.pixels - xy[:,:,None,:]
        # Perspective Jacobian projects isotropic 3D covariance to an ellipse.
        r = xyz[...,:2]/z[...,None]
        scale2 = (focal*scene["radius"]/z).square()
        base = scale2 + .25
        along = (delta*r[:,:,None]).sum(-1)
        maha = (delta.square().sum(-1) - along.square() *
                (scale2/(base+scale2*r.square().sum(-1)))[:,:,None])/base[:,:,None]
        alpha = (-.5*maha.clamp_min(0)).exp() * (scene["opacity"]*(xyz[...,2]>.01))[:,:,None]
        order = z.argsort(1)
        alpha = alpha.gather(1,order[:,:,None].expand_as(alpha)).clamp(max=.999)
        transmission = (1-alpha).cumprod(1)
        weights = alpha * F.pad(transmission[:,:-1],(0,0,1,0),value=1.)
        return (weights * scene["gray"].gather(1,order)[:,:,None]).sum(1).reshape(-1,24,32)
