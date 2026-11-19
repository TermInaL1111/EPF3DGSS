"""Low-resolution calibrated EV3DS pilot. The first source camera defines the world."""
import torch
from torch import nn
import torch.nn.functional as F


def grid(height, width):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return torch.stack([x, y], -1).reshape(-1, 2).float()


class EventGaussians(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(4, 32, 5, 2, 2), nn.GELU(),
                                     nn.Conv2d(32, 64, 3, 2, 1), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.GELU())
        self.attributes = nn.Conv2d(64, 4, 1)
        self.camera = nn.Linear(64, 9)
        nn.init.zeros_(self.camera.weight)
        self.camera.bias.data.copy_(torch.tensor([1.,0,0,0,1,0,0,0,0]))
        self.distill = nn.Conv2d(64, 768, 1)
        self.register_buffer("rays", (grid(6, 8) - torch.tensor([3.5, 2.5])) / (888.8889/80))
        self.register_buffer("pixels", grid(24, 32))

    def forward(self, events):
        left, right = self.encoder(events.flatten(0, 1)).reshape(-1,2,64,6,8).unbind(1)
        feature = self.fusion(torch.cat([left, right], 1))
        reverse = self.fusion(torch.cat([right, left], 1))
        a, b, translation = self.camera(feature.mean((-2,-1))).chunk(3, -1)
        u = F.normalize(a, dim=-1)
        v = F.normalize(b - (b*u).sum(-1, keepdim=True)*u, dim=-1)
        rotation = torch.stack([u,v,torch.cross(u,v,dim=-1)], -1)
        attrs = self.attributes(torch.stack([feature,reverse],1).flatten(0,1))
        depth, radius, opacity, gray = attrs.flatten(2).transpose(1,2).unbind(-1)
        depth = F.softplus(depth) + .03
        centers = F.pad(self.rays,(0,1),value=1.) * depth[...,None]
        first, second = centers.reshape(-1,2,48,3).unbind(1)
        centers = torch.cat([first, second @ rotation.transpose(1,2) + translation[:,None]], 1)
        return dict(centers=centers, radius=(F.softplus(radius)*.03+.001).reshape(-1,96),
                    opacity=opacity.sigmoid().reshape(-1,96), gray=gray.sigmoid().reshape(-1,96),
                    rotation=rotation, translation=translation, feature=feature)

    def render(self, scene, target):
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
