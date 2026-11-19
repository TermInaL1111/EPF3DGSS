"""Event-adapted DVLT geometry backbone with a pixel-aligned Gaussian readout."""
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from dvlt.model.dvlt.model import DVLTModel, _slice_expand_flatten
from dvlt.common.rays import rays_to_pose
from gsplat import rasterization


def load_rgb(assets):
    net = DVLTModel(**json.loads((assets/"config.json").read_text()))
    net.load_state_dict(load_file(assets/"dvlt.safetensors"),strict=True)
    return net.cuda().eval().requires_grad_(False)


def voxelize(raw):
    x,y,t,p = raw
    valid = (x>=0)&(x<640)&(y>=0)&(y<480)
    x,y,t,p = x[valid].astype(int),y[valid].astype(int),t[valid],np.where(p[valid]>0,1.,-1.)
    volume = np.zeros((5,480,640),np.float32)
    if len(t):
        tau = 4*(t-t.min())/max(float(t.max()-t.min()),1e-12)
        low = tau.astype(int)
        alpha = tau-low
        np.add.at(volume,(low,y,x),p*(1-alpha))
        np.add.at(volume,(np.minimum(low+1,4),y,x),p*alpha)
    return torch.from_numpy(volume)


class LowRankQKV(nn.Module):
    def __init__(self, base, rank=16):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features,rank,bias=False)
        self.up = nn.Linear(rank,base.out_features,bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self,x):
        return self.base(x)+self.up(self.down(x))


def gated_update(block, correction, x, interval, pos, density, special):
    gates = block.depth_scale(interval)[:,None]
    step = torch.cat([block.depth_scale._sinusoidal(t) for t in interval.unbind(-1)],-1)
    conditions = torch.cat([density,step[:,None].expand(-1,x.shape[1],-1)],-1)
    ca,cm = (correction(conditions)*special).unbind(-1)
    attn,mlp,out = gates.chunk(3,-1)
    x = x + (1+ca[...,None])*attn*block.ls1(block.attn(block.norm1(x),pos=pos))
    x = x + (1+cm[...,None])*mlp*block.ls2(block.mlp(block.norm2(x)))
    return x*out


class EventDVLT(nn.Module):
    def __init__(self, assets):
        super().__init__()
        self.backbone = load_rgb(assets)
        blocks = self.backbone.recurrent_blocks[0]
        for block in [*self.backbone.patch_embed_encoder.blocks,blocks.frame_attn,blocks.global_attn]:
            block.attn.qkv = LowRankQKV(block.attn.qkv)
        self.corrections = nn.ModuleList()
        for branch in (blocks.frame_attn,blocks.global_attn):
            branch.depth_scale.proj.requires_grad_(True)
            correction = nn.Sequential(nn.Linear(129,16),nn.SiLU(),nn.Linear(16,2))
            nn.init.zeros_(correction[-1].weight)
            nn.init.zeros_(correction[-1].bias)
            self.corrections.append(correction)
        self.channel_scale = nn.Parameter(1/torch.tensor([.229,.224,.225])[None,:,None,None])
        self.channel_bias = nn.Parameter(-torch.tensor([.485,.456,.406])[None,:,None,None]/
                                         torch.tensor([.229,.224,.225])[None,:,None,None])
        self.event_adapter = nn.Conv2d(32,3,1)
        nn.init.zeros_(self.event_adapter.weight)
        nn.init.zeros_(self.event_adapter.bias)
        self.geometry_features = nn.Conv2d(768,24,1)
        self.field_features = nn.Conv2d(32,12,1)
        # The geometry tokens and F³ detail jointly parameterize surface Gaussians.
        self.gs_head = nn.Sequential(nn.Conv2d(41,24,3,padding=1),nn.GELU(),nn.Conv2d(24,10,1))
        nn.init.zeros_(self.gs_head[-1].weight)
        nn.init.zeros_(self.gs_head[-1].bias)
        self.log_metric_scale = nn.Parameter(torch.tensor(-.7))
        self.backbone.eval()

    def geometry(self,voxel,field,steps=12,mask_fraction=0.):
        b,s = voxel.shape[:2]
        h,w = 168,224
        v = voxel.flatten(0,1)
        v = F.normalize(v.flatten(2),dim=2).reshape_as(v)
        channels = torch.stack([v.clamp_min(0).sum(1),v.abs().sum(1),(-v).clamp_min(0).sum(1)],1)
        channels = channels/channels.amax((-1,-2),keepdim=True).clamp_min(1e-7)
        inputs = F.interpolate(channels,(h,w),mode="area")*self.channel_scale+self.channel_bias
        inputs = inputs + F.interpolate(self.event_adapter(field.flatten(0,1)),(h,w),mode="bilinear",align_corners=False)
        patch = self.backbone.patch_embed_encoder(inputs,is_training=True)["x_norm_patchtokens"]
        density = F.adaptive_avg_pool2d(channels[:,1:2],(h//14,w//14)).flatten(2).transpose(1,2)
        density = density/density.amax(1,keepdim=True).clamp_min(1e-6)
        if mask_fraction:
            hidden = torch.rand_like(density)<mask_fraction
            patch = patch.masked_fill(hidden,0)
        else:
            hidden = torch.zeros_like(density,dtype=torch.bool)
        n = self.backbone
        extra = n.patch_start_idx
        reg = n.register_token.expand(b,s,-1,-1).reshape(b*s,-1,768)
        x = torch.cat([_slice_expand_flatten(n.camera_token,b,s),reg,patch],1)
        density = F.pad(density,(0,0,extra,0))
        spatial = F.pad(torch.ones_like(density[:,extra:]),(0,0,extra,0))
        pos = n._get_rope_positions(b*s,h,w,x.device)
        block = n.recurrent_blocks[0]
        for k in range(steps):
            pair = x.new_tensor([k/(steps-1),min(k+1,steps-1)/(steps-1)],dtype=torch.float32)
            x = gated_update(block.frame_attn,self.corrections[0],x,pair.expand(b*s,2),pos,density,spatial)
            x = gated_update(block.global_attn,self.corrections[1],x.reshape(b,-1,768),pair.expand(b,2),None,
                             density.reshape(b,-1,1),spatial.reshape(b,-1,1)).reshape(b*s,-1,768)
        return x,pos,hidden

    def forward(self,voxel,field,steps=12,mask_fraction=0.):
        b,s = voxel.shape[:2]
        tokens,pos,hidden = self.geometry(voxel,field,steps,mask_fraction)
        n = self.backbone
        raw = n._decode(tokens,168,224,b,s,pos)
        # Follow the released DVLT ray fitting procedure. Its fitted poses have arbitrary scale.
        with torch.no_grad(), torch.autocast("cuda",enabled=False):
            rays = raw["rays"].float()
            poses,intrinsic = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
            relative = torch.linalg.solve(poses[:,:1],poses)
        patch = tokens[:,n.patch_start_idx:].transpose(1,2).reshape(b*s,768,12,16)
        size = (480,640)
        def resize(x):
            return F.interpolate(x,size,mode="bilinear",align_corners=False)
        detail = torch.cat([resize(self.geometry_features(patch)),resize(self.field_features(field.flatten(0,1))),
                            voxel.flatten(0,1).asinh()],1)
        attrs = self.gs_head(detail).float()
        scale = self.log_metric_scale.exp()
        depth = resize(raw["depth"].reshape(b*s,1,168,224).float())*scale*(.25*attrs[:,:1].tanh()).exp()
        # Scale the predicted intrinsics using the pixel-center convention of the resize.
        k = intrinsic.reshape(b*s,3,3)
        y,x = torch.meshgrid(torch.arange(480,device=k.device),torch.arange(640,device=k.device),indexing="ij")
        xx = ((x+.5)*224/640-.5-k[:,0,2,None,None])/k[:,0,0,None,None].clamp_min(1)
        yy = ((y+.5)*168/480-.5-k[:,1,2,None,None])/k[:,1,1,None,None].clamp_min(1)
        local = torch.stack([xx,yy,torch.ones_like(xx)],-1)*depth[:,0,:,:,None]
        local = local.reshape(b,s,-1,3)
        center = local @ relative[:,:,:3,:3].transpose(-1,-2) + scale*relative[:,:,:3,3][:,:,None]
        sigma = depth/(k[:,0,0,None,None,None].clamp_min(1)*640/224)
        sizes = sigma*(attrs[:,1:4].clamp(-3,3)).exp()
        quat = attrs[:,4:8]+attrs.new_tensor([1.,0,0,0])[None,:,None,None]
        def pack(x):
            return x.permute(0,2,3,1).reshape(b,-1,x.shape[1]).contiguous()
        return dict(centers=center.flatten(1,2).contiguous(),scales=pack(sizes),quats=pack(F.normalize(quat,dim=1)),
                    gray=pack(attrs[:,8:9].sigmoid()),opacity=pack(attrs[:,9:10].sigmoid())[...,0].contiguous(),
                    tokens=tokens,masked=hidden,source_depth=depth.reshape(b,s,480,640),
                    rotation=relative[:,1,:3,:3],translation=relative[:,1,:3,3]*scale)


def render(cloud,query):
    k=query.new_tensor([[888.8889,0,320],[0,888.8889,240],[0,0,1]]).expand(len(query),1,3,3)
    image,_,_=rasterization(cloud["centers"],cloud["quats"],cloud["scales"],cloud["opacity"],cloud["gray"],
                             torch.linalg.inv(query)[:,None],k,640,480,packed=False,eps2d=.25)
    return image[:,0,:,:,0]
