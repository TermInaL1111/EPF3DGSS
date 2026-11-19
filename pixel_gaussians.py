"""Native-resolution event decoder with one anisotropic Gaussian per source pixel."""
import torch
import torch.nn.functional as F
from torch import nn

from model import EventGaussians, grid


class PixelGaussians(EventGaussians):
    def __init__(self):
        super().__init__(36,4)
        # A compact two-stream decoder combines native event detail with multiview context.
        self.context = nn.Conv2d(64,12,1)
        self.detail = nn.Sequential(nn.Conv2d(16,16,3,padding=1),nn.GELU(),nn.Conv2d(16,10,1))
        nn.init.zeros_(self.detail[-1].weight)
        nn.init.zeros_(self.detail[-1].bias)
        xy = (grid(480,640)-torch.tensor([319.5,239.5]))/888.8889
        self.register_buffer("native_rays",F.pad(xy,(0,1),value=1),persistent=False)

    def forward(self,events,native):
        feature, reverse = self.view_features(events)
        coarse = self.scene_from_features(events,feature,reverse)
        batch = len(events)
        if native.shape != (batch,2,4,480,640):
            raise ValueError("Expected four native event channels per EV3DS source")
        both = torch.stack([feature,reverse],1).flatten(0,1)
        def upsample(x):
            return F.interpolate(x,size=(480,640),mode="bilinear",align_corners=False)
        high = torch.cat([upsample(self.context(both)),native.flatten(0,1)],1)
        correction = self.detail(high)
        def expand_grid(x,channels):
            field = x.reshape(batch*2,24,32,channels).permute(0,3,1,2)
            return upsample(field)
        depth = expand_grid(coarse["source_depth"],1)*correction[:,:1].clamp(-3,3).exp()
        local = self.native_rays * depth.flatten(2).transpose(1,2)
        first,second = local.reshape(batch,2,-1,3).unbind(1)
        centers = torch.cat([first,second @ coarse["rotation"].transpose(1,2)+coarse["translation"][:,None]],1)
        def pack(x):
            return x.flatten(2).transpose(1,2).reshape(batch,-1,x.shape[1]).contiguous()
        scales = expand_grid(coarse["scales"],3)*correction[:,1:4].clamp(-3,3).exp()/20
        # The coarse and residual quaternions use the reference-camera coordinate system.
        quats = F.normalize(expand_grid(coarse["quats"],4)+correction[:,4:8],dim=1)
        def attribute(key,delta):
            return pack((expand_grid(coarse[key],1).clamp(.0001,.9999).logit()+delta).sigmoid())[...,0]
        return dict(centers=centers, scales=pack(scales), quats=pack(quats),
                    opacity=attribute("opacity",correction[:,8:9]),gray=attribute("gray",correction[:,9:10]),
                    source_depth=depth.reshape(batch,2,480,640),rotation=coarse["rotation"],
                    translation=coarse["translation"],feature=coarse["feature"])
