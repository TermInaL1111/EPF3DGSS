"""Event-adapted DVLT geometry backbone and Gaussian readouts."""
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from dvlt.model.dvlt.model import DVLTModel, _slice_expand_flatten
from dvlt.common.rays import rays_to_pose
from dvlt.common.rotation import mat_to_quat
from gsplat import rasterization


def load_rgb(assets, device='cuda'):
    net = DVLTModel(**json.loads((assets/"config.json").read_text()))
    net.load_state_dict(load_file(assets/"dvlt.safetensors"),strict=True)
    return net.to(device).eval().requires_grad_(False)


def load_scene_weights(model, checkpoint):
    """Load a complete event-only Gaussian scene model and return (steps, use_f3)."""
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    steps, use_f3 = 12, True
    if isinstance(checkpoint.get('geometry'), str):
        if checkpoint['geometry'] != 'predicted':
            raise ValueError('Event-only reconstruction requires predicted geometry')
        steps, use_f3 = 16, checkpoint['f3']
        model.configure_head(checkpoint.get('head', 'legacy'))
        expected = {name for name,p in model.named_parameters() if p.requires_grad}
        checkpoint = checkpoint['model']
        if not use_f3:
            expected = {key for key in expected if not key.startswith('event_adapter.')}
    if set(checkpoint) != expected:
        raise ValueError('Checkpoint must contain the complete geometry adapters and Gaussian head')
    model.load_state_dict(checkpoint,strict=False)
    return steps,use_f3


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
    def __init__(self, assets, device='cuda'):
        super().__init__()
        self.backbone = load_rgb(assets, device)
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
        self.head_type = 'legacy'

    def configure_head(self, kind):
        if kind == self.head_type:
            return
        if kind == 'fixed':
            head = GaussianAttributes()
        elif kind in ('voxel', 'octree'):
            from voxel_gaussians import VoxelGaussianHead

            head = VoxelGaussianHead(octree=kind == "octree")
        else:
            raise ValueError(f'Unknown Gaussian head: {kind}')
        device = self.log_metric_scale.device
        self.gs_head = head.to(device).train(self.training)
        self.head_type = kind

    def geometry(self,voxel,field,steps=12,mask_fraction=0.,input_mask=None,feature_states=None):
        b,s = voxel.shape[:2]
        h,w = 168,224
        v = voxel.flatten(0,1)
        v = F.normalize(v.flatten(2),dim=2).reshape_as(v)
        channels = torch.stack([v.clamp_min(0).sum(1),v.abs().sum(1),(-v).clamp_min(0).sum(1)],1)
        channels = channels/channels.amax((-1,-2),keepdim=True).clamp_min(1e-7)
        inputs = F.interpolate(channels,(h,w),mode="area")*self.channel_scale+self.channel_bias
        inputs = inputs + F.interpolate(self.event_adapter(field.flatten(0,1)),(h,w),mode="bilinear",align_corners=False)
        patch = self.backbone.patch_embed_encoder(inputs,masks=input_mask,is_training=True)["x_norm_patchtokens"]
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
        taps = tuple(round(steps * fraction / 4) for fraction in (1, 2, 3, 4)) if feature_states is not None else ()
        for k in range(steps):
            pair = x.new_tensor([k/(steps-1),min(k+1,steps-1)/(steps-1)],dtype=torch.float32)
            x = gated_update(block.frame_attn,self.corrections[0],x,pair.expand(b*s,2),pos,density,spatial)
            x = gated_update(block.global_attn,self.corrections[1],x.reshape(b,-1,768),pair.expand(b,2),None,
                             density.reshape(b,-1,1),spatial.reshape(b,-1,1)).reshape(b*s,-1,768)
            if k+1 in taps:
                feature_states.append(x)
        return x,pos,hidden

    def forward(self,voxel,field,steps=12,mask_fraction=0.,return_geometry=False, calibrated_geometry=None,
                source_cameras=None, depth_refinement=None, event_leaf_ranking=False, matched_leaf_budget=False,
                appearance_inputs=None, depth_readout=None):
        b,s = voxel.shape[:2]
        if depth_refinement is not None and calibrated_geometry is not None:
            raise ValueError('Depth refinement requires predicted geometry')
        if depth_readout is not None and (source_cameras is None or calibrated_geometry is not None):
            raise ValueError('Depth readout interventions require fixed source cameras and predicted geometry')
        if (event_leaf_ranking or matched_leaf_budget) and (self.head_type != 'octree' or depth_refinement is None):
            raise ValueError('Event leaf ranking requires an octree head and boundary predictions')
        leaf_scores = None
        use_voxels = self.head_type in ('voxel', 'octree')
        if use_voxels and (steps < 4 or calibrated_geometry is not None):
            raise ValueError('The voxel head requires predicted depth and at least four recurrent steps')
        feature_states = [] if use_voxels else None
        tokens,pos,hidden = self.geometry(voxel,field,steps,mask_fraction,feature_states=feature_states)
        n = self.backbone
        source_valid = None
        raw = None
        if calibrated_geometry is None:
            raw = n._decode(tokens,168,224,b,s,pos)
            # The published DVLT ray fit recovers poses up to scale.
            with torch.no_grad(), torch.autocast("cuda",enabled=False):
                rays = raw["rays"].float()
                poses,intrinsic = rays_to_pose(rays,torch.ones_like(rays[...,0]),168,224,14)
                relative = torch.linalg.solve(poses[:,:1],poses)
            if source_cameras is not None:
                intrinsic, relative = source_cameras
        else:
            metric_depth, intrinsic, relative = calibrated_geometry
            source_valid = metric_depth > 0
        def patch_grid(state):
            return state[:,n.patch_start_idx:].transpose(1,2).reshape(b*s,768,12,16)

        size = (480,640)
        def resize(x):
            return F.interpolate(x,size,mode="bilinear",align_corners=False)
        appearance_voxel, appearance_field = (voxel,field) if appearance_inputs is None else appearance_inputs
        appearance = torch.cat([resize(self.field_features(appearance_field.flatten(0,1))),
                                appearance_voxel.flatten(0,1).asinh()],1)
        fixed = self.head_type == 'fixed'
        if calibrated_geometry is None:
            depth_prediction, depth_confidence = ((raw['depth'],raw['depth_conf'])
                                                  if depth_readout is None else depth_readout)
            scale = self.log_metric_scale.exp()
            coarse_depth = depth_prediction.reshape(b*s,1,168,224).float()
            depth = resize(coarse_depth)
            if depth_refinement is not None:
                confidence = depth_confidence.reshape_as(coarse_depth).float()
                corrected = depth_refinement(patch_grid(tokens), coarse_depth, confidence)
                if event_leaf_ranking or matched_leaf_budget:
                    from event_depth_boundary import pixel_boundary_probability

                    corrected, edges = corrected
                    leaf_scores = pixel_boundary_probability(edges).reshape(b, -1)
                if corrected.shape != depth.shape:
                    raise ValueError('Refined depth must match the Gaussian source grid')
                depth = corrected
            depth = depth * scale
        else:
            scale = metric_depth.new_tensor(1.)
            depth = metric_depth.reshape(b*s,1,480,640).clamp_min(1e-4)
        if feature_states is None:
            detail = torch.cat([resize(self.geometry_features(patch_grid(tokens))), appearance],1)
            if fixed:
                log_size, quat, gray, opacity = self.gs_head(detail,depth)
            else:
                attrs = self.gs_head(detail).float()
                depth = depth*(.25*attrs[:,:1].tanh()).exp()
                log_size, quat, gray, opacity = attrs[:,1:4],attrs[:,4:8],attrs[:,8:9],attrs[:,9:10]
        # Scale the predicted intrinsics using the pixel-center convention of the resize.
        k = intrinsic.reshape(b*s,3,3)
        y,x = torch.meshgrid(torch.arange(480,device=k.device),torch.arange(640,device=k.device),indexing="ij")
        xx = ((x+.5)*224/640-.5-k[:,0,2,None,None])/k[:,0,0,None,None].clamp_min(1)
        yy = ((y+.5)*168/480-.5-k[:,1,2,None,None])/k[:,1,1,None,None].clamp_min(1)
        local = torch.stack([xx,yy,torch.ones_like(xx)],-1)*depth[:,0,:,:,None]
        local = local.reshape(b,s,-1,3)
        translation_scale = 1. if source_cameras is not None else scale
        if feature_states is not None:
            # Keep metric coordinates in FP32 before assigning voxel membership.
            with torch.autocast(voxel.device.type, enabled=False):
                center = local.float() @ relative[:,:,:3,:3].float().transpose(-1,-2)
                center = center + translation_scale*relative[:,:,:3,3].float()[:,:,None]
            points = center.flatten(1,2)
            valid = torch.isfinite(points).all(-1) & (depth.reshape(b,-1) > 0)
            confidence = None
            if self.head_type == 'octree':
                confidence = resize(depth_confidence.reshape(b*s,1,168,224).float()).reshape(b,-1)
            result = self.gs_head([self.geometry_features(patch_grid(state)) for state in feature_states],
                                  appearance, points, valid, confidence=confidence, boundary=leaf_scores,
                                  event_leaf_ranking=event_leaf_ranking, matched_leaf_budget=matched_leaf_budget)
            result.update(tokens=tokens,masked=hidden,source_depth=depth.reshape(b,s,480,640),
                          rotation=relative[:,1,:3,:3],translation=relative[:,1,:3,3]*translation_scale)
            if return_geometry:
                result['raw_geometry'] = raw
            return result
        center = local @ relative[:,:,:3,:3].transpose(-1,-2) + translation_scale*relative[:,:,:3,3][:,:,None]
        sigma = depth/(k[:,0,0,None,None,None].clamp_min(1)*640/224)
        sizes = sigma*log_size.clamp(-3,3).exp()
        quat = F.normalize(quat+quat.new_tensor([1.,0,0,0])[None,:,None,None],dim=1)
        if fixed:
            camera_q = mat_to_quat(relative[:,:,:3,:3].float()).roll(1,-1).reshape(b*s,4,1,1)
            qw, qv = camera_q[:,:1], camera_q[:,1:]
            w, v = quat[:,:1], quat[:,1:]
            quat = torch.cat([qw*w-(qv*v).sum(1,keepdim=True), qw*v+w*qv+torch.cross(qv.expand_as(v),v,dim=1)],1)
        def pack(x):
            return x.permute(0,2,3,1).reshape(b,-1,x.shape[1]).contiguous()
        result = dict(centers=center.flatten(1,2).contiguous(),scales=pack(sizes),quats=pack(quat),
                    gray=pack(gray.sigmoid()),opacity=pack(opacity.sigmoid())[...,0].contiguous(),
                    tokens=tokens,masked=hidden,source_depth=depth.reshape(b,s,480,640),
                    rotation=relative[:,1,:3,:3],translation=relative[:,1,:3,3]*translation_scale)
        if source_valid is not None:
            result["opacity"] = result["opacity"] * source_valid.reshape(b,-1)
        if return_geometry:
            result["raw_geometry"] = raw
        return result


class GaussianAttributes(nn.Module):
    """Separate event intensity and Gaussian shape branches at fixed depth."""

    def __init__(self):
        super().__init__()
        self.geometry = nn.Conv2d(24, 24, 3, padding=1)
        self.appearance = nn.Conv2d(17, 24, 3, padding=1)
        self.intensity = nn.Conv2d(24, 1, 1)
        self.shape = nn.Sequential(
            nn.Conv2d(26, 24, 3, padding=1), nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.GELU())
        self.scale = nn.Conv2d(24, 3, 1)
        self.rotation = nn.Conv2d(24, 4, 1)
        self.opacity = nn.Conv2d(24, 1, 1)
        for layer in (self.scale, self.rotation, self.opacity):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, depth):
        geometry, appearance = features.split((24,17),dim=1)
        combined = F.gelu(self.geometry(geometry)+self.appearance(appearance))
        gray = self.intensity(combined)
        shape = self.shape(torch.cat((combined,depth.clamp_min(1e-6).log(),gray.sigmoid()),1))
        return tuple(layer(shape).float() for layer in (self.scale,self.rotation)) + (gray.float(),self.opacity(shape).float())


def render(cloud,query,intrinsics=None,*,return_alpha=False):
    if intrinsics is None:
        intrinsics = query.new_tensor([[888.8889,0,320],[0,888.8889,240],[0,0,1]]).expand(len(query),3,3)
    k = intrinsics[:,None]
    image,alpha,_=rasterization(cloud["centers"],cloud["quats"],cloud["scales"],cloud["opacity"],cloud["gray"],
                                 torch.linalg.inv(query)[:,None],k,640,480,packed=False,eps2d=.25)
    gray = image[:,0,:,:,0]
    return (gray,alpha[:,0,:,:,0]) if return_alpha else gray
