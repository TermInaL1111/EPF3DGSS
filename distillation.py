"""Paper-defined objectives, kept separate from dataset/model adaptations.

Deja-Events: https://arxiv.org/html/2611.21302v1, Eq. 7.
REALM: https://arxiv.org/html/2605.00271v2, Eq. 1 / Appendix A.2.
These are local transcriptions, not official training code.
"""
import torch
import torch.nn.functional as F


def deja_events_distance(student, teacher, token_mask=None):
    """[..., tokens, channels]; each view includes camera/register tokens.

Call with the teacher/student run at the SAME K. No spatial validity weights
are part of Eq. 7; spatial alignment is a prerequisite of this objective.
"""
    if student.shape != teacher.shape:
        raise ValueError('Teacher and student token shapes must match')
    student, teacher = student.float(), teacher.detach().float()
    magnitude = teacher.norm(dim=-1)
    weight = magnitude / magnitude.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    error = ((student-teacher).abs().mean(-1)*weight +
             1-F.cosine_similarity(student, teacher, dim=-1))
    # Optional REALM-inspired curriculum applied to the DVLT token objective.
    if token_mask is not None:
        if token_mask.shape != error.shape or not token_mask.any(dim=-1).all():
            raise ValueError('Each view needs a nonempty token mask')
        return ((error*token_mask).sum(-1)/token_mask.sum(-1)).mean()
    return error.mean()


def realm_activity_mask(voxel, epoch, patch_size=14):
    """Appendix A.2: input already resized/cropped onto the student grid.

Epoch is one-based. Nonzero voxel occupancy follows the paper's dataloader
recipe; callers must not substitute calibration validity for event activity.
"""
    if epoch < 1 or voxel.ndim != 4:
        raise ValueError('Expected one-based epoch and [views,bins,H,W] voxel')
    if voxel.shape[-2] % patch_size or voxel.shape[-1] % patch_size:
        raise ValueError('Input size must be a multiple of patch size')
    occupied = voxel.ne(0).any(dim=1, keepdim=True).float()
    mask = F.max_pool2d(occupied, patch_size, patch_size)
    radius = 6 if epoch >= 20 else 4 if epoch >= 15 else 2 if epoch >= 10 else 0
    for _ in range(radius):
        mask = F.max_pool2d(mask, 3, stride=1, padding=1)
    return mask.flatten(1).bool()


def realm_patch_drop_probability(epoch):
    """Appendix A.2: 30% dropping starting at epoch 8, not a rising ratio."""
    if epoch < 1:
        raise ValueError('Epoch is one-based')
    return .3 if epoch >= 8 else 0.


def realm_patch_distance(student, teacher, activity_mask):
    """Eq. 1 patch component: 0.1 MSE + 0.3 cosine + 0.6 L1.

Expects normalized outputs as emitted by DUNE, NOT unit-L2-normalized tokens.
CLS handling remains the caller's responsibility; this function is explicitly
only the spatial part, not a claim to reproduce the complete REALM trainer.
"""
    if student.shape != teacher.shape or activity_mask.shape != student.shape[:-1]:
        raise ValueError('Incompatible token/mask shapes')
    student, teacher = student.float(), teacher.detach().float()
    weight = activity_mask.to(student.dtype)
    denominator = weight.sum(-1)
    if (denominator == 0).any():
        raise ValueError('Empty activity mask: reject or explicitly skip the sample')
    error = (.1*(student-teacher).square().mean(-1) +
             .3*(1-F.cosine_similarity(student, teacher, dim=-1)) +
             .6*(student-teacher).abs().mean(-1))
    return ((error*weight).sum(-1)/denominator).sum()


def sample_shared_steps(teacher, generator=None):
    """Use pinned DVLT's own round-to-bin Beta sampler (model.py::_sample_K).

Sample once outside both forwards and pass this integer to student.geometry.
Do not call forward_train: its stochastic-depth solver is not the same as
running exactly K full refinement steps for feature distillation.
"""
    if (teacher.min_steps, teacher.num_steps, teacher.k_sampler_beta_a,
            teacher.k_sampler_beta_b) != (8, 16, 2, 1):
        raise ValueError('Teacher config does not match Deja-Events K distribution')
    return teacher._sample_K(generator)


@torch.no_grad()
def teacher_final_tokens(teacher, images, steps):
    """Pinned DVLT inference solver, explicit K, camera/register tokens retained."""
    if not 8 <= steps <= 16:
        raise ValueError('Distillation K must be in [8,16]')
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be frozen and in eval mode')
    from dvlt.model.dvlt.model import _slice_expand_flatten
    b, s, _, h, w = images.shape
    patch = teacher._encode_images(images)
    register = teacher.register_token.expand(b,s,-1,-1).reshape(b*s,teacher.num_register_tokens,-1)
    camera = _slice_expand_flatten(teacher.camera_token,b,s)
    tokens = torch.cat([camera,register,patch],1)
    pos = teacher._get_rope_positions(b*s,h,w,images.device)
    previous = teacher.inference_steps
    try:
        teacher.inference_steps = steps
        return teacher._solve_inference(tokens,pos,b,s)
    finally:
        teacher.inference_steps = previous
