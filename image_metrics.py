"""Grayscale image metrics for project renders in the normalized [0,1] range."""
import torch
import torch.nn.functional as F


def gray_metrics(prediction, target, *, padded=False):
    """Return per-image PSNR and valid-window SSIM for tensors shaped [N,H,W]."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('Expected matching [N,H,W] grayscale images')
    if min(prediction.shape[-2:]) < 11:
        raise ValueError('SSIM requires images at least 11 pixels across')
    prediction, target = prediction.float(), target.float()
    psnr = -10*(prediction-target).square().mean((-2,-1)).clamp_min(1e-12).log10()

    offsets = torch.arange(-5,6,device=prediction.device,dtype=prediction.dtype)
    window = (-offsets.square()/(2*1.5**2)).exp()
    window = window/window.sum()
    kernel = (window[:,None]*window[None,:])[None,None]
    x,y = prediction[:,None],target[:,None]
    mean_x,mean_y,mean_x2,mean_y2,mean_xy = F.conv2d(
        torch.cat([x,y,x.square(),y.square(),x*y]),kernel,padding=5 if padded else 0).chunk(5)
    var_x = (mean_x2-mean_x.square()).clamp_min(0)
    var_y = (mean_y2-mean_y.square()).clamp_min(0)
    covariance = mean_xy-mean_x*mean_y
    c1,c2 = .01**2,.03**2
    ssim = ((2*mean_x*mean_y+c1)*(2*covariance+c2) /
            ((mean_x.square()+mean_y.square()+c1)*(var_x+var_y+c2)))
    return psnr,ssim.mean((1,2,3))


class PerceptualMetrics:
    """Compute the two LPIPS backbones used by the upstream evaluation code."""

    def __init__(self, device):
        import lpips

        self.models = {name: lpips.LPIPS(net=name).eval().requires_grad_(False).to(device)
                       for name in ('vgg','alex')}

    @torch.no_grad()
    def __call__(self, prediction, target):
        # Public LPIPS takes RGB normalized to [-1,1]. Repeat grayscale channels.
        x = prediction[:,None].float().repeat(1,3,1,1)*2-1
        y = target[:,None].float().repeat(1,3,1,1)*2-1
        return {f'lpips_{name}':float(model(x,y).mean()) for name,model in self.models.items()}
