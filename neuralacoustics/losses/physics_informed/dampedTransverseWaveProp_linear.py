import importlib.simple
import torch
import torch.nn.functional as F

fd2d = importlib.import_module("solvers.FDTD.2D.dampedTransverseWaveProp_linear.dampedTransverseWaveProp_linear")

class WaveEqnLoss:
  def __init__(self, dev, w, h, mu, rho, gamma):
    self.mu    = mu
    self.rho   = rho
    self.gamma = gamma

    mask = torch.zeros((h, w), dtype=torch.bool)

    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    self.mask_boundary = mask.unsqueeze(0).unsqueeze(0).to(dev)  # (1,1,H,W)

  def __call__(self, u, u_prev):
    return F.mse_loss(u, fd2d.fdtd_step(u_prev, u, self.rho, self.mu, self.gamma, self.mask_boundary))
