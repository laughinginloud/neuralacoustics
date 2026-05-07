import importlib.simple
import torch.nn.functional as F

fd2d = importlib.import_module("solvers.FDTD.2D.dampedTransverseWaveProp_linear.dampedTransverseWaveProp_linear")

# TODO: passare primo input?
class WaveEqnLoss:
  def __init__(self, w, h, mu, rho, gamma, srate, loss=F.mse_loss):
    fd2d.setupVars(w, h, mu, rho, gamma, srate)
    self.loss = loss

  def __call__(self, u):
    lhs, rhs = fd2d.eqn(u)
    return self.loss(lhs, rhs)
