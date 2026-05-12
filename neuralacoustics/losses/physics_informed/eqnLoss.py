import torch
import importlib
import torch.nn.functional as F

from neuralacoustics.utils import getConfigParser

class EqnLoss:
  def __init__(self, prj_root, caller, loss=F.mse_loss):
    # TODO: parsing config

    self.vars = {}

    # DA CONFIG DATASET GENERATOR
    self.vars['mu']    = 0.1
    self.vars['rho']   = 0.5
    self.vars['gamma'] = 0
    self.vars['w']     = 64
    self.vars['h']     = 64
    self.vars['srate'] = 44100

    self.vars['batch_size'] = 20  # DA CONFIG DEL TRAINER (non dataset generator)

    self.n_steps = 20  # TODO: controllare a chi serve (solo al training?)

    # TODO: sostituire solver con function pointer
    self.solver = importlib.import_module("solvers.FDTD.2D.dampedTransverseWaveProp_linear.dampedTransverseWaveProp_linear")

    self.solver.setupVars(self.vars)
    self.loss = loss

  def __call__(self, u):
    _bs, _x, _y, t = u.shape

    loss = torch.empty(t)

    # TODO: tradurre
    # il for è per gestire sia il caso single_step che il caso multiple_step
    # nel caso single step ci aspettiamo una soluzione alla volta, e sarà il train a fare l'average
    # nel caso multiple step, riceviamo più soluzioni e facciamo la prediction
    for i in range(0, t):
      lhs, rhs = self.solver.eqn(u[..., i:i+1])
      loss[i] = self.loss(lhs, rhs)   # TODO: controllare se numero puro o insieme di dim batch

    return torch.mean(loss)

  def resetSeq(self):
    self.solver.resetSeq()

# SE SINGLE_STEP, loss
# SE MULTI_STEP, media loss
