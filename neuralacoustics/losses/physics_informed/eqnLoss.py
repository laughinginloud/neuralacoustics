import torch
import importlib
import torch.nn.functional as F

from neuralacoustics.utils import getConfigParser, openConfig

class EqnLoss:
  def __init__(self, prj_root, caller, dev, loss=F.mse_loss):
    self.dev = dev
    self.loss = loss

    config, _ = getConfigParser(prj_root, caller)

    self.vars = {'batch_size': config['training'].getint('batch_size')}
    
    config = config['dataset_generation'].get('dataset_generator')
    config = config.replace('PRJ_ROOT', prj_root)
    config = config + '/' + config.split('/')[-1] + '.ini'
    config = openConfig(config, caller)

    self.vars |= dict(config.items('numerical_model_parameters'))
    # self.vars |= dict(config.items('dataset_generator_parameters'))

    for k, v in self.vars.items():
      try:
        tmp = int(v)
        self.vars[k] = tmp
      except ValueError:
        try:
          tmp = float(v)
          self.vars[k] = tmp
        except ValueError:
          pass

    config = config['dataset_generator_parameters'].get('numerical_model')
    config = config.replace('PRJ_ROOT', prj_root)
    config = config + '/' + config.split('/')[-1] + '.ini'
    config = openConfig(config, caller)

    config = config['solver'].get('solver')
    config = config.split('/')
    config = '.'.join(config[1:] + [config[-1]])

    self.solver = importlib.import_module(config)
    self.solver.setupVars(self.vars | {'dev': self.dev})

  def __call__(self, u):
    time = u.shape[-1]

    loss = torch.empty(size=(time,), device=self.dev)

    # TODO: tradurre
    # il for è per gestire sia il caso single_step che il caso multiple_step
    # nel caso single step ci aspettiamo una soluzione alla volta, e sarà il train a fare l'average
    # nel caso multiple step, riceviamo più soluzioni e facciamo la prediction
    for i in range(0, time):
      lhs, rhs = self.solver.eqn(u[..., i:i+1])
      loss[i] = self.loss(lhs, rhs)   # TODO: controllare se numero puro o insieme di dim batch

    return torch.mean(loss)

  def resetSeq(self):
    self.solver.resetSeq()

# SE SINGLE_STEP, loss
# SE MULTI_STEP, media loss
