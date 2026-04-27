from neuralop.losses.equation_losses import ICLoss as base

class ICLoss(base):
    def __call__(self, y_pred, x):
        return super().__call__(y_pred.permute(0, 3, 1, 2), x.permute(0, 3, 2, 1))
