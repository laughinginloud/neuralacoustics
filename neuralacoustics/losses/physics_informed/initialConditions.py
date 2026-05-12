from neuralop.losses.equation_losses import ICLoss as base

class ICLoss(base):
    def __call__(self, y_pred, x, **kwargs):
        y_pred = y_pred.permute(0, 3, 1, 2)[:, None, ...]
        x      = x     .permute(0, 3, 1, 2)[:, None, ...]
        return super().__call__(y_pred, x, **kwargs)
