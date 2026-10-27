import lightning as L


class LossWeightScheduler(L.Callback):
    """Linearly ramps the weight of one term of a :class:`~eotorch.models.loss.CombinedLoss`.

    The weight goes from ``start`` to ``end`` over the first ``warmup_epochs`` epochs.
    A boundary term can destabilize an undertrained network (Kervadec et al. 2019), so a
    common approach is to let the region terms (e.g. CE / Dice) converge first:

    >>> task = SegmentationTask(..., loss={"ce": 0.5, "dice": 0.5, "boundary_distance": 0.0})
    >>> trainer = L.Trainer(callbacks=[LossWeightScheduler("boundary_distance", end=0.3)])
    """

    def __init__(
        self, term: str, start: float = 0.0, end: float = 0.3, warmup_epochs: int = 30
    ) -> None:
        super().__init__()
        self.term = term
        self.start = start
        self.end = end
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        progress = min(trainer.current_epoch / max(self.warmup_epochs, 1), 1.0)
        weight = self.start + (self.end - self.start) * progress
        pl_module.criterion.weights[self.term] = weight
