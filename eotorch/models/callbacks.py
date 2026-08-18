import lightning as L


class BoundaryWeightScheduler(L.Callback):
    """
    Linearly ramps a boundary-aware criterion's ``boundary_weight`` (and the
    complementary ``region_weight = 1 - boundary_weight``) from ``start`` to ``end``
    over the first ``warmup_epochs`` epochs of training.

    An unbounded distance-weighted boundary term can destabilize an undertrained
    network (Kervadec et al. 2019, "Boundary loss for highly unbalanced
    segmentation"); letting the region term (CE/Dice) converge first and ramping
    positional pressure up afterward is the standard mitigation. Works with any
    criterion exposing mutable ``boundary_weight``/``region_weight`` attributes, e.g.
    :class:`eotorch.models.loss.MultiClassCEDiceBoundaryLoss`,
    :class:`eotorch.models.loss.MultiClassCEDiceBoundaryDistanceLoss`, or
    :class:`eotorch.models.loss.BCEDiceBoundaryLoss`.

    Args:
        start: Boundary weight at epoch 0.
        end: Boundary weight once ``warmup_epochs`` is reached.
        warmup_epochs: Number of epochs over which to ramp from ``start`` to ``end``.

    Example:
        >>> model.criterion = MultiClassCEDiceBoundaryDistanceLoss(
        ...     num_classes=3, ignore_index=0, ce_weight=0.4, dice_weight=0.6,
        ...     boundary_weight=0.0, region_weight=1.0,  # scheduler drives these
        ... )
        >>> trainer = L.Trainer(
        ...     callbacks=[BoundaryWeightScheduler(start=0.0, end=0.3, warmup_epochs=30)],
        ... )

    .. note::
       If you swap ``model.criterion`` manually as in the example above, it isn't
       captured by ``save_hyperparameters()`` and must be reapplied any time a
       fresh task instance is constructed (e.g. before ``trainer.fit`` on a new
       object) -- reusing the same live model instance across ``trainer.fit`` calls,
       as with a resume, is unaffected.
    """

    def __init__(self, start: float = 0.0, end: float = 0.3, warmup_epochs: int = 30) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        progress = min(trainer.current_epoch / max(self.warmup_epochs, 1), 1.0)
        boundary_weight = self.start + (self.end - self.start) * progress
        pl_module.criterion.boundary_weight = boundary_weight
        pl_module.criterion.region_weight = 1.0 - boundary_weight
