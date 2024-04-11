from copy import deepcopy
from typing import Any, Dict, List, Optional, Type
# from pytorch_lightning.loops.fit_loop import _FitLoop
# from pytorch_lightning.trainer.states import TrainerFn
# from torch.utils.data.dataset import Dataset, Subset

from lightning.pytorch.loops.fit_loop import _FitLoop
from lightning.pytorch.trainer.states import TrainerFn

# from sklearn.model_selection import KFold


class KFoldLoop(_FitLoop):
    def __init__(self, num_folds: int, trainer, min_epoch=0, max_epochs=None) -> None:
        super().__init__(trainer,  min_epoch, max_epochs)
        self.num_folds = num_folds
        self.current_fold: int = 0

    # @property
    # def done(self) -> bool:
    #     return self.current_fold >= self.num_folds
    
    def run(self):
        # while self.current_fold < self.num_folds:
        #     self.fit_loop.run()
        #     self.trainer.test_loop.run()
        #     self.current_fold += 1
        self.setup_data()
        if self.skip:
            return
        self.reset()
        self.on_run_start()
        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        self.on_run_end()
        # 如果我能成功跑两次，那么循环应该可以，如果循环可以，那么加条件也可以
        self.setup_data()
        if self.skip:
            return
        self.reset()
        self.on_run_start()
        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False
        self.on_run_end()

        

    # def connect(self, fit_loop: _FitLoop) -> None:
    #     self.fit_loop = fit_loop

    # def reset(self) -> None:
    #     """Nothing to reset in this loop."""

    # def on_run_start(self, *args: Any, **kwargs: Any) -> None:
    #     """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
    #     model."""
    #     # assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
    #     # self.trainer.datamodule.setup_folds(self.num_folds)
    #     self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    # def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
    #     """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
    #     print(f"STARTING FOLD {self.current_fold}")
    #     # assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
    #     self.trainer.datamodule.setup_fold_index(self.current_fold)

    # def advance(self, *args: Any, **kwargs: Any) -> None:
    #     """Used to the run a fitting and testing on the current hold."""
    #     # self._reset_fitting()  # requires to reset the tracking stage.
    #     # self.setup_data()
    #     self.fit_loop.run()

    #     # self._reset_testing()  # requires to reset the tracking stage.
    #     self.setup_data()
    #     self.trainer.test_loop.run()
    #     self.current_fold += 1  # increment fold tracking number.

    # def on_advance_end(self) -> None:
    #     """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
    #     # self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
    #     # restore the original weights + optimizers and schedulers.
    #     self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
    #     self.trainer.strategy.setup_optimizers(self.trainer)
    #     self.replace(fit_loop=_FitLoop)

    # def on_run_end(self) -> None:
    #     """Used to compute the performance of the ensemble model on the test set."""
    #     # checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
    #     # voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
    #     # voting_model.trainer = self.trainer
    #     # This requires to connect the new model and move it the right device.
    #     # self.trainer.strategy.connect(voting_model)
    #     self.trainer.strategy.model_to_device()
    #     self.trainer.test_loop.run()

    # def on_save_checkpoint(self) -> Dict[str, int]:
    #     return {"current_fold": self.current_fold}

    # def on_load_checkpoint(self, state_dict: Dict) -> None:
    #     self.current_fold = state_dict["current_fold"]

    # def _reset_fitting(self) -> None:
    #     self.trainer.reset_train_dataloader()
    #     self.trainer.reset_val_dataloader()
    #     self.trainer.state.fn = TrainerFn.FITTING
    #     self.trainer.training = True

    # def _reset_testing(self) -> None:
    #     self.trainer.reset_test_dataloader()
    #     self.trainer.state.fn = TrainerFn.TESTING
    #     self.trainer.testing = True

    # def __getattr__(self, key) -> Any:
    #     # requires to be overridden as attributes of the wrapped loop are being accessed.
    #     if key not in self.__dict__:
    #         return getattr(self.fit_loop, key)
    #     return self.__dict__[key]

    # def __setstate__(self, state: Dict[str, Any]) -> None:
    #     self.__dict__.update(state)
