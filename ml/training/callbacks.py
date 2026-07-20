

import logging
from typing import Dict

logger = logging.getLogger("brain_tumor_pipeline")

class Callback:

    def on_epoch_end(self, epoch: int, metrics: Dict) -> None:
        pass

    def on_train_begin(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass

class EarlyStopping(Callback):

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "loss",
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode

        self._counter: int = 0
        self._best_value: float = float("inf") if mode == "min" else float("-inf")
        self._stopped: bool = False

    def __call__(self, metrics: Dict) -> bool:
        
        current = metrics.get(self.monitor)
        if current is None:
            logger.warning(
                "EarlyStopping: monitor key \'%s\' not found in metrics.",
                self.monitor,
            )
            return False

        improved = (
            current < self._best_value - self.min_delta
            if self.mode == "min"
            else current > self._best_value + self.min_delta
        )

        if improved:
            self._best_value = current
            self._counter = 0
        else:
            self._counter += 1
            logger.debug(
                "EarlyStopping: no improvement for %d / %d epoch(s). "
                "Best %s=%.6f  Current=%.6f",
                self._counter, self.patience,
                self.monitor, self._best_value, current,
            )

        if self._counter >= self.patience:
            self._stopped = True
            logger.info(
                "EarlyStopping triggered: %s did not improve for %d epochs.",
                self.monitor, self.patience,
            )
            return True

        return False

    def reset(self) -> None:
        
        self._counter = 0
        self._best_value = float("inf") if self.mode == "min" else float("-inf")
        self._stopped = False
