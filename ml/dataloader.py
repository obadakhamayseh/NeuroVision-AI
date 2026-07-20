

from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import Config
from dataset import BrainTumorDataset
from utils import get_logger

class DataLoaderBundle:

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def __repr__(self) -> str:
        return (
            f"DataLoaderBundle("
            f"train={len(self.train_loader.dataset)} samples, "  
            f"val={len(self.val_loader.dataset)} samples, "       
            f"test={len(self.test_loader.dataset)} samples)"      
        )

    def summary(self) -> None:
        
        logger = get_logger()
        logger.info("=" * 60)
        logger.info("DATA LOADER SUMMARY")
        logger.info("=" * 60)
        for name, loader in [
            ("Train", self.train_loader),
            ("Val  ", self.val_loader),
            ("Test ", self.test_loader),
        ]:
            n = len(loader.dataset)  
            bs = loader.batch_size
            n_batches = len(loader)
            logger.info(
                "  %s → %5d samples  |  batch_size=%d  |  %d batches",
                name, n, bs, n_batches,
            )
        logger.info("=" * 60)

class DataLoaderFactory:

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = get_logger()

    def build(
        self,
        train_dataset: BrainTumorDataset,
        val_dataset: BrainTumorDataset,
        test_dataset: BrainTumorDataset,
        use_weighted_sampler: bool = False,
    ) -> DataLoaderBundle:
        
        train_loader = self._build_train_loader(
            train_dataset, use_weighted_sampler
        )
        val_loader = self._build_eval_loader(val_dataset, split="val")
        test_loader = self._build_eval_loader(test_dataset, split="test")

        bundle = DataLoaderBundle(train_loader, val_loader, test_loader)
        bundle.summary()
        return bundle

    def _build_train_loader(
        self,
        dataset: BrainTumorDataset,
        use_weighted_sampler: bool,
    ) -> DataLoader:
        
        sampler: Optional[WeightedRandomSampler] = None
        shuffle = self.cfg.SHUFFLE_TRAIN

        if use_weighted_sampler:

            shuffle = False
            sample_weights = dataset.get_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,  
            )
            self.logger.info(
                "WeightedRandomSampler attached to training loader."
            )

        pin_mem = self.cfg.PIN_MEMORY and torch.cuda.is_available()

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=pin_mem,
            drop_last=self.cfg.DROP_LAST_TRAIN,

            persistent_workers=(self.cfg.NUM_WORKERS > 0),
        )
        self.logger.info(
            "Train DataLoader: %d samples, batch=%d, workers=%d, "
            "pin_memory=%s, drop_last=%s, sampler=%s.",
            len(dataset),
            self.cfg.BATCH_SIZE,
            self.cfg.NUM_WORKERS,
            pin_mem,
            self.cfg.DROP_LAST_TRAIN,
            "WeightedRandom" if sampler else "None",
        )
        return loader

    def _build_eval_loader(
        self,
        dataset: BrainTumorDataset,
        split: str = "eval",
    ) -> DataLoader:
        
        pin_mem = self.cfg.PIN_MEMORY and torch.cuda.is_available()
        shuffle = (
            self.cfg.SHUFFLE_VAL if split == "val" else self.cfg.SHUFFLE_TEST
        )
        drop_last = (
            self.cfg.DROP_LAST_VAL if split == "val" else self.cfg.DROP_LAST_TEST
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=pin_mem,
            drop_last=drop_last,
            persistent_workers=(self.cfg.NUM_WORKERS > 0),
        )
        self.logger.info(
            "%s DataLoader: %d samples, batch=%d, workers=%d, pin_memory=%s.",
            split.capitalize(),
            len(dataset),
            self.cfg.BATCH_SIZE,
            self.cfg.NUM_WORKERS,
            pin_mem,
        )
        return loader
