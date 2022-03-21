from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler

from .dataset import Emotion_Dataset

class DataPipeline(LightningDataModule):
    def __init__(self, root, data_type, cv_split, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = Emotion_Dataset        
        self.root = root
        self.data_type = data_type
        self.cv_split = cv_split
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                root = self.root,
                split = "TRAIN",
                data_type = self.data_type,
                cv_split = self.cv_split   
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                root = self.root,
                split = "VALID",
                data_type = self.data_type,
                cv_split = self.cv_split   
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                root = self.root,
                split = "TEST",
                data_type = self.data_type,
                cv_split = self.cv_split   
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, root, split, data_type, cv_split) -> Dataset:
        dataset = dataset_builder(root, split, data_type, cv_split)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, **kwargs) -> DataLoader:
        all_indices = list(range(len(dataset)))
        sampler = SubsetRandomSampler(all_indices)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)
        return DataLoader(
            dataset,
            sampler=batch_sampler,
            batch_size=None,
            num_workers = num_workers,
            persistent_workers=False,
            **kwargs
        )
