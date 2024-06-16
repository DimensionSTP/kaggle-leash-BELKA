from typing import Dict, Any, List
import joblib

import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class KaggleLeashBELKADataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        split_ratio: float,
        seed: int,
        data_column_name: str,
        target_column_names: List[str],
        num_devices: int,
        batch_size: int,
        pretrained_model_name: str,
        data_max_length: int,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.data_column_name = data_column_name
        self.target_column_names = target_column_names
        self.num_devices = num_devices
        self.batch_size = batch_size
        self.data_encoder = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            use_fast=True,
        )
        dataset = self.get_dataset()
        self.datas = dataset["datas"]
        self.labels = dataset["labels"]
        self.data_max_length = data_max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Any]:
        encoded = self.encode_molecule(self.datas[idx])
        encoded["labels"] = torch.tensor(
            [self.labels[idx]],
            dtype=torch.long,
        ).squeeze(0)
        return {
            "encoded": encoded,
            "index": idx,
        }

    def get_dataset(self) -> Dict[str, List[Any]]:
        if self.split in ["train", "val"]:
            parquet_path = f"{self.data_path}/preprocessed_train_2_000_000.parquet"
            data = pl.read_parquet(parquet_path).to_pandas()
            train_data, val_data = train_test_split(
                data,
                test_size=self.split_ratio,
                random_state=self.seed,
                shuffle=True,
            )
            if self.split == "train":
                data = train_data
            else:
                data = val_data
        elif self.split == "test":
            parquet_path = f"{self.data_path}/preprocessed_{self.split}.parquet"
            data = pl.read_parquet(parquet_path).to_pandas()
        elif self.split == "predict":
            parquet_path = f"{self.data_path}/preprocessed_test.parquet"
            data = pl.read_parquet(parquet_path).to_pandas()
            if self.num_devices > 1:
                last_row = data.iloc[-1]
                total_batch_size = self.num_devices * self.batch_size
                remainder = (len(data) % total_batch_size) % self.num_devices
                if remainder != 0:
                    num_dummies = self.num_devices - remainder
                    repeated_rows = pd.DataFrame([last_row] * num_dummies)
                    repeated_rows.reset_index(
                        drop=True,
                        inplace=True,
                    )
                    data = pd.concat(
                        [
                            data,
                            repeated_rows,
                        ],
                        ignore_index=True,
                    )
        else:
            raise ValueError(f"Inavalid split: {self.split}")

        datas = data[self.data_column_name].to_numpy()
        labels = data[self.target_column_names].to_numpy()
        return {
            "datas": datas,
            "labels": labels,
        }

    def encode_molecule(
        self,
        data: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.data_encoder(
            data,
            padding="max_length",
            max_length=self.data_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return encoded
