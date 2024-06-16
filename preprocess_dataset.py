from typing import List

import polars as pl
from rdkit import Chem

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
)
def preprocess_dataset(
    config: DictConfig,
) -> None:
    if config.split == "train":
        df = pl.read_parquet(
            f"{config.connected_dir}/data/{config.split}.parquet",
            columns=[
                config.data_column_name,
                config.label_type_column_name,
                config.result_column_name,
            ],
            n_rows=config.num_rows,
        )
    elif config.split == "test":
        df = pl.read_parquet(
            f"{config.connected_dir}/data/{config.split}.parquet",
            columns=[
                config.data_column_name,
            ],
        )
    else:
        raise ValueError(f"Invalid split argument: {config.split}")

    if config.split == "train":
        dfs = []
        for i, target_column_name in enumerate(config.target_column_names):
            sub_df = df[i::3]
            sub_df = sub_df.rename(
                {
                    config.result_column_name: target_column_name,
                }
            )
            if i == 0:
                dfs.append(
                    sub_df.drop(
                        [
                            config.order_column_name,
                            config.label_type_column_name,
                        ]
                    )
                )
            else:
                dfs.append(sub_df[[target_column_name]])
        df = pl.concat(dfs, how="horizontal")
        df = df.sample(n=config.num_samples)

    if config.split == "test":
        for target_column_name in config.target_column_names:
            df = df.with_columns(
                pl.lit(
                    0,
                    dtype=pl.Int64,
                ).alias(target_column_name)
            )
        df = df.with_columns(
            pl.lit(
                0,
                dtype=pl.Int64,
            ).alias(config.result_column_name)
        )

    def normalize(
        x: str,
    ) -> str:
        mol = Chem.MolFromSmiles(x)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )
        return smiles

    df = df.with_columns(
        pl.col(config.data_column_name).map_elements(
            normalize,
            return_dtype=pl.Utf8,
        )
    )

    if config.split == "train":
        df.write_parquet(
            f"{config.connected_dir}/data/preprocessed_{config.split}_{config.num_samples}.parquet"
        )
    if config.split == "test":
        df.write_parquet(
            f"{config.connected_dir}/data/preprocessed_{config.split}.parquet"
        )


if __name__ == "__main__":
    preprocess_dataset()
