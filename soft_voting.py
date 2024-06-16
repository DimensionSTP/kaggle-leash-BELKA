import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="voting.yaml",
)
def softly_vote_probs(
    config: DictConfig,
) -> None:
    connected_dir = config.connected_dir
    submission_file = config.submission_file
    result_column_name = config.result_column_name
    voted_file = config.voted_file
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_probs = None
    for prob_file, weight in votings.items():
        try:
            prob_df = pl.read_csv(f"{connected_dir}/submissions/{prob_file}.csv")
        except:
            raise FileNotFoundError(f"prob file {prob_file} does not exist")
        prob = prob_df[result_column_name].to_numpy()
        if weighted_probs is None:
            weighted_probs = prob * weight
        else:
            weighted_probs += prob * weight

    submission_df = pl.read_csv(submission_file)
    submission_df[result_column_name] = weighted_probs
    submission_df.write_csv(
        voted_file,
    )


if __name__ == "__main__":
    softly_vote_probs()
