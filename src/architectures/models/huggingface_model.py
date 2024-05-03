from typing import Dict

import torch
from torch import nn

from transformers import AutoModelForImageClassification


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            output_hidden_states=False,
            ignore_mismatched_sizes=True,
        )

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        output = self.model(**encoded)
        return output
