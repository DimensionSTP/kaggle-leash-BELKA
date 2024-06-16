from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from transformers import AutoModel, AutoConfig


class HuggingFaceModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        dropout_ratio: float,
        num_labels: int,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(
            pretrained_model_name,
            output_hidden_states=False,
        )
        self.dropout = nn.Dropout(
            p=dropout_ratio,
        )
        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
        )
        self.classifier = nn.Linear(
            self.model_config.hidden_size,
            self.model_config.num_labels,
        )

    def forward(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        model_output = self.model(**encoded)
        logits = self.classifier(
            self.dropout(
                model_output.pooler_output,
            )
        )
        loss = F.binary_cross_entropy_with_logits(
            logits,
            encoded["labels"],
        )
        output = {
            "logits": logits,
            "loss": loss,
        }
        return output
