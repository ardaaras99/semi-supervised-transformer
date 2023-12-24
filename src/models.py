import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.auto import AutoModel
from vector_vis_graph.vvg import natural_vvg


def get_visibility_graph(multivariate_tensor: torch.Tensor) -> torch.Tensor:
    """
    :param multivariate_tensor: L, D
    :return: L, L
    """
    return torch.rand((multivariate_tensor.shape[0], multivariate_tensor.shape[0])).to("mps")


class BaseClassifier(nn.Module):
    def __init__(self, model_name, n_class, p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # fan_out = list(self.encoder.modules())[-2].out_features
        # TODO: check if this is the correct way to get the fan_out
        fan_out = 768
        self.linear = nn.Linear(fan_out, n_class)
        self.dropout = nn.Dropout(p)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.encoder(input_ids, attention_mask)[0][:, 0]
        out = self.dropout(cls_feats)
        logits = self.linear(out)
        out = F.log_softmax(logits, dim=1)
        return out


class VGClassifier(nn.Module):
    def __init__(self, model_name, n_class, p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # fan_out = list(self.encoder.modules())[-2].out_features
        fan_out = 768
        self.W = torch.nn.Parameter(torch.randn(fan_out, n_class))
        self.dropout = nn.Dropout(p)

    def forward(self, input_ids, attention_mask):
        token_embs = self.encoder(input_ids, attention_mask)[0]  # B, L, D
        out = torch.zeros(token_embs.shape[0], self.W.shape[-1], device=torch.device("mps"))  # B, n_class
        for i, example in enumerate(token_embs):
            vg = natural_vvg(
                example.to("cpu"),
            ).to("mps")
            vg = vg + vg.T
            # retrieve CLS token
            out[i] = torch.linalg.multi_dot([vg, example, self.W])[0]

        out = F.log_softmax(out, dim=1)
        return out
