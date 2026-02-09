import torch
import torch.nn as nn


class ResolutionEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cross_attn_dim: int,
        target_dim: int | None = None,
        sequence_length: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cross_attn_dim = cross_attn_dim
        self.target_dim = target_dim or cross_attn_dim
        self.sequence_length = max(1, sequence_length)

        self.class_embedding = nn.Embedding(
            num_embeddings=num_classes + 1, embedding_dim=cross_attn_dim
        )

        self.proj = None
        if self.target_dim != cross_attn_dim:
            self.proj = nn.Linear(cross_attn_dim, self.target_dim)

    def set_target_dim(self, target_dim: int):
        if target_dim != self.target_dim:
            self.target_dim = target_dim
            if self.proj is None or self.proj.out_features != target_dim:
                self.proj = nn.Linear(self.cross_attn_dim, target_dim)

    def set_sequence_length(self, sequence_length: int):
        self.sequence_length = max(1, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()
        embedding = self.class_embedding(x)
        if self.proj is not None:
            embedding = self.proj(embedding)
        if self.sequence_length > 1:
            return embedding.unsqueeze(1).repeat(1, self.sequence_length, 1)
        return embedding

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, num_resolutions: int = 2):
        if pretrained_model_path.endswith(".pt") or pretrained_model_path.endswith(
            ".pth"
        ):
            state_dict = torch.load(pretrained_model_path, map_location="cpu")

            if "class_embedding.weight" in state_dict:
                num_classes = state_dict["class_embedding.weight"].shape[0]
                embedding_dim = state_dict["class_embedding.weight"].shape[1]

                model = cls(num_classes=num_classes, cross_attn_dim=embedding_dim)
                model.load_state_dict(state_dict)
                return model

        return cls(num_classes=num_classes, cross_attn_dim=768)
