import torch
from torch import nn
import torch.nn.functional as F

from models.architectures.SuffixCVAE import Decoder, Encoder, Encoder
from src.configs.schema import LatentConfig, DecoderConfig, EncoderConfig

class SuffixCVAE(nn.Module):
    """
    PyTorch model for a trace-level VAE (c_dim = 0) or Conditional VAE (c_dim > 0)

    (attributes, activities) -> (attribute_embeds, event_features) -> (attribute_embeds, control_flow)
    -> trace_repr -> z -> trace_repr (reconstructed) -> (attributes_rec, event_features_rec) -> (attributes_rec, activities_rec)
    """
    def __init__(self, *,encoderConfig: EncoderConfig, decoderConfig: DecoderConfig, latentConfig: LatentConfig):
        super().__init__()

        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.latentConfig = latentConfig

        # Shared between the encoder and decoder, so owned by the VAE itself
        self.activity_embedding = nn.Embedding(
            encoderConfig.vocab_size + 1, encoderConfig.activity_embedding_dim, padding_idx=encoderConfig.vocab_size,
        )
        self.resource_embedding = nn.Embedding(
            encoderConfig.num_resources + 1, encoderConfig.resource_embedding_dim, padding_idx=encoderConfig.num_resources,
        )

        self.encoder = Encoder(encoderConfig, self.activity_embedding, self.resource_embedding)
        self.decoder = Decoder(decoderConfig, self.activity_embedding, self.resource_embedding)

    @property
    def z_dim(self) -> int:
        return self.latentConfig.latent_dim

    @property
    def is_conditional(self) -> bool:
        return self.latentConfig.condition_dim > 0

    # p(z|x,c)
    def encode(self, x, c=None):
        """Compute the mean and standard deviation of the latent distribution given the input trace and optional conditional label"""
        return self.encoder(x, c)

    # p(x|z,c)
    def decode(self, z, c=None):
        """Reconstruct the trace given the latent representation and optional conditional label"""
        return self.decoder(z, c)

    def forward(self, x, c=None):
        mean, std = self.encode(x, c)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon # reparametrization trick

        return self.decode(z, c), mean, std
