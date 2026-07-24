import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.configs.schema import EncoderConfig

class Encoder(nn.Module):
    """PyTorch model for the encoder of a VAE (c_dim = 0) or Conditional VAE (c_dim > 0)"""
    def __init__(self, config: EncoderConfig, activity_embedding: nn.Embedding, resource_embedding: nn.Embedding):
        super().__init__()
        
        self.config = config
        self.activity_embedding = activity_embedding
        self.resource_embedding = resource_embedding
        self.dropout = nn.Dropout(p=config.dropout_p)

        self.attribute_embeddings = nn.ModuleDict({
            attr['name']: nn.Embedding(len(attr['possible_values']), config.attribute_embedding_dim)
            for attr in config.trace_attributes if attr['type'] == 'categorical'
        })
        self.has_attributes = len(config.trace_attributes) > 0
        self.attribute_compressor = nn.Linear(config.total_attribute_embedding_dim, config.compressed_attribute_dim)

        

        self.trace_encoder = nn.LSTM(
            input_size=config.event_dim,
            hidden_size=config.control_flow_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout_p if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.to_mean = nn.Linear(config.trace_repr_dim + config.c_dim, config.z_dim)
        self.to_std = nn.Linear(config.trace_repr_dim + config.c_dim, config.z_dim)

    def _embed_attributes(self, attributes):
        """Embed the trace attributes and compress them to a single vector"""
        if not self.has_attributes:
            return None

        embeds = []
        for attr in self.config.trace_attributes:
            # If the attribute is categorical, use the embedding layer
            if attr['type'] == 'categorical':
                embeds.append(self.attribute_embeddings[attr['name']](attributes[attr['name']]))
            # If the attribute is numerical, just use the raw value (as a float)
            elif attr['type'] == 'numerical':
                embeds.append(attributes[attr['name']].unsqueeze(1).float())
            else:
                raise ValueError(f"Unknown trace attribute type: {attr['type']}")

        # Concatenate all attribute embeddings and compress them
        embeds = torch.cat(embeds, dim=1)
        return self.dropout(F.relu(self.attribute_compressor(embeds)))


    def _embed_control_flow(self, activities, timestamps, resources):
        """Embed the control flow of the trace: its activities, timestamps, and resources"""
        # Embed activities
        event_features = self.activity_embedding(activities)                    # (batch_size, max_trace_length, activity_embedding_dim)

        # Add a dimension (2D -> 3D) to timestamps to concatenate with event features
        timestamps = timestamps.unsqueeze(dim=2)                                # (batch_size, max_trace_length, 1)
        event_features = torch.cat((event_features, timestamps), dim=2)         # (batch_size, max_trace_length, activity_embedding_dim + 1)

        # Embed resources and concatenate with event features
        resource_embeds = self.resource_embedding(resources)                    # (batch_size, max_trace_length, resource_embedding_dim)
        event_features = torch.cat((event_features, resource_embeds), dim=2)    # (batch_size, max_trace_length, activity_embedding_dim + 1 + resource_embedding_dim)

        # Retrieve the lengths of the traces (number of events)
        trace_lengths = activities.argmax(dim=1).to('cpu')                      # (batch_size,)

        # Pack the padded sequence of event features for LSTM processing
        packed_events = pack_padded_sequence(event_features, trace_lengths, batch_first=True, enforce_sorted=False)
        # Pass the packed sequence through the LSTM to get the control flow representation
        packed_control_flow, _ = self.trace_encoder(packed_events)
        # Unpack the packed sequence back to a padded sequence
        control_flow, _ = pad_packed_sequence(packed_control_flow, batch_first=True)

        # Select the last hidden state of the LSTM for each trace, which corresponds to the control flow representation of the entire trace
        control_flow = control_flow[torch.arange(control_flow.shape[0]), trace_lengths-1]   # (batch_size, hidden_size)
        control_flow = self.dropout(control_flow)

        return control_flow

    def forward(self, x, c=None):
        """
        Compute the forward pass of the encoder.
        Args:
            - x: tuple of (attributes, activities, timestamps, resources)
            - c: conditional label, if the model is conditional
        Returns:
            - mean: mean of the latent distribution
            - std: standard deviation of the latent distribution
        """
        # Unpack the input
        attributes, activities, timestamps, resources = x

        # Embed the trace attributes and control flow
        attribute_embeds = self._embed_attributes(attributes)
        control_flow = self._embed_control_flow(activities, timestamps, resources)

        # Concatenate the attribute embeddings and control flow representation to form the trace representation
        trace_representation = torch.cat((attribute_embeds, control_flow), dim=1)

        # concat conditional label
        if self.spec.is_conditional:
            trace_representation = torch.cat((trace_representation, c), dim=1)
        # Compute the mean and standard deviation of the latent distribution
        return self.to_mean(trace_representation), self.to_std(trace_representation)
