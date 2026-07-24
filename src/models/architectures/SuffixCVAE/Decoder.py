import torch
from torch import nn
import torch.nn.functional as F

from src.configs.schema import DecoderConfig

class Decoder(nn.Module):
    """PyTorch model for the decoder of a VAE (c_dim = 0) or Conditional VAE (c_dim > 0)"""
    def __init__(self, config: DecoderConfig, activity_embedding, resource_embedding):
        super().__init__()

        self.config = config
        self.activity_embedding = activity_embedding
        self.resource_embedding = resource_embedding

        self.dropout = nn.Dropout(p=config.dropout_p)

        trace_repr_dim = config.trace_repr_dim

        # c_dim is 0 when the model is not conditional
        self.to_trace_repr = nn.Linear(config.z_dim+config.c_dim, trace_repr_dim)

        self.attribute_decoders = nn.ModuleDict()
        for trace_attribute in config.trace_attributes:
            if trace_attribute['type'] == 'categorical':
                out_dim = len(trace_attribute['possible_values'])
            elif trace_attribute['type'] == 'numerical':
                out_dim = 1
            else:
                raise Exception(f'Unknown trace attribute type: {trace_attribute["type"]}')

            self.attribute_decoders[trace_attribute['name']] = nn.Sequential(
                nn.Linear(trace_repr_dim, trace_repr_dim//2),
                nn.ReLU(),
                self.dropout,
                nn.Linear(trace_repr_dim//2, out_dim),
            )

        self.activity_lstm = nn.LSTM(
            input_size=trace_repr_dim+config.activity_embedding_dim,
            hidden_size=config.control_flow_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout_p if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.resource_lstm = nn.LSTM(
            input_size=trace_repr_dim+config.activity_embedding_dim+config.resource_embedding_dim,
            hidden_size=config.control_flow_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout_p if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.ts_lstm = nn.LSTM(
            input_size=trace_repr_dim+config.activity_embedding_dim+1,
            hidden_size=config.control_flow_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.dropout_p if config.num_lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.activity_head = nn.Linear(config.control_flow_dim, config.num_activities)
        self.ts_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(config.control_flow_dim, config.control_flow_dim // 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(config.control_flow_dim // 2, 1),
        )
        self.resource_head = nn.Linear(config.control_flow_dim, config.num_resources)


    def _decode_attributes(self, trace_repr):
        """Reconstruct the trace attributes from the trace representation"""
        attributes_rec = {}
        if not self.config.has_trace_attributes:
            return attributes_rec

        for trace_attribute in self.spec.trace_attributes:
            attribute = self.attribute_decoders[trace_attribute['name']](trace_repr)   # (batch_size, out_dim)
            # Categorical attributes are reconstructed as a distribution over their possible values,
            # numerical attributes as a single value in [0, 1]
            attribute = F.softmax(attribute, dim=1) if trace_attribute['type'] == 'categorical' else F.sigmoid(attribute)
            attributes_rec[trace_attribute['name']] = attribute

        return attributes_rec

    def forward(self, z, c=None):
        """
        Compute the forward pass of the decoder.
        Args:
            - z: latent representation sampled from the encoder's distribution
            - c: conditional label, if the model is conditional
        Returns:
            - attributes_rec: reconstructed trace attributes, one tensor per attribute
            - activities_rec: reconstructed activities, one probability distribution per event
            - ts_rec: reconstructed timestamps, one value per event
            - resources_rec: reconstructed resources, one probability distribution per event
        """
        spec = self.spec

        # Concatenate the conditional label
        if spec.is_conditional:
            z = torch.cat((z, c), dim=1)

        # Expand the latent representation back into a trace representation
        trace_repr = self.dropout(F.relu(self.to_trace_repr(z)))                # (batch_size, trace_repr_dim)

        # Reconstruct the trace attributes
        attributes_rec = self._decode_attributes(trace_repr)

        # The constants below follow the batch, so the decoder needs no device of its own
        batch_size = trace_repr.shape[0]

        def initial_hidden():
            return (
                trace_repr.new_zeros(spec.num_lstm_layers, batch_size, spec.control_flow_dim),
                trace_repr.new_zeros(spec.num_lstm_layers, batch_size, spec.control_flow_dim),
            )

        # Seed the autoregressive generation with the end-of-trace activity and resource, and a zero timestamp
        eot_activity_embed = self.activity_embedding(torch.tensor([spec.num_activities-1], dtype=torch.int64, device=trace_repr.device)).repeat(batch_size, 1)   # (batch_size, activity_embedding_dim)
        eot_resource_embed = self.resource_embedding(torch.tensor([spec.num_resources-1], dtype=torch.int64, device=trace_repr.device)).repeat(batch_size, 1)   # (batch_size, resource_embedding_dim)
        ts_initial = trace_repr.new_zeros(batch_size, 1)                        # (batch_size, 1)

        activity_lstm_input = torch.cat((trace_repr, eot_activity_embed), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim)
        activity_lstm_hidden = initial_hidden()

        resource_lstm_input = torch.cat((trace_repr, eot_activity_embed, eot_resource_embed), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim + resource_embedding_dim)
        resource_lstm_hidden = initial_hidden()

        ts_lstm_input = torch.cat((trace_repr, eot_activity_embed, ts_initial), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim + 1)
        ts_lstm_hidden = initial_hidden()
        decoder_activity_outputs, decoder_resource_outputs, decoder_ts_outputs = [], [], []

        activity_rec, resource_rec, ts_rec = eot_activity_embed, eot_resource_embed, ts_initial
        for _ in range(spec.max_trace_length):
            # Run the activity LSTM one step forward
            activity_lstm_output, activity_lstm_hidden = self.activity_lstm(activity_lstm_input, activity_lstm_hidden)   # (batch_size, 1, control_flow_dim)

            # Decode the LSTM output into an activity, then re-embed it to feed the following steps
            activity_rec = self.activity_head(activity_lstm_output)             # (batch_size, 1, num_activities)
            decoder_activity_outputs.append(activity_rec)
            activity_rec = activity_rec.view(activity_rec.shape[0], -1).argmax(dim=1)   # (batch_size,)
            activity_rec = self.activity_embedding(activity_rec)                # (batch_size, activity_embedding_dim)

            # Run the resource LSTM one step forward, conditioned on the activity just decoded
            resource_lstm_input = torch.cat((trace_repr, activity_rec, resource_rec), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim + resource_embedding_dim)
            resource_lstm_output, resource_lstm_hidden = self.resource_lstm(resource_lstm_input, resource_lstm_hidden)   # (batch_size, 1, control_flow_dim)

            # Decode the LSTM output into a resource, then re-embed it to feed the following steps
            resource_rec = self.resource_head(resource_lstm_output)             # (batch_size, 1, num_resources)
            decoder_resource_outputs.append(resource_rec)
            resource_rec = resource_rec.view(resource_rec.shape[0], -1).argmax(dim=1)   # (batch_size,)
            resource_rec = self.resource_embedding(resource_rec)                # (batch_size, resource_embedding_dim)

            # Run the timestamp LSTM one step forward, conditioned on the activity just decoded
            ts_lstm_input = torch.cat((trace_repr, activity_rec, ts_rec), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim + 1)
            ts_lstm_output, ts_lstm_hidden = self.ts_lstm(ts_lstm_input, ts_lstm_hidden)   # (batch_size, 1, control_flow_dim)

            # Decode the LSTM output into a timestamp
            ts_rec = self.ts_head(ts_lstm_output)                               # (batch_size, 1)
            decoder_ts_outputs.append(ts_rec)

            # Feed the activity decoded at step t as input to step t+1
            activity_lstm_input = torch.cat((trace_repr, activity_rec), dim=1).view(batch_size, 1, -1)   # (batch_size, 1, trace_repr_dim + activity_embedding_dim)

        # Stack the per-step outputs into a single tensor per component
        activities_rec = torch.cat(decoder_activity_outputs, dim=1)             # (batch_size, max_trace_length, num_activities)
        activities_rec = F.softmax(activities_rec, dim=2)

        resources_rec = torch.cat(decoder_resource_outputs, dim=1)              # (batch_size, max_trace_length, num_resources)
        resources_rec = F.softmax(resources_rec, dim=2)

        ts_rec = torch.cat(decoder_ts_outputs, dim=1)                           # (batch_size, max_trace_length)

        return attributes_rec, activities_rec, ts_rec, resources_rec
