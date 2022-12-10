import math

import torch
from torch import Tensor, distributions, nn

from src.utils.helpers import to_cuda_variable
from src.utils.model import Model


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderTransformer(Model):
    def __init__(self,
                 note_embedding_dim,
                 rnn_hidden_size,
                 num_layers,
                 num_notes,
                 dropout,
                 bidirectional,
                 z_dim,
                 transformer_layer_class,
                 transformer_class):
        super(EncoderTransformer, self).__init__()
        self.bidirectional = False
        self.num_directions = 2 if bidirectional else 1
        self.note_embedding_dim = note_embedding_dim
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.dropout = dropout
        self.transformer_layer_class = transformer_layer_class
        self.transformer_class = transformer_class

        self.pos_encoder = PositionalEncoding(note_embedding_dim, dropout)

        encoder_layer = self.transformer_layer_class(
            d_model=note_embedding_dim,
            nhead=4,
            dim_feedforward=256 * 8,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = self.transformer_class(
            encoder_layer,
            num_layers = self.num_layers,
        )

        self.num_notes = num_notes
        self.note_embedding_layer = nn.Embedding(self.num_notes,
                                                 self.note_embedding_dim)

        self.num_linear_layers = 256
        self.linear_mean = nn.Sequential(
            nn.Linear(self.num_directions * self.num_linear_layers,
                      self.num_directions * (256 // 2)),
            nn.SELU(),
            nn.Linear(self.num_directions * (256 // 2), z_dim)
        )

        self.linear_log_std = nn.Sequential(
            nn.Linear(self.num_directions * self.num_linear_layers,
                      self.num_directions * (256 // 2)),
            nn.SELU(),
            nn.Linear(self.num_directions * (256 // 2), z_dim)
        )

        self.xavier_initialization()

    def __repr__(self):
        """
        String Representation of class
        :return: string, class representation
        """
        return f'Encoder(' \
               f'{self.note_embedding_dim},' \
               f'{self.rnn_class},' \
               f'{self.num_layers},' \
               f'{self.rnn_hidden_size},' \
               f'{self.dropout},' \
               f'{self.bidirectional},' \
               f'{self.z_dim},' \
               f')'

    def embed_forward(self, score_tensor):
        """
        Performs the forward pass of the embedding layer
        :param score_tensor: torch tensor,
                (batch_size, measure_seq_len)
        :return: torch tensor,
                (batch_size, measure_seq_len, embedding_size)
        """
        x = self.note_embedding_layer(score_tensor)
        return x

    def forward(self, score_tensor):
        """
        Performs the forward pass of the src, overrides torch method
        :param score_tensor: torch Variable
                (batch_size, measure_seq_len)
        :return: torch distribution
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nan_check = torch.isnan(param.data)
                if nan_check.nonzero().size(0) > 0:
                    print('Encoder has become nan')
                    raise ValueError

        batch_size, measure_seq_len = score_tensor.size()

        # embed score
        embedded_seq = self.embed_forward(score_tensor=score_tensor) * math.sqrt(self.note_embedding_dim)
        embedded_seq = self.pos_encoder(embedded_seq)

        # pass through RNN
        out = self.transformer(embedded_seq)
        out = out.transpose(0, 1).contiguous()
        out = out.view(batch_size, -1)


        # compute distribution parameters
        z_mean = self.linear_mean(out)
        z_log_std = self.linear_log_std(out)

        z_distribution = distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
        return z_distribution
