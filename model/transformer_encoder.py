import torch
from torch import nn
import torch.nn.functional as F

class VAETransformerEncoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_vae_latent, dropout=0.1, activation='relu'):
    super(VAETransformerEncoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_vae_latent = d_vae_latent
    self.dropout = dropout
    self.activation = activation

    self.tr_encoder_layer = nn.TransformerEncoderLayer(
      d_model, n_head, d_ff, dropout, activation
    )
    self.tr_encoder = nn.TransformerEncoder(
      self.tr_encoder_layer, n_layer
    )

    self.fc_mu = nn.Linear(d_model, d_vae_latent)
    self.fc_logvar = nn.Linear(d_model, d_vae_latent)

  def forward(self, x, padding_mask=None):
    out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
    hidden_out = out[0, :, :]
    mu, logvar = self.fc_mu(hidden_out), self.fc_logvar(hidden_out)

    return hidden_out, mu, logvar
