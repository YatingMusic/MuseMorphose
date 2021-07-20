import torch
from torch import nn
import torch.nn.functional as F
from transformer_encoder import VAETransformerEncoder
from transformer_helpers import (
  weights_init, PositionalEncoding, TokenEmbedding, generate_causal_mask
)

class VAETransformerDecoder(nn.Module):
  def __init__(self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation='relu', cond_mode='in-attn'):
    super(VAETransformerDecoder, self).__init__()
    self.n_layer = n_layer
    self.n_head = n_head
    self.d_model = d_model
    self.d_ff = d_ff
    self.d_seg_emb = d_seg_emb
    self.dropout = dropout
    self.activation = activation
    self.cond_mode = cond_mode

    if cond_mode == 'in-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
    elif cond_mode == 'pre-attn':
      self.seg_emb_proj = nn.Linear(d_seg_emb + d_model, d_model, bias=False)

    self.decoder_layers = nn.ModuleList()
    for i in range(n_layer):
      self.decoder_layers.append(
        nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
      )

  def forward(self, x, seg_emb):
    if not hasattr(self, 'cond_mode'):
      self.cond_mode = 'in-attn'
    attn_mask = generate_causal_mask(x.size(0)).to(x.device)
    # print (attn_mask.size())

    if self.cond_mode == 'in-attn':
      seg_emb = self.seg_emb_proj(seg_emb)
    elif self.cond_mode == 'pre-attn':
      x = torch.cat([x, seg_emb], dim=-1)
      x = self.seg_emb_proj(x)

    out = x
    for i in range(self.n_layer):
      if self.cond_mode == 'in-attn':
        out += seg_emb
      out = self.decoder_layers[i](out, src_mask=attn_mask)

    return out

class MuseMorphose(nn.Module):
  def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, 
    dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
    d_vae_latent, d_embed, n_token,
    enc_dropout=0.1, enc_activation='relu',
    dec_dropout=0.1, dec_activation='relu',
    d_rfreq_emb=32, d_polyph_emb=32,
    n_rfreq_cls=8, n_polyph_cls=8,
    is_training=True, use_attr_cls=True,
    cond_mode='in-attn'
  ):
    super(MuseMorphose, self).__init__()
    self.enc_n_layer = enc_n_layer
    self.enc_n_head = enc_n_head
    self.enc_d_model = enc_d_model
    self.enc_d_ff = enc_d_ff
    self.enc_dropout = enc_dropout
    self.enc_activation = enc_activation

    self.dec_n_layer = dec_n_layer
    self.dec_n_head = dec_n_head
    self.dec_d_model = dec_d_model
    self.dec_d_ff = dec_d_ff
    self.dec_dropout = dec_dropout
    self.dec_activation = dec_activation  

    self.d_vae_latent = d_vae_latent
    self.n_token = n_token
    self.is_training = is_training

    self.cond_mode = cond_mode
    self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
    self.d_embed = d_embed
    self.pe = PositionalEncoding(d_embed)
    self.dec_out_proj = nn.Linear(dec_d_model, n_token)
    self.encoder = VAETransformerEncoder(
      enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_vae_latent, enc_dropout, enc_activation
    )

    self.use_attr_cls = use_attr_cls
    if use_attr_cls:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent + d_polyph_emb + d_rfreq_emb,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )
    else:
      self.decoder = VAETransformerDecoder(
        dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_vae_latent,
        dropout=dec_dropout, activation=dec_activation,
        cond_mode=cond_mode
      )

    if use_attr_cls:
      self.d_rfreq_emb = d_rfreq_emb
      self.d_polyph_emb = d_polyph_emb
      self.rfreq_attr_emb = TokenEmbedding(n_rfreq_cls, d_rfreq_emb, d_rfreq_emb)
      self.polyph_attr_emb = TokenEmbedding(n_polyph_cls, d_polyph_emb, d_polyph_emb)
    else:
      self.rfreq_attr_emb = None
      self.polyph_attr_emb = None

    self.emb_dropout = nn.Dropout(self.enc_dropout)
    self.apply(weights_init)
    

  def reparameterize(self, mu, logvar, use_sampling=True, sampling_var=1.):
    std = torch.exp(0.5 * logvar).to(mu.device)
    if use_sampling:
      eps = torch.randn_like(std).to(mu.device) * sampling_var
    else:
      eps = torch.zeros_like(std).to(mu.device)

    return eps * std + mu

  def get_sampled_latent(self, inp, padding_mask=None, use_sampling=False, sampling_var=0.):
    token_emb = self.token_emb(inp)
    enc_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    mu, logvar = mu.reshape(-1, mu.size(-1)), logvar.reshape(-1, mu.size(-1))
    vae_latent = self.reparameterize(mu, logvar, use_sampling=use_sampling, sampling_var=sampling_var)

    return vae_latent

  def generate(self, inp, dec_seg_emb, rfreq_cls=None, polyph_cls=None, keep_last_only=True):
    token_emb = self.token_emb(inp)
    dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))

    if rfreq_cls is not None and polyph_cls is not None:
      dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
      dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
      dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    out = self.decoder(dec_inp, dec_seg_emb_cat)
    out = self.dec_out_proj(out)

    if keep_last_only:
      out = out[-1, ...]

    return out


  def forward(self, enc_inp, dec_inp, dec_inp_bar_pos, rfreq_cls=None, polyph_cls=None, padding_mask=None):
    # [shape of enc_inp] (seqlen_per_bar, bsize, n_bars_per_sample)
    enc_bt_size, enc_n_bars = enc_inp.size(1), enc_inp.size(2)
    enc_token_emb = self.token_emb(enc_inp)

    # [shape of dec_inp] (seqlen_per_sample, bsize)
    # [shape of rfreq_cls & polyph_cls] same as above 
    # -- (should copy each bar's label to all corresponding indices)
    dec_token_emb = self.token_emb(dec_inp)

    enc_token_emb = enc_token_emb.reshape(
      enc_inp.size(0), -1, enc_token_emb.size(-1)
    )
    enc_inp = self.emb_dropout(enc_token_emb) + self.pe(enc_inp.size(0))
    dec_inp = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))

    # [shape of padding_mask] (bsize, n_bars_per_sample, seqlen_per_bar)
    # -- should be `True` for padded indices (i.e., those >= seqlen of the bar), `False` otherwise
    if padding_mask is not None:
      padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

    _, mu, logvar = self.encoder(enc_inp, padding_mask=padding_mask)
    vae_latent = self.reparameterize(mu, logvar)
    vae_latent_reshaped = vae_latent.reshape(enc_bt_size, enc_n_bars, -1)

    dec_seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_vae_latent).to(vae_latent.device)
    for n in range(dec_inp.size(1)):
      # [shape of dec_inp_bar_pos] (bsize, n_bars_per_sample + 1)
      # -- stores [[start idx of bar #1, sample #1, ..., start idx of bar #K, sample #1, seqlen of sample #1], [same for another sample], ...]
      for b, (st, ed) in enumerate(zip(dec_inp_bar_pos[n, :-1], dec_inp_bar_pos[n, 1:])):
        dec_seg_emb[st:ed, n, :] = vae_latent_reshaped[n, b, :]

    if rfreq_cls is not None and polyph_cls is not None and self.use_attr_cls:
      dec_rfreq_emb = self.rfreq_attr_emb(rfreq_cls)
      dec_polyph_emb = self.polyph_attr_emb(polyph_cls)
      dec_seg_emb_cat = torch.cat([dec_seg_emb, dec_rfreq_emb, dec_polyph_emb], dim=-1)
    else:
      dec_seg_emb_cat = dec_seg_emb

    dec_out = self.decoder(dec_inp, dec_seg_emb_cat)
    dec_logits = self.dec_out_proj(dec_out)

    return mu, logvar, dec_logits

  def compute_loss(self, mu, logvar, beta, fb_lambda, dec_logits, dec_tgt):
    recons_loss = F.cross_entropy(
      dec_logits.view(-1, dec_logits.size(-1)), dec_tgt.contiguous().view(-1), 
      ignore_index=self.n_token - 1, reduction='mean'
    ).float()

    kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
    kl_before_free_bits = kl_raw.mean()
    kl_after_free_bits = kl_raw.clamp(min=fb_lambda)
    kldiv_loss = kl_after_free_bits.mean()

    return {
      'beta': beta,
      'total_loss': recons_loss + beta * kldiv_loss,
      'kldiv_loss': kldiv_loss,
      'kldiv_raw': kl_before_free_bits,
      'recons_loss': recons_loss
    }