import sys, os, time
sys.path.append('./model')

from model.musemorphose import MuseMorphose
from dataloader import REMIFullSongTransformerDataset
from torch.utils.data import DataLoader

from utils import pickle_load
from torch import nn, optim
import torch
import numpy as np

import yaml
config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
trained_steps = config['training']['trained_steps']
lr_decay_steps = config['training']['lr_decay_steps']
lr_warmup_steps = config['training']['lr_warmup_steps']
no_kl_steps = config['training']['no_kl_steps']
kl_cycle_steps = config['training']['kl_cycle_steps']
kl_max_beta = config['training']['kl_max_beta']
free_bit_lambda = config['training']['free_bit_lambda']
max_lr, min_lr = config['training']['max_lr'], config['training']['min_lr']

ckpt_dir = config['training']['ckpt_dir']
params_dir = os.path.join(ckpt_dir, 'params/')
optim_dir = os.path.join(ckpt_dir, 'optim/')
pretrained_params_path = config['model']['pretrained_params_path']
pretrained_optim_path = config['model']['pretrained_optim_path']
ckpt_interval = config['training']['ckpt_interval']
log_interval = config['training']['log_interval']
val_interval = config['training']['val_interval']
constant_kl = config['training']['constant_kl']

recons_loss_ema = 0.
kl_loss_ema = 0.
kl_raw_ema = 0.

def log_epoch(log_file, log_data, is_init=False):
  if is_init:
    with open(log_file, 'w') as f:
      f.write('{:4} {:8} {:12} {:12} {:12} {:12}\n'.format('ep', 'steps', 'recons_loss', 'kldiv_loss', 'kldiv_raw', 'ep_time'))

  with open(log_file, 'a') as f:
    f.write('{:<4} {:<8} {:<12} {:<12} {:<12} {:<12}\n'.format(
      log_data['ep'], log_data['steps'], round(log_data['recons_loss'], 5), round(log_data['kldiv_loss'], 5), round(log_data['kldiv_raw'], 5), round(log_data['time'], 2)
    ))

def beta_cyclical_sched(step):
  step_in_cycle = (step - 1) % kl_cycle_steps
  cycle_progress = step_in_cycle / kl_cycle_steps

  if step < no_kl_steps:
    return 0.
  if cycle_progress < 0.5:
    return kl_max_beta * cycle_progress * 2.
  else:
    return kl_max_beta

def compute_loss_ema(ema, batch_loss, decay=0.95):
  if ema == 0.:
    return batch_loss
  else:
    return batch_loss * (1 - decay) + ema * decay

def train_model(epoch, model, dloader, dloader_val, optim, sched):
  model.train()

  print ('[epoch {:03d}] training ...'.format(epoch))
  print ('[epoch {:03d}] # batches = {}'.format(epoch, len(dloader)))
  st = time.time()

  for batch_idx, batch_samples in enumerate(dloader):
    model.zero_grad()
    batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
    batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
    batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
    batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
    batch_inp_lens = batch_samples['length']
    batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
    batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
    batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)

    global trained_steps
    trained_steps += 1

    mu, logvar, dec_logits = model(
      batch_enc_inp, batch_dec_inp, 
      batch_inp_bar_pos, batch_rfreq_cls, batch_polyph_cls,
      padding_mask=batch_padding_mask
    )

    if not constant_kl:
      kl_beta = beta_cyclical_sched(trained_steps)
    else:
      kl_beta = kl_max_beta
    losses = model.compute_loss(mu, logvar, kl_beta, free_bit_lambda, dec_logits, batch_dec_tgt)
    
    # anneal learning rate
    if trained_steps < lr_warmup_steps:
      curr_lr = max_lr * trained_steps / lr_warmup_steps
      optim.param_groups[0]['lr'] = curr_lr
    else:
      sched.step(trained_steps - lr_warmup_steps)

    # clip gradient & update model
    losses['total_loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()

    global recons_loss_ema, kl_loss_ema, kl_raw_ema
    recons_loss_ema = compute_loss_ema(recons_loss_ema, losses['recons_loss'].item())
    kl_loss_ema = compute_loss_ema(kl_loss_ema, losses['kldiv_loss'].item())
    kl_raw_ema = compute_loss_ema(kl_raw_ema, losses['kldiv_raw'].item())

    print (' -- epoch {:03d} | batch {:03d}: len: {}\n\t * loss = (RC: {:.4f} | KL: {:.4f} | KL_raw: {:.4f}), step = {}, beta: {:.4f} time_elapsed = {:.2f} secs'.format(
      epoch, batch_idx, batch_inp_lens, recons_loss_ema, kl_loss_ema, kl_raw_ema, trained_steps, kl_beta, time.time() - st
    ))

    if not trained_steps % log_interval:
      log_data = {
        'ep': epoch,
        'steps': trained_steps,
        'recons_loss': recons_loss_ema,
        'kldiv_loss': kl_loss_ema,
        'kldiv_raw': kl_raw_ema,
        'time': time.time() - st
      }
      log_epoch(
        os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
      )

    if not trained_steps % val_interval:
      vallosses = validate(model, dloader_val)
      with open(os.path.join(ckpt_dir, 'valloss.txt'), 'a') as f:
        f.write('[step {}] RC: {:.4f} | KL: {:.4f} | [val] | RC: {:.4f} | KL: {:.4f}\n'.format(
          trained_steps, 
          recons_loss_ema, 
          kl_raw_ema,
          np.mean(vallosses[0]),
          np.mean(vallosses[1])
        ))
      model.train()

    if not trained_steps % ckpt_interval:
      torch.save(model.state_dict(),
        os.path.join(params_dir, 'step_{:d}-RC_{:.3f}-KL_{:.3f}-model.pt'.format(
            trained_steps,
            recons_loss_ema, 
            kl_raw_ema
          ))
      )
      torch.save(optim.state_dict(),
        os.path.join(optim_dir, 'step_{:d}-RC_{:.3f}-KL_{:.3f}-optim.pt'.format(
            trained_steps,
            recons_loss_ema, 
            kl_raw_ema
          ))
      )

  print ('[epoch {:03d}] training completed\n  -- loss = (RC: {:.4f} | KL: {:.4f} | KL_raw: {:.4f})\n  -- time elapsed = {:.2f} secs.'.format(
    epoch, recons_loss_ema, kl_loss_ema, kl_raw_ema, time.time() - st
  ))
  log_data = {
    'ep': epoch,
    'steps': trained_steps,
    'recons_loss': recons_loss_ema,
    'kldiv_loss': kl_loss_ema,
    'kldiv_raw': kl_raw_ema,
    'time': time.time() - st
  }
  log_epoch(
    os.path.join(ckpt_dir, 'log.txt'), log_data, is_init=not os.path.exists(os.path.join(ckpt_dir, 'log.txt'))
  )

def validate(model, dloader, n_rounds=8, use_attr_cls=True):
  model.eval()
  loss_rec = []
  kl_loss_rec = []

  print ('[info] validating ...')
  with torch.no_grad():
    for i in range(n_rounds):
      print ('[round {}]'.format(i+1))

      for batch_idx, batch_samples in enumerate(dloader):
        model.zero_grad()

        batch_enc_inp = batch_samples['enc_input'].permute(2, 0, 1).to(device)
        batch_dec_inp = batch_samples['dec_input'].permute(1, 0).to(device)
        batch_dec_tgt = batch_samples['dec_target'].permute(1, 0).to(device)
        batch_inp_bar_pos = batch_samples['bar_pos'].to(device)
        batch_padding_mask = batch_samples['enc_padding_mask'].to(device)
        if use_attr_cls:
          batch_rfreq_cls = batch_samples['rhymfreq_cls'].permute(1, 0).to(device)
          batch_polyph_cls = batch_samples['polyph_cls'].permute(1, 0).to(device)
        else:
          batch_rfreq_cls = None
          batch_polyph_cls = None

        mu, logvar, dec_logits = model(
          batch_enc_inp, batch_dec_inp, 
          batch_inp_bar_pos, batch_rfreq_cls, batch_polyph_cls,
          padding_mask=batch_padding_mask
        )

        losses = model.compute_loss(mu, logvar, 0.0, 0.0, dec_logits, batch_dec_tgt)
        if not (batch_idx + 1) % 10:
          print ('batch #{}:'.format(batch_idx + 1), round(losses['recons_loss'].item(), 3))

        loss_rec.append(losses['recons_loss'].item())
        kl_loss_rec.append(losses['kldiv_raw'].item())
    
  return loss_rec, kl_loss_rec

if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    config['data']['data_dir'], config['data']['vocab_path'], 
    do_augment=True, 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_bars=config['data']['max_bars'],
    pieces=pickle_load(config['data']['train_split']),
    pad_to_same=True
  )
  dset_val = REMIFullSongTransformerDataset(
    config['data']['data_dir'], config['data']['vocab_path'], 
    do_augment=False, 
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['data']['dec_seqlen'], 
    model_max_bars=config['data']['max_bars'],
    pieces=pickle_load(config['data']['val_split']),
    pad_to_same=True
  )
  print ('[info]', '# training samples:', len(dset.pieces))

  dloader = DataLoader(dset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)
  dloader_val = DataLoader(dset_val, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8)

  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  if pretrained_params_path:
    model.load_state_dict( torch.load(pretrained_params_path) )

  model.train()
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print ('[info] model # params:', n_params)

  opt_params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adam(opt_params, lr=max_lr)
  if pretrained_optim_path:
    optimizer.load_state_dict( torch.load(pretrained_optim_path) )
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, lr_decay_steps, eta_min=min_lr
  )

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(params_dir):
    os.makedirs(params_dir)
  if not os.path.exists(optim_dir):
    os.makedirs(optim_dir)

  for ep in range(config['training']['max_epochs']):
    train_model(ep+1, model, dloader, dloader_val, optimizer, scheduler)