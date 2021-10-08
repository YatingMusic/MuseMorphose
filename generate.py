import sys, os, random, time
from copy import deepcopy
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose

from utils import pickle_load, numpy_to_tensor, tensor_to_numpy
from remi2midi import remi2midi

import torch
import yaml
import numpy as np
from scipy.stats import entropy

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
data_dir = config['data']['data_dir']
vocab_path = config['data']['vocab_path']
data_split = 'pickles/test_pieces.pkl'

ckpt_path = sys.argv[2]
out_dir = sys.argv[3]
n_pieces = int(sys.argv[4])
n_samples_per_piece = int(sys.argv[5])

###########################################
# little helpers
###########################################
def word2event(word_seq, idx2event):
  return [ idx2event[w] for w in word_seq ]

def get_beat_idx(event):
  return int(event.split('_')[-1])

###########################################
# sampling utilities
###########################################
def temperatured_softmax(logits, temperature):
  try:
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    assert np.count_nonzero(np.isnan(probs)) == 0
  except:
    print ('overflow detected, use 128-bit')
    logits = logits.astype(np.float128)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    probs = probs.astype(float)
  return probs

def nucleus(probs, p):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][1]
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:3] # just assign a value
    candi_probs = np.array([probs[i] for i in candi_index], dtype=np.float64)
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

########################################
# generation
########################################
def get_latent_embedding_fast(model, piece_data, use_sampling=False, sampling_var=0.):
  # reshape
  batch_inp = piece_data['enc_input'].permute(1, 0).long().to(device)
  batch_padding_mask = piece_data['enc_padding_mask'].bool().to(device)

  # get latent conditioning vectors
  with torch.no_grad():
    piece_latents = model.get_sampled_latent(
      batch_inp, padding_mask=batch_padding_mask, 
      use_sampling=use_sampling, sampling_var=sampling_var
    )

  return piece_latents

def generate_on_latent_ctrl_vanilla_truncate(
        model, latents, rfreq_cls, polyph_cls, event2idx, idx2event, 
        max_events=12800, primer=None,
        max_input_len=1280, truncate_len=512, 
        nucleus_p=0.9, temperature=1.2
      ):
  latent_placeholder = torch.zeros(max_events, 1, latents.size(-1)).to(device)
  rfreq_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  polyph_placeholder = torch.zeros(max_events, 1, dtype=int).to(device)
  print ('[info] rhythm cls: {} | polyph_cls: {}'.format(rfreq_cls, polyph_cls))

  if primer is None:
    generated = [event2idx['Bar_None']]
  else:
    generated = [event2idx[e] for e in primer]
    latent_placeholder[:len(generated), 0, :] = latents[0].squeeze(0)
    rfreq_placeholder[:len(generated), 0] = rfreq_cls[0]
    polyph_placeholder[:len(generated), 0] = polyph_cls[0]
    
  target_bars, generated_bars = latents.size(0), 0

  steps = 0
  time_st = time.time()
  cur_pos = 0
  failed_cnt = 0

  cur_input_len = len(generated)
  generated_final = deepcopy(generated)
  entropies = []

  while generated_bars < target_bars:
    if len(generated) == 1:
      dec_input = numpy_to_tensor([generated], device=device).long()
    else:
      dec_input = numpy_to_tensor([generated], device=device).permute(1, 0).long()

    latent_placeholder[len(generated)-1, 0, :] = latents[ generated_bars ]
    rfreq_placeholder[len(generated)-1, 0] = rfreq_cls[ generated_bars ]
    polyph_placeholder[len(generated)-1, 0] = polyph_cls[ generated_bars ]

    dec_seg_emb = latent_placeholder[:len(generated), :]
    dec_rfreq_cls = rfreq_placeholder[:len(generated), :]
    dec_polyph_cls = polyph_placeholder[:len(generated), :]

    # sampling
    with torch.no_grad():
      logits = model.generate(dec_input, dec_seg_emb, dec_rfreq_cls, dec_polyph_cls)
    logits = tensor_to_numpy(logits[0])
    probs = temperatured_softmax(logits, temperature)
    word = nucleus(probs, nucleus_p)
    word_event = idx2event[word]

    if 'Beat' in word_event:
      event_pos = get_beat_idx(word_event)
      if not event_pos >= cur_pos:
        failed_cnt += 1
        print ('[info] position not increasing, failed cnt:', failed_cnt)
        if failed_cnt >= 128:
          print ('[FATAL] model stuck, exiting ...')
          return generated
        continue
      else:
        cur_pos = event_pos
        failed_cnt = 0

    if 'Bar' in word_event:
      generated_bars += 1
      cur_pos = 0
      print ('[info] generated {} bars, #events = {}'.format(generated_bars, len(generated_final)))
    if word_event == 'PAD_None':
      continue

    if len(generated) > max_events or (word_event == 'EOS_None' and generated_bars == target_bars - 1):
      generated_bars += 1
      generated.append(event2idx['Bar_None'])
      print ('[info] gotten eos')
      break

    generated.append(word)
    generated_final.append(word)
    entropies.append(entropy(probs))

    cur_input_len += 1
    steps += 1

    assert cur_input_len == len(generated)
    if cur_input_len == max_input_len:
      generated = generated[-truncate_len:]
      latent_placeholder[:len(generated)-1, 0, :] = latent_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0, :]
      rfreq_placeholder[:len(generated)-1, 0] = rfreq_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]
      polyph_placeholder[:len(generated)-1, 0] = polyph_placeholder[cur_input_len-truncate_len:cur_input_len-1, 0]

      print ('[info] reset context length: cur_len: {}, accumulated_len: {}, truncate_range: {} ~ {}'.format(
        cur_input_len, len(generated_final), cur_input_len-truncate_len, cur_input_len-1
      ))
      cur_input_len = len(generated)

  assert generated_bars == target_bars
  print ('-- generated events:', len(generated_final))
  print ('-- time elapsed: {:.2f} secs'.format(time.time() - time_st))
  return generated_final[:-1], time.time() - time_st, np.array(entropies)


########################################
# change attribute classes
########################################
def random_shift_attr_cls(n_samples, upper=4, lower=-3):
  return np.random.randint(lower, upper, (n_samples,))


if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    data_dir, vocab_path, 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=pickle_load(data_split),
    pad_to_same=False
  )
  pieces = random.sample(range(len(dset)), n_pieces)
  print ('[sampled pieces]', pieces)
  
  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  model.eval()
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)

  times = []
  for p in pieces:
    # fetch test sample
    p_data = dset[p]
    p_id = p_data['piece_id']
    p_bar_id = p_data['st_bar_id']
    p_data['enc_input'] = p_data['enc_input'][ : p_data['enc_n_bars'] ]
    p_data['enc_padding_mask'] = p_data['enc_padding_mask'][ : p_data['enc_n_bars'] ]

    orig_p_cls_str = ''.join(str(c) for c in p_data['polyph_cls_bar'])
    orig_r_cls_str = ''.join(str(c) for c in p_data['rhymfreq_cls_bar'])

    orig_song = p_data['dec_input'].tolist()[:p_data['length']]
    orig_song = word2event(orig_song, dset.idx2event)
    orig_out_file = os.path.join(out_dir, 'id{}_bar{}_orig'.format(
        p, p_bar_id
    ))
    print ('[info] writing to ...', orig_out_file)
    # output reference song's MIDI
    _, orig_tempo = remi2midi(orig_song, orig_out_file + '.mid', return_first_tempo=True, enforce_tempo=False)

    # save metadata of reference song (events & attr classes)
    print (*orig_song, sep='\n', file=open(orig_out_file + '.txt', 'a'))
    np.save(orig_out_file + '-POLYCLS.npy', p_data['polyph_cls_bar'])
    np.save(orig_out_file + '-RHYMCLS.npy', p_data['rhymfreq_cls_bar'])


    for k in p_data.keys():
      if not torch.is_tensor(p_data[k]):
        p_data[k] = numpy_to_tensor(p_data[k], device=device)
      else:
        p_data[k] = p_data[k].to(device)

    p_latents = get_latent_embedding_fast(
                  model, p_data, 
                  use_sampling=config['generate']['use_latent_sampling'],
                  sampling_var=config['generate']['latent_sampling_var']
                )
    p_cls_diff = random_shift_attr_cls(n_samples_per_piece)
    r_cls_diff = random_shift_attr_cls(n_samples_per_piece)

    piece_entropies = []
    for samp in range(n_samples_per_piece):
      p_polyph_cls = (p_data['polyph_cls_bar'] + p_cls_diff[samp]).clamp(0, 7).long()
      p_rfreq_cls = (p_data['rhymfreq_cls_bar'] + r_cls_diff[samp]).clamp(0, 7).long()

      print ('[info] piece: {}, bar: {}'.format(p_id, p_bar_id))
      out_file = os.path.join(out_dir, 'id{}_bar{}_sample{:02d}_poly{}_rhym{}'.format(
        p, p_bar_id, samp + 1,
        '+{}'.format(p_cls_diff[samp]) if p_cls_diff[samp] >= 0 else p_cls_diff[samp], 
        '+{}'.format(r_cls_diff[samp]) if r_cls_diff[samp] >= 0 else r_cls_diff[samp]
      ))      
      print ('[info] writing to ...', out_file)
      if os.path.exists(out_file + '.txt'):
        print ('[info] file exists, skipping ...')
        continue

      # print (p_polyph_cls, p_rfreq_cls)

      # generate
      song, t_sec, entropies = generate_on_latent_ctrl_vanilla_truncate(
                                  model, p_latents, p_rfreq_cls, p_polyph_cls, dset.event2idx, dset.idx2event,
                                  max_input_len=config['generate']['max_input_dec_seqlen'], 
                                  truncate_len=min(512, config['generate']['max_input_dec_seqlen'] - 32), 
                                  nucleus_p=config['generate']['nucleus_p'], 
                                  temperature=config['generate']['temperature'],
                                  
                                )
      times.append(t_sec)

      song = word2event(song, dset.idx2event)
      print (*song, sep='\n', file=open(out_file + '.txt', 'a'))
      remi2midi(song, out_file + '.mid', enforce_tempo=True, enforce_tempo_val=orig_tempo)

      # save metadata of the generation
      np.save(out_file + '-POLYCLS.npy', tensor_to_numpy(p_polyph_cls))
      np.save(out_file + '-RHYMCLS.npy', tensor_to_numpy(p_rfreq_cls))
      print ('[info] piece entropy: {:.4f} (+/- {:.4f})'.format(
        entropies.mean(), entropies.std()
      ))
      piece_entropies.append(entropies.mean())

  print ('[time stats] {} songs, generation time: {:.2f} secs (+/- {:.2f})'.format(
    n_pieces * n_samples_per_piece, np.mean(times), np.std(times)
  ))
  print ('[entropy] {:.4f} (+/- {:.4f})'.format(
    np.mean(piece_entropies), np.std(piece_entropies)
  ))