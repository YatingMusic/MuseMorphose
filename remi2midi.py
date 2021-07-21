import os, pickle, random, copy
import numpy as np

import miditoolkit

##############################
# constants
##############################
DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16


##############################
# containers for conversion
##############################
class ConversionEvent(object):
  def __init__(self, event, is_full_event=False):
    if not is_full_event:
      if 'Note' in event:
        self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
      elif 'Chord' in event:
        self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
      else:
        self.name, self.value = event.split('_')
    else:
      self.name, self.value = event['name'], event['value']
  def __repr__(self):
    return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
  def __init__(self, pitch, bar, position, duration, velocity):
    self.pitch = pitch
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
    self.duration = duration
    self.velocity = velocity
  
class TempoEvent(object):
  def __init__(self, tempo, bar, position):
    self.tempo = tempo
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

class ChordEvent(object):
  def __init__(self, chord_val, bar, position):
    self.chord_val = chord_val
    self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

##############################
# conversion functions
##############################
def read_generated_txt(generated_path):
  f = open(generated_path, 'r')
  return f.read().splitlines()

def remi2midi(events, output_midi_path=None, is_full_event=False, return_first_tempo=False, enforce_tempo=False, enforce_tempo_val=None):
  events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
  # print (events[:20])

  assert events[0].name == 'Bar'
  temp_notes = []
  temp_tempos = []
  temp_chords = []

  cur_bar = 0
  cur_position = 0

  for i in range(len(events)):
    if events[i].name == 'Bar':
      if i > 0:
        cur_bar += 1
    elif events[i].name == 'Beat':
      cur_position = int(events[i].value)
      assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
    elif events[i].name == 'Tempo':
      temp_tempos.append(TempoEvent(
        int(events[i].value), cur_bar, cur_position
      ))
    elif 'Note_Pitch' in events[i].name and \
         (i+1) < len(events) and 'Note_Velocity' in events[i+1].name and \
         (i+2) < len(events) and 'Note_Duration' in events[i+2].name:
      # check if the 3 events are of the same instrument
      temp_notes.append(
        NoteEvent(
          pitch=int(events[i].value), 
          bar=cur_bar, position=cur_position, 
          duration=int(events[i+2].value), velocity=int(events[i+1].value)
        )
      )
    elif 'Chord' in events[i].name:
      temp_chords.append(
        ChordEvent(events[i].value, cur_bar, cur_position)
      )
    elif events[i].name in ['EOS', 'PAD']:
      continue

  # print (len(temp_tempos), len(temp_notes))
  midi_obj = miditoolkit.midi.parser.MidiFile()
  midi_obj.instruments = [
    miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
  ]

  for n in temp_notes:
    midi_obj.instruments[0].notes.append(
      miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.duration))
    )

  if enforce_tempo is False:
    for t in temp_tempos:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )
  else:
    if enforce_tempo_val is None:
      enforce_tempo_val = temp_tempos[1]
    for t in enforce_tempo_val:
      midi_obj.tempo_changes.append(
        miditoolkit.TempoChange(t.tempo, int(t.start_tick))
      )

  
  for c in temp_chords:
    midi_obj.markers.append(
      miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
    )
  for b in range(cur_bar):
    midi_obj.markers.append(
      miditoolkit.Marker('Bar-{}'.format(b+1), int(DEFAULT_BAR_RESOL * b))
    )

  if output_midi_path is not None:
    midi_obj.dump(output_midi_path)

  if not return_first_tempo:
    return midi_obj
  else:
    return midi_obj, temp_tempos