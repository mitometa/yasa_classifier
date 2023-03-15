import yaml
import os
import sys
from py_console import console
import numpy as np
import pandas as pd
import glob
import datetime

def read_yaml(filename):
  with open(filename, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
  return cfg

def create_channel_map(eeg_array, config, end = "extend"):
  channel_map = {}
  n_channels = eeg_array.shape[0]
  n_named_chan = len(config['selected_channels'])
  if  n_named_chan < n_channels:
    print(f"Data has {n_channels} channels but only {n_named_chan} named channels")
    # TODO: add "prepend option"
    if end == "extend":
      print("Adding channel(s) for camera(s) at the end")
      cam_names = [f"cam{i}" for i in range(1, n_channels - n_named_chan + 1)]
      new_channel_names = list(config['channel_names'])
      new_channel_names.extend(cam_names)
      for key, value in enumerate(new_channel_names):
        channel_map[key] = value 
    return(channel_map)
  else:
    return(config['channel_names'])

def validate_file_exists(folder, pattern):
  files_match = glob.glob(os.path.join(folder, pattern))
  if not files_match:
    console.error(f"{pattern} not found in {folder}", severe=True)
    sys.exit()
  else:
    if len(files_match) > 1:
      console.warning(f"{pattern} in more than a signle file")
      print(files_match)
      console.error(f"Stopping for pattern fixing")
      sys.exit()
    # unlist using first element, assume only one match...
    filepath = files_match[0]
    return filepath

def normalize_ttl(ttl_matrix, method="max"):
  if method == "max":
    max_per_channel = ttl_matrix.max(axis=1, keepdims=True)
    # remove the zeros so we can devide
    max_per_channel = np.where(max_per_channel == 0, 1, max_per_channel)
    out = ttl_matrix / max_per_channel
    return(out)

def find_pulse_onset(ttl_file, ttl_idx, timestamps_file, buffer, round=False):
  """
  This function reads the ttl pulse file
  Subsets the ttl_file array on ttl_idx
  buffer is sf / 4
  Finds pulse onset by calling np diff and looking for the moments where np.diff is positive
  There's two ways to call this. You can either return the rounded down timestamp (round=True) 
  # or interpolate from the closest timestamp assuming constant sampling rate.
  Rerturns the timestamps according to sampling frequency (sf)
  """
  sf = 4 * buffer
  ttl_events = np.load(ttl_file)
  # todo find in config
  photo_events = ttl_events[ttl_idx, :].flatten()
  pulse_onset = np.where(np.diff(photo_events, prepend=0) > 0)[0]
  # get division and remainder
  div_array = np.array([divmod(i, buffer) for i in pulse_onset])
  # TODO div_array[:, 0] has the rounded version
  # timestamps.iloc[div_array[:, 0], :] + sampling_period * div_array[:, 1] is the way to calculate the proper timestamp
  # this assumes constant sampling rate between known timestamps
  timestamps = pd.read_csv(timestamps_file)
  # TODO: not sure this works for all sf 
  sampling_period_ms = 1/sf * 1000
  out = timestamps.iloc[div_array[:,0], :].copy()
  if round:
    return out
  else:
    out.iloc[:, 0] = out.iloc[:, 0] + sampling_period_ms * div_array[:, 1]
    dt = (sampling_period_ms * div_array[:, 1]).astype('timedelta64[ms]')
    out.iloc[:, 1] = pd.to_datetime(out.iloc[:,1]) + dt
    return out
