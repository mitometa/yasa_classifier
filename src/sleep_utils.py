from scipy.io import loadmat
import numpy as np
import mne
from mne.io import RawArray
from src.staging import SleepStaging
import rolling
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

def get_files(eeg_file, emg_file):
  # load files
  eeg = loadmat(eeg_file)
  emg = loadmat(emg_file)
  # reshape everything to singleton
  eeg_array = np.squeeze(eeg["EEG"])
  emg_array = np.squeeze(emg["EMG"])
  return eeg_array, emg_array


def predict_mouse(eeg, emg, sf, epoch_sec, path_to_model=None):
  info =  mne.create_info(["eeg","emg"], 
                          sf, 
                          ch_types='misc', 
                          verbose=False)
  raw_array = RawArray(np.vstack((eeg, 
                                  emg)),
                       info, verbose=False)
  sls = SleepStaging(raw_array,
                     eeg_name="eeg", 
                     emg_name="emg")
  # this will use the new fit function
  sls.fit(epoch_sec=epoch_sec)
  # the auto will use these features
  # "/home/matias/anaconda3/lib/python3.7/site-packages/yasa/classifiers/clf_eeg+emg_lgb_0.5.0.joblib"
  predicted_labels = sls.predict(path_to_model=path_to_model)
#np.save("predicted_labels.npy", predicted_labels)
  return predicted_labels



# This function maps labels between the two standards (Accusleep and yasa)
def accusleep_to_yasa(df):
  # yasa values for hypnogram
  #* -2  = Unscored
  #* -1  = Artefact / Movement
  #* 0   = Wake
  #* 1   = N1 sleep
  #* 2   = N2 sleep
  #* 3   = N3 sleep
  #* 4   = REM sleep
  # arrange into proper shape
  accusleep_dict = {
    1:4,#"R",
    2:0,#"W",
    3:2,#"N"
  }
  # re-map the label values
  df["label"] = df["label"].map(accusleep_dict)
  label_array = df["label"].values
  return label_array

import yasa
def plot_spectrogram(eeg, hypno, sf, epoch_sec):
  # upsample to data
  label_df = yasa.hypno_upsample_to_data(hypno,
  sf_hypno=1/epoch_sec, 
  data=eeg, sf_data=sf)
  fig = yasa.plot_spectrogram(eeg,
                      hypno=label_df, 
                      win_sec = 10,
                      sf=sf,
                      # default is 'RdBu_r'
                      # cmap='Spectral_r',
                      # manage the scale contrast,     larger values better contrast
                      trimperc = 1)
  return fig


def evaluate(predicted_labels, true_labels, as_string=True, plot_cm=False):
  if not as_string:
    # convert labels to string
    predicted_labels_string = yasa.hypno_int_to_str(predicted_labels)
    true_labels_string = yasa.hypno_int_to_str(true_labels)
    # Get labels in the order that they will show in cm
    cm_labels = yasa.hypno_int_to_str(np.unique(true_labels))
  else:
    predicted_labels_string = predicted_labels
    true_labels_string = true_labels
    cm_labels = np.unique(true_labels)
  
  from sklearn.metrics import accuracy_score
  accuracy = accuracy_score(
    y_true = true_labels_string,
    y_pred = predicted_labels_string
    )
  from sklearn.metrics import cohen_kappa_score
  from sklearn.metrics import confusion_matrix
  cohen_kappa = cohen_kappa_score(
    predicted_labels_string, true_labels_string)
  print(f"Cohen's Kappa: {cohen_kappa}")
  cm = confusion_matrix(predicted_labels_string, true_labels_string)
  if plot_cm:
    ax=plt.subplot()
    sns.heatmap(cm/np.sum(cm), annot=True, 
                fmt='.2%', cmap='Blues')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix') 
    ax.xaxis.set_ticklabels(cm_labels) 
    ax.yaxis.set_ticklabels(cm_labels)
    plt.show()
    plt.close()
  return accuracy, cm, cohen_kappa


### Smoothing functions ######

def smooth_sequence(seq, kernel_size=None, sf = None, method = "mode"):
  if sf is None:
    sf = 1
    print("Warning: no sf provided, using `kernel_size` in sample space")
  else:
    # TODO: we should probably make sure this is int better than this
    kernel_size = int(kernel_size * sf)
  match method:
    case 'mode':
         out = smooth_sequence_mode(seq, kernel_size)
    case 'min_length':
         out = smooth_sequence_gaps(seq, kernel_size)
    case _:
        print(f"method {method} has to be `mode` or `min_length`")
        # we return empty but should raise an error actually
        return
  return out 

def smooth_sequence_mode(seq, kernel_size):
    half_kernel = kernel_size // 2
    padded_seq = np.pad(seq, (half_kernel, half_kernel), mode='edge')
    smoothed_seq = np.array(list(rolling.Apply(padded_seq, kernel_size, operation=lambda x: scipy.stats.mode(x).mode[0], window_type='variable')))
    return smoothed_seq[half_kernel:len(seq) + half_kernel]


def smooth_sequence_gaps(seq, kernel_size):
  seq_copy = seq.copy()
  # returns start_idx stop_idx count
  rle_arr = rle(seq_copy)
  short_runs = np.where(rle_arr[:, 2] < kernel_size)[0]
  short_starts = rle_arr[short_runs, 0]
  short_ends = rle_arr[short_runs, 1]
  # TODO: add quality control here, it might be that kernel_size is too big and 
  # len(short_runs) is ~ len(seq) -> everything will be turned into np.nan!!!
  # replace to np.nan
  for i in short_runs:
    start_idx, end_idx = rle_arr[i, 0], rle_arr[i, 1]
    if start_idx == 0:
      # this is left edge case
      # fill with the value to the right of end_idx
      next_idx = end_idx + 1
      # check that the next one is also not empty
      while next_idx in short_starts:
        next_idx = next_idx + 1
      seq_copy[start_idx:end_idx] = seq_copy[next_idx]
    elif start_idx == len(seq) - 1:
      # this is right edge case
      # fill with the value to the left of start_idx
      previous_end_idx = start_idx - 1
      while previous_end_idx in short_ends:
        previous_end_idx = previous_end_idx - 1 
      seq_copy[start_idx:] = seq_copy[previous_end_idx]
    else:
      # this is the gap
      # fill with previous value
      previous_end_idx = start_idx - 1
      while previous_end_idx in short_ends:
        previous_end_idx = previous_end_idx - 1
      seq_copy[start_idx:end_idx] = seq_copy[previous_end_idx]
  return seq_copy

# implementation inspired from https://gist.github.com/akTwelve/dc0bbbf26fb14493898fc74cd2aa7f74
def rle(arr):
  start_idx = np.nonzero(np.append(arr, 0) != np.append(0, arr))[0]
  # first value will be zero
  stop_idx = start_idx[1:]
  # Remove the last value to avoid getting it twice
  start_idx = start_idx[:-1]
  rle_arr = stop_idx - start_idx
  return np.array([start_idx, stop_idx, rle_arr]).T

# This might not be neccesary given we have the rle info
def fill_edges(seq):
  # fill edge nans with nearest non-nan values
  nan_idxs = np.where(pd.isnull(seq))[0]
  # TODO: ensure len(nan_idx) < len(seq) (basically, there are some non-nans in the sequence)
  for idx in nan_idxs:
      if idx == 0:  # fill edge nans with nearest non-nan values
          # fill left edge nan with first non-nan to the right
          fill_idx = np.where(~pd.isnull(seq[idx:]))[0]
          seq[idx] = seq[idx + fill_idx[0]]
      elif idx == len(seq) - 1:
          # fill right edge nan with first non-nan to the left
          fill_idx = np.where(~pd.isnull(seq[:idx]))[0]
          seq[idx] = seq[fill_idx[-1]]
  return seq


#seq = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'B', 'C', 'C', 'A', 'A'], dtype=object)
#smooth_sequence(seq, 4, 'mode')
