import numpy as np
import sys
import datetime

defs = {
    'r9.4': {
        'ed_params': {
            'window_lengths':[3, 6], 'thresholds':[1.4, 1.1],
            'peak_height':0.2
        }
    },
    'r9': {
        'ed_params': {
            'window_lengths':[5, 10], 'thresholds':[2.0, 1.1],
            'peak_height':1.2
        }
    }
}

def get_raw(h5):
  rk = h5["Raw/Reads"].keys()[0]

  raw = h5["Raw/Reads"][rk]["Signal"]
  meta = h5["UniqueGlobalKey/channel_id"].attrs
  offset = meta["offset"]
  raw_unit = meta['range'] / meta['digitisation']
  raw = (raw + offset) * raw_unit
  sl = meta["sampling_rate"]

  return raw, sl

def find_stall(events, threshold):
  count_above = 0
  start_ev_ind = 0
  for ev_ind, event in enumerate(events[:100]):
      if event['mean'] <= threshold:
          count_above = 0
      else:
          count_above += 1

      if count_above == 2:
          start_ev_ind = ev_ind - 1
          break
      
  new_start = 0
  count = 0
  for idx in range(start_ev_ind, len(events)):
      if events['mean'][idx] > threshold:
          count = 0
      else:
          count += 1

      if count == 3:
          new_start = idx - 2
          break

  return new_start 

def get_tstat(s, s2, wl):
   eta = 1e-100
   windows1 = np.concatenate([[s[wl-1]], s[wl:] - s[:-wl]])
   windows2 = np.concatenate([[s2[wl-1]], s2[wl:] - s2[:-wl]])
   means = windows1 / wl
   variances = windows2 / wl - means * means
   variances = np.maximum(variances, eta)
   delta = means[wl:] - means[:-wl]
   deltav = (variances[wl:] + variances[:-wl]) / wl
   return np.concatenate([np.zeros(wl), np.abs(delta / np.sqrt(deltav)), np.zeros(wl-1)])
   

def extract_events(h5, chem):
  print "ed"
  raw, sl = get_raw(h5)

  events = event_detect(raw, sl, **defs[chem]["ed_params"])
  med, mad = med_mad(events['mean'][-100:])
  max_thresh = med + 1.48 * 2 + mad

  first_event = find_stall(events, max_thresh)

  return events[first_event:]

def med_mad(data):
  dmed = np.median(data)
  dmad = np.median(abs(data - dmed))
  return dmed, dmad

def compute_prefix_sums(data):
  data_sq = data * data
  return np.cumsum(data), np.cumsum(data_sq)

def peak_detect(short_data, long_data, short_window, long_window, short_threshold, long_threshold,
peak_height):
  long_mask = 0
  NO_PEAK_POS = -1
  NO_PEAK_VAL = 1e100
  peaks = []
  short_peak_pos = NO_PEAK_POS
  short_peak_val = NO_PEAK_VAL
  short_found_peak = False
  long_peak_pos = NO_PEAK_POS
  long_peak_val = NO_PEAK_VAL
  long_found_peak = False

  for i in range(len(short_data)):
    val = short_data[i]
    if short_peak_pos == NO_PEAK_POS:
      if val < short_peak_val:
        short_peak_val = val
      elif val - short_peak_val > peak_height:
        short_peak_val = val
        short_peak_pos = i
    else:
      if val > short_peak_val:
        short_peak_pos = i
        short_peak_val = val
      if short_peak_val > short_threshold:
        long_mask = short_peak_pos + short_window
        long_peak_pos = NO_PEAK_POS
        long_peak_val = NO_PEAK_VAL
        long_found_peak = False
      if short_peak_val - val > peak_height and short_peak_val > short_threshold:
        short_found_peak = True
      if short_found_peak and (i - short_peak_pos) > short_window / 2:
        peaks.append(short_peak_pos)
        short_peak_pos = NO_PEAK_POS
        short_peak_val = val
        short_found_peak = False

    if i <= long_mask:
      continue
    val = long_data[i]
    if long_peak_pos == NO_PEAK_POS:
      if val < long_peak_val:
        long_peak_val = val
      elif val - long_peak_val > peak_height:
        long_peak_val = val
        long_peak_pos = i
    else:
      if val > long_peak_val:
        long_peak_pos = i
        long_peak_val = val
      if long_peak_val - val > peak_height and long_peak_val > long_threshold:
        long_found_peak = True
      if long_found_peak and (i - long_peak_pos) > long_window / 2:
        peaks.append(long_peak_pos)
        long_peak_pos = NO_PEAK_POS
        long_peak_val = val
        long_found_peak = False

  return peaks

def generate_events(ss1, ss2, peaks, sample_rate):
  peaks.append(len(ss1))
  events = np.empty(len(peaks), dtype=[('start', float), ('length', float),
                                       ('mean', float), ('stdv', float)])
  s = 0
  s1 = 0
  s2 = 0
  for i, e in enumerate(peaks):
    events[i]["start"] = s
    l = e - s
    events[i]["length"] = l
    m = (ss1[e-1] - s1) / l
    events[i]["mean"] = m
    v = max(0.0, (ss2[e-1] - s2) / l - m*m)
    events[i]["stdv"] = np.sqrt(v)
    s = e
    s1 = ss1[e-1]
    s2 = ss2[e-2]

  events["start"] /= sample_rate
  events["length"] /= sample_rate

  return events


def event_detect(raw_data, sample_rate, window_lengths=[16, 40], thresholds=[8.0, 4.0], peak_height = 1.0):
    """Basic, standard even detection using two t-tests

    :param raw_data: ADC values
    :param sample_rate: Sampling rate of data in Hz
    :param window_lengths: Length 2 list of window lengths across
        raw data from which `t_stats` are derived
    :param thresholds: Length 2 list of thresholds on t-statistics
    :peak_height: Absolute height a peak in signal must rise below
        previous and following minima to be considered relevant
    """
    sums, sumsqs = compute_prefix_sums(raw_data)

    tstats = []
    for i, w_len in enumerate(window_lengths):
        tstat = get_tstat(sums, sumsqs, w_len)
        tstats.append(tstat)

    peaks = peak_detect(tstats[0], tstats[1], window_lengths[0], window_lengths[1], thresholds[0],
                        thresholds[1], peak_height)
    events = generate_events(sums, sumsqs, peaks, sample_rate)

    return events


