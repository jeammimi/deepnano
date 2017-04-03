import h5py
import sys
import numpy as np
import os
import re
import dateutil.parser
import datetime
import argparse

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

def scale(X):
  m25 = np.percentile(X[:,0], 25)
  m75 = np.percentile(X[:,0], 75)
  s50 = np.median(X[:,2])
  me25 = 0.07499809
  me75 = 0.26622871
  se50 = 0.6103758
  ret = np.array(X)
  scale = (me75 - me25) / (m75 - m25)
  m25 *= scale
  shift = me25 - m25
  ret[:,0] = X[:,0] * scale + shift
  ret[:,1] = ret[:,0]**2
  
  sscale = se50 / s50

  ret[:,2] = X[:,2] * sscale
  return ret

def preproc_event(mean, std, length):
  mean = mean / 100.0 - 0.66
  std = std - 1
  return [mean, mean*mean, std, length]

def get_base_loc(h5):
  base_loc = "Analyses/Basecall_2D_000"
  try:
    events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  except:
    base_loc = "Analyses/Basecall_1D_000"
  return base_loc

def extract_scaling(h5, read_type, base_loc):
  scale = h5[base_loc+"/Summary/basecall_1d_"+read_type].attrs["scale"]
  scale_sd = h5[base_loc+"/Summary/basecall_1d_"+read_type].attrs["scale_sd"]
  shift = h5[base_loc+"/Summary/basecall_1d_"+read_type].attrs["shift"]
  drift = h5[base_loc+"/Summary/basecall_1d_"+read_type].attrs["drift"]
  return scale, scale_sd, shift, drift
