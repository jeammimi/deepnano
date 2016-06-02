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
