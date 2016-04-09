from rnn_fin import RnnPredictor
import h5py
import sys
import numpy as np
import theano as th
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

def predict_and_write(events, ntwk, fo, read_name):
  o1, o2 = ntwk.predict(events) 
  if fo:
    o1m = (np.argmax(o1, 1))
    o2m = (np.argmax(o2, 1))
    print >>fo, ">%s" % read_name
    for a, b in zip(o1m, o2m):
      if a < 4:
        fo.write(chars[a])
      if b < 4:
        fo.write(chars[b])
    fo.write('\n')
  return o1, o2

def extract_timing(h5, ret):
  try:
    log = h5["Analyses/Basecall_2D_000/Log"][()]
    temp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling template.*", log).groups()[0])
    comp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling complement.*", log).groups()[0])
    comp_end_time = dateutil.parser.parse(re.search(r"(.*) Aligning hairpin.*", log).groups()[0])

    start_2d_time = dateutil.parser.parse(re.search(r"(.*) Performing full 2D.*", log).groups()[0])
    end_2d_time = dateutil.parser.parse(re.search(r"(.*) Workflow completed.*", log).groups()[0])

    ret["temp_time"] = comp_time - temp_time
    ret["comp_time"] = comp_end_time - comp_time
    ret["2d_time"] = end_2d_time - start_2d_time
  except:
    pass

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

def extract_1d_event_data(h5, read_type, base_loc, scale, scale_sd, shift, drift):
  events = h5[base_loc+"/BaseCalled_%s/Events" % read_type]
  index = 0.0
  data = []
  for e in events:
    mean = (e["mean"] - shift - index * drift) / scale
    stdv = e["stdv"] / scale_sd
    length = e["length"]
    data.append(preproc_event(mean, stdv, length))
    index += e["length"]
  return np.array(data, dtype=np.float32)

