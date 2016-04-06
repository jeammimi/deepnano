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
