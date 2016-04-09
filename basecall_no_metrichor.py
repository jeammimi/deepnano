import argparse
from rnn_fin import RnnPredictor
import h5py
import sys
import numpy as np
import theano as th
import os
import re
import dateutil.parser
import datetime
from helpers import *
import subprocess

def get_scaling_template(events, has_std):
  down = 48.4631279889
  up = 65.7312554591
  our_down = np.percentile(events["mean"], 10)
  our_up = np.percentile(events["mean"], 90)
  scale = (our_up - our_down) / (up - down)
  shift = (our_up / scale - up) * scale

  sd = 0.807981325017
  if has_std:
    return scale, np.percentile(events["stdv"], 50) / sd, shift
  else:
    return scale, np.sqrt(np.percentile(events["variance"], 50)) / sd, shift
    

def get_scaling_complement(events, has_std):
  down = 49.2638926877
  up = 69.0192568072
  our_down = np.percentile(events["mean"], 10)
  our_up = np.percentile(events["mean"], 90)
  scale = (our_up - our_down) / (up - down)
  shift = (our_up / scale - up) * scale

  sd = 1.04324844612
  if has_std:
    return scale, np.percentile(events["stdv"], 50) / sd, shift
  else:
    return scale, np.sqrt(np.percentile(events["variance"], 50)) / sd, shift

def template_complement_loc(events):
  abasic_level = np.percentile(events["mean"], 99) + 5
  abasic_locs = (events["mean"] > abasic_level).nonzero()[0]
  last = -47
  run_len = 1
  runs = []
  for x in abasic_locs:
    if x - last == 1:
      run_len += 1
    else:
      if run_len >= 5:
        if len(runs) and last - runs[-1][0] < 50:
          run_len = last - runs[-1][0]
          run_len += runs[-1][1]
          runs[-1] = (last, run_len)
        else:
          runs.append((last, run_len))
      run_len = 1
    last = x
  to_sort = []
  mid = len(events) / 2
  low_third = len(events) / 3
  high_third = len(events) / 3 * 2
  for r in runs:
    if r[0] < low_third:
      continue
    if r[0] > high_third:
      continue
    to_sort.append((abs(r[0] - mid), r[0] - r[1], r[0]))
  to_sort.sort()
  if len(to_sort) == 0:
    return None
  trim_size = 10
  return {"temp": (trim_size, to_sort[0][1] - trim_size),
          "comp": (to_sort[0][2] + trim_size, len(events) - trim_size)}

def load_read_data(read_file):
  h5 = h5py.File(read_file, "r")
  ret = {}

  read_key = h5["Analyses/EventDetection_000/Reads"].keys()[0]
  base_events = h5["Analyses/EventDetection_000/Reads"][read_key]["Events"]
  temp_comp_loc = template_complement_loc(base_events)
  if not temp_comp_loc:
    return None
  sampling_rate = h5["UniqueGlobalKey/channel_id"].attrs["sampling_rate"]

  events = base_events[temp_comp_loc["temp"][0]:temp_comp_loc["temp"][1]]
  has_std = True
  try:
    std = events[0]["stdv"]
  except:
    has_std = False
  tscale2, tscale_sd2, tshift2 = get_scaling_template(events, has_std)

  index = 0.0
  ret["temp_events2"] = []
  for e in events:
    mean = (e["mean"] - tshift2) / tscale2
    if has_std:
      stdv = e["stdv"] / tscale_sd2
    else:
      stdv = np.sqrt(e["variance"]) / tscale_sd2
    length = e["length"] / sampling_rate
    ret["temp_events2"].append(preproc_event(mean, stdv, length))
  
  events = base_events[temp_comp_loc["comp"][0]:temp_comp_loc["comp"][1]]
  cscale2, cscale_sd2, cshift2 = get_scaling_complement(events, has_std)

  index = 0.0
  ret["comp_events2"] = []
  for e in events:
    mean = (e["mean"] - cshift2) / cscale2
    if has_std:
      stdv = e["stdv"] / cscale_sd2
    else:
      stdv = np.sqrt(e["variance"]) / cscale_sd2
    length = e["length"] / sampling_rate
    ret["comp_events2"].append(preproc_event(mean, stdv, length))

  ret["temp_events2"] = np.array(ret["temp_events2"], dtype=np.float32)
  ret["comp_events2"] = np.array(ret["comp_events2"], dtype=np.float32)

  return ret

parser = argparse.ArgumentParser()
parser.add_argument('--template_net', type=str, default="nets_data/map6temp.npz")
parser.add_argument('--complement_net', type=str, default="nets_data/map6comp.npz")
parser.add_argument('--big_net', type=str, default="nets_data/map6-2d-no-metr10.npz")
parser.add_argument('reads', type=str, nargs='+')
parser.add_argument('--type', type=str, default="all", help="One of: template, complement, 2d, all, use comma to separate multiple options, eg.: template,complement")
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--directory', type=str, default='', help="Directory where read files are stored")


args = parser.parse_args()
types = args.type.split(',')
do_template = False
do_complement = False
do_2d = False

if "all" in types or "template" in types:
  do_template = True
if "all" in types or "complement" in types:
  do_complement = True
if "all" in types or "2d" in types:
  do_2d = True

assert do_template or do_complement or do_2d, "Nothing to do"
assert len(args.reads) != 0 or len(args.directory) != 0, "Nothing to basecall"

if do_template or do_2d:
  print "loading template net"
  temp_net = RnnPredictor(args.template_net)
  print "done"
if do_complement or do_2d:
  print "loading complement net"
  comp_net = RnnPredictor(args.complement_net)
  print "done"
if do_2d:
  print "loading 2D net"
  big_net = RnnPredictor(args.big_net)
  print "done"

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

fo = open(args.output, "w")

files = args.reads
if len(args.directory):
  files += [os.path.join(args.directory, x) for x in os.listdir(args.directory)]  

for i, read in enumerate(files):
  basename = os.path.basename(read)
  try:
    data = load_read_data(read)
  except Exception as e:
    print e
    print "error at file", read
    continue

  if do_template or do_2d:
    o1, o2 = predict_and_write(
        data["temp_events2"], temp_net, 
        fo if do_template else None,
        "%s_template_rnn" % basename)

  if do_complement or do_2d:
    o1c, o2c = predict_and_write(
        data["comp_events2"], comp_net, 
        fo if do_complement else None,
        "%s_complement_rnn" % basename)

  if do_2d:
    p = subprocess.Popen("./align_2d", stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    f2d = p.stdin
    print >>f2d, len(o1)+len(o2)
    for a, b in zip(o1, o2):
      print >>f2d, " ".join(map(str, a))
      print >>f2d, " ".join(map(str, b))
    print >>f2d, len(o1c)+len(o2c)
    for a, b in zip(o1c, o2c):
      print >>f2d, " ".join(map(str, a))
      print >>f2d, " ".join(map(str, b))
    f2do, f2de = p.communicate()
    lines = f2do.strip().split('\n')
    print >>fo, ">%d_2d_rnn_simple" % i
    print >>fo, lines[0].strip()
    events_2d = []
    for l in lines[1:]:
      temp_ind, comp_ind = map(int, l.strip().split())
      e = []
      if temp_ind == -1:
        e += [0, 0, 0, 0, 0]
      else: 
        e += [1] + list(data["temp_events2"][temp_ind])
      if comp_ind == -1:
        e += [0, 0, 0, 0, 0]
      else:
        e += [1] + list(data["comp_events2"][comp_ind])
      events_2d.append(e)
    events_2d = np.array(events_2d, dtype=np.float32)
    predict_and_write(events_2d, big_net, fo, "%s_2d_rnn" % i)
