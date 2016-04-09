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

def preproc_event(mean, std, length):
  mean = mean / 100.0 - 0.66
  std = std - 1
  return [mean, mean*mean, std, length]

def get_scaling_template(events):
  down = 48.4631279889
  up = 65.7312554591
  our_down = np.percentile(events["mean"], 10)
  our_up = np.percentile(events["mean"], 90)
  scale = (our_up - our_down) / (up - down)
  shift = (our_up / scale - up) * scale

  sd = 0.807981325017
  return scale, np.percentile(events["stdv"], 50) / sd, shift

def get_scaling_complement(events):
  down = 49.2638926877
  up = 69.0192568072
  our_down = np.percentile(events["mean"], 10)
  our_up = np.percentile(events["mean"], 90)
  scale = (our_up - our_down) / (up - down)
  shift = (our_up / scale - up) * scale

  sd = 1.04324844612
  return scale, np.percentile(events["stdv"], 50) / sd, shift

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

#  print "temp_comp_loc", temp_comp_loc["temp"], temp_comp_loc["comp"]
#  print h5["Analyses/Basecall_2D_000/Summary/split_hairpin"].attrs["start_index_temp"],
#  print h5["Analyses/Basecall_2D_000/Summary/split_hairpin"].attrs["end_index_temp"],
#  print h5["Analyses/Basecall_2D_000/Summary/split_hairpin"].attrs["start_index_comp"],
#  print h5["Analyses/Basecall_2D_000/Summary/split_hairpin"].attrs["end_index_comp"]

  sampling_rate = h5["UniqueGlobalKey/channel_id"].attrs["sampling_rate"]

  try:
    ret["called_template"] = h5["Analyses/Basecall_2D_000/BaseCalled_template/Fastq"][()].split('\n')[1]
    ret["called_complement"] = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Fastq"][()].split('\n')[1]
    ret["called_2d"] = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Fastq"][()].split('\n')[1]
  except Exception as e:
    print "wat", e 
    return None
  events = base_events[temp_comp_loc["temp"][0]:temp_comp_loc["temp"][1]]
  tscale2, tscale_sd2, tshift2 = get_scaling_template(events)

  index = 0.0
  ret["temp_events2"] = []
  for e in events:
    mean = (e["mean"] - tshift2) / tscale2
    stdv = e["stdv"] / tscale_sd2
    length = e["length"] / sampling_rate
    ret["temp_events2"].append(preproc_event(mean, stdv, length))
  events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  tscale = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["scale"]
  tscale_sd = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["scale_sd"]
  tshift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["shift"]
  tdrift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["drift"]
  index = 0.0
  ret["temp_events"] = []
  for e in events:
    mean = (e["mean"] - tshift - index * tdrift) / tscale
    stdv = e["stdv"] / tscale_sd
    length = e["length"]
    ret["temp_events"].append(preproc_event(mean, stdv, length))
    index += e["length"]

  events = base_events[temp_comp_loc["comp"][0]:temp_comp_loc["comp"][1]]
  cscale2, cscale_sd2, cshift2 = get_scaling_complement(events)

  index = 0.0
  ret["comp_events2"] = []
  for e in events:
    mean = (e["mean"] - cshift2) / cscale2
    stdv = e["stdv"] / cscale_sd2
    length = e["length"] / sampling_rate
    ret["comp_events2"].append(preproc_event(mean, stdv, length))

  events = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Events"]
  cscale = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["scale"]
  cscale_sd = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["scale_sd"]
  cshift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["shift"]
  cdrift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["drift"]
  index = 0.0
  ret["comp_events"] = []
  for e in events:
    mean = (e["mean"] - cshift - index * cdrift) / cscale
    stdv = e["stdv"] / cscale_sd
    length = e["length"]
    ret["comp_events"].append(preproc_event(mean, stdv, length))
    index += e["length"]

  ret["temp_events2"] = np.array(ret["temp_events2"], dtype=np.float32)
  ret["comp_events2"] = np.array(ret["comp_events2"], dtype=np.float32)
  ret["temp_events"] = np.array(ret["temp_events"], dtype=np.float32)
  ret["comp_events"] = np.array(ret["comp_events"], dtype=np.float32)

  al = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Alignment"]
  ret["al"] = al
  temp_events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  comp_events = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Events"]
  ret["2d_events"] = []
  for a in al:
    ev = []
    if a[0] == -1:
      ev += [0, 0, 0, 0, 0]
    else:
      e = temp_events[a[0]]
      mean = (e["mean"] - tshift - index * tdrift) / cscale
      stdv = e["stdv"] / tscale_sd
      length = e["length"]
      ev += [1] + preproc_event(mean, stdv, length)
    if a[1] == -1:
      ev += [0, 0, 0, 0, 0]
    else:
      e = comp_events[a[1]]
      mean = (e["mean"] - cshift - index * cdrift) / cscale
      stdv = e["stdv"] / cscale_sd
      length = e["length"]
      ev += [1] + preproc_event(mean, stdv, length)
    ret["2d_events"].append(ev) 
  ret["2d_events"] = np.array(ret["2d_events"], dtype=np.float32)
  return ret

parser = argparse.ArgumentParser()
parser.add_argument('--template_net', type=str, default="nets_data/map6temp.npz")
parser.add_argument('--complement_net', type=str, default="nets_data/map6comp.npz")
parser.add_argument('--big_net', type=str, default="nets_data/map6-2d-big.npz")
parser.add_argument('reads', type=str, nargs='+')
parser.add_argument('--type', type=str, default="all", help="One of: template, complement, 2d, all, use comma to separate multiple options, eg.: template,complement")
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--output_orig', action='store_true', default=True)

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
  big_net_orig = RnnPredictor("nets_data/map6-2d-big.npz")
  print "done"

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

fo = open(args.output, "w")

total_bases = [0, 0, 0]

for i, read in enumerate(args.reads):
  if True:
    data = load_read_data(read)
#  except Exception as e:
#    print e
#    print "error at file", read
#    continue
  if not data:  
    continue
  if args.output_orig:
    print >>fo, ">%d_template" % i
    print >>fo, data["called_template"]
    print >>fo, ">%d_complement" % i
    print >>fo, data["called_complement"]
    print >>fo, ">%d_2d" % i
    print >>fo, data["called_2d"]

  if do_template or do_2d:
    o1, o2 = temp_net.predict(data["temp_events"]) 
    o1m = (np.argmax(o1, 1))
    o2m = (np.argmax(o2, 1))
    print >>fo, ">%d_temp_rnn" % i
    for a, b in zip(o1m, o2m):
      if a < 4:
        fo.write(chars[a])
      if b < 4:
        fo.write(chars[b])
    fo.write('\n')
    o1, o2 = temp_net.predict(data["temp_events2"]) 
    o1m = (np.argmax(o1, 1))
    o2m = (np.argmax(o2, 1))
    if do_template:
      print >>fo, ">%d_temp_rnn2" % i
      for a, b in zip(o1m, o2m):
        if a < 4:
          fo.write(chars[a])
        if b < 4:
          fo.write(chars[b])
      fo.write('\n')

  if do_complement or do_2d:
    o1c, o2c = comp_net.predict(data["comp_events"]) 
    o1cm = (np.argmax(o1c, 1))
    o2cm = (np.argmax(o2c, 1))
    print >>fo, ">%d_comp_rnn" % i
    for a, b in zip(o1cm, o2cm):
      if a < 4:
        fo.write(chars[a])
      if b < 4:
        fo.write(chars[b])
    fo.write('\n')
    o1c, o2c = comp_net.predict(data["comp_events2"]) 
    o1cm = (np.argmax(o1c, 1))
    o2cm = (np.argmax(o2c, 1))
    if do_complement:
      print >>fo, ">%d_comp_rnn2" % i
      for a, b in zip(o1cm, o2cm):
        if a < 4:
          fo.write(chars[a])
        if b < 4:
          fo.write(chars[b])
      fo.write('\n')

  if do_2d:
    f2d = open("2d.in", "w")
    print >>f2d, len(o1)+len(o2)
    for a, b in zip(o1, o2):
      print >>f2d, " ".join(map(str, a))
      print >>f2d, " ".join(map(str, b))
    print >>f2d, len(o1c)+len(o2c)
    for a, b in zip(o1c, o2c):
      print >>f2d, " ".join(map(str, a))
      print >>f2d, " ".join(map(str, b))
    f2d.close()
    os.system("./align_2d <2d.in >2d.out")
    f2do = open("2d.out")
    call2d = f2do.next().strip()
    print >>fo, ">%d_2d_rnn_simple" % i
    print >>fo, call2d

    start_temp_ours = None
    end_temp_ours = None
    start_comp_ours = None
    end_comp_ours = None
    events_2d = []
    for l in f2do:
      temp_ind, comp_ind = map(int, l.strip().split())
      e = []
      if temp_ind == -1:
        e += [0, 0, 0, 0, 0]
      else: 
        e += [1] + list(data["temp_events2"][temp_ind])
        if not start_temp_ours:
          start_temp_ours = temp_ind
        end_temp_ours = temp_ind
      if comp_ind == -1:
        e += [0, 0, 0, 0, 0]
      else:
        e += [1] + list(data["comp_events2"][comp_ind])
        if not end_comp_ours:
          end_comp_ours = comp_ind
        start_comp_ours = comp_ind
      events_2d.append(e)
    events_2d = np.array(events_2d, dtype=np.float32)
    o1c, o2c = big_net.predict(events_2d) 
    o1cm = (np.argmax(o1c, 1))
    o2cm = (np.argmax(o2c, 1))
    print >>fo, ">%d_2d_rnn2" % i
    for a, b in zip(o1cm, o2cm):
      if a < 4:
        fo.write(chars[a])
      if b < 4:
        fo.write(chars[b])
    fo.write('\n')
    o1c, o2c = big_net.predict(data["2d_events"]) 
    o1cm = (np.argmax(o1c, 1))
    o2cm = (np.argmax(o2c, 1))
    print >>fo, ">%d_2d_rnn" % i
    for a, b in zip(o1cm, o2cm):
      if a < 4:
        fo.write(chars[a])
      if b < 4:
        fo.write(chars[b])
    fo.write('\n')

    start_temp_th = None
    end_temp_th = None
    start_comp_th = None
    end_comp_th = None
    for a in data["al"]: 
      if a[0] != -1:
        if not start_temp_th:
          start_temp_th = a[0]
        end_temp_th = a[0]
      if a[1] != -1:
        if not end_comp_th:
          end_comp_th = a[1]
        start_comp_th = a[1]

    print "Ours:",
    print start_temp_ours, end_temp_ours, start_comp_ours, end_comp_ours,
    print 1. * len(events_2d) / (end_temp_ours - start_temp_ours + end_comp_ours - start_comp_ours) 
    print "Their:",
    print start_temp_th, end_temp_th, start_comp_th, end_comp_th,
    print 1. * len(data["al"]) / (end_temp_th - start_temp_th + end_comp_th - start_comp_th) 
    print
