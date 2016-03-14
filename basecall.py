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

def load_read_data(read_file):
  h5 = h5py.File(read_file, "r")
  ret = {}
  
  log = h5["Analyses/Basecall_2D_000/Log"][()]
  temp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling template.*", log).groups()[0])
  comp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling complement.*", log).groups()[0])
  comp_end_time = dateutil.parser.parse(re.search(r"(.*) Aligning hairpin.*", log).groups()[0])

  start_2d_time = dateutil.parser.parse(re.search(r"(.*) Performing full 2D.*", log).groups()[0])
  end_2d_time = dateutil.parser.parse(re.search(r"(.*) Workflow completed.*", log).groups()[0])

  ret["temp_time"] = comp_time - temp_time
  ret["comp_time"] = comp_end_time - comp_time
  ret["2d_time"] = end_2d_time - start_2d_time

  try:
    ret["called_template"] = h5["Analyses/Basecall_2D_000/BaseCalled_template/Fastq"][()].split('\n')[1]
    ret["called_complement"] = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Fastq"][()].split('\n')[1]
    ret["called_2d"] = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Fastq"][()].split('\n')[1]
  except Exception as e:
    print "wat", e 
    return None
  events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  ret["mp_template"] = []
  for e in events:
    if e["move"] == 1:
      ret["mp_template"].append(e["mp_state"][2])
    if e["move"] == 2:
      ret["mp_template"].append(e["mp_state"][1:3])
  ret["mp_template"] = "".join(ret["mp_template"])
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
  ret["temp_events"] = np.array(ret["temp_events"], dtype=np.float32)
  ret["comp_events"] = np.array(ret["comp_events"], dtype=np.float32)

  al = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Alignment"]
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
parser.add_argument('--timing', action='store_true', default=False)
parser.add_argument('--type', type=str, default="all", help="One of: template, complement, 2d, all, use comma to separate multiple options, eg.: template,complement")
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--output_orig', action='store_true', default=False)

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

if do_template:
  temp_net = RnnPredictor(args.template_net)
if do_complement:
  comp_net = RnnPredictor(args.complement_net)
if do_2d:
  big_net = RnnPredictor(args.big_net)

chars = "ACGT"
mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

fo = open(args.output, "w")

total_bases = [0, 0, 0]

for i, read in enumerate(args.reads):
  data = load_read_data(read)
  if not data:  
    continue
  if args.output_orig:
    print >>fo, ">%d_template" % i
    print >>fo, data["called_template"]
    print >>fo, ">%d_mp_template" % i
    print >>fo, data["mp_template"]
    print >>fo, ">%d_complement" % i
    print >>fo, data["called_complement"]
    print >>fo, ">%d_2d" % i
    print >>fo, data["called_2d"]

  temp_start = datetime.datetime.now()
  if do_template:
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
  temp_time = datetime.datetime.now() - temp_start

  comp_start = datetime.datetime.now()
  if do_complement:
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
  comp_time = datetime.datetime.now() - comp_start

  start_2d = datetime.datetime.now()
  if do_2d:
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
  time_2d = datetime.datetime.now() - start_2d

  if args.timing:
    print "Events: %d/%d" % (len(data["temp_events"]), len(data["comp_events"]))
    print "Our times: %f/%f/%f" % (temp_time.total_seconds(), comp_time.total_seconds(),
       time_2d.total_seconds())
    print "Our times per base: %f/%f/%f" % (
      temp_time.total_seconds() / len(data["temp_events"]),
      comp_time.total_seconds() / len(data["comp_events"]),
      time_2d.total_seconds() / (len(data["comp_events"]) + len(data["temp_events"])))
    print "Their times: %f/%f/%f" % (data["temp_time"].total_seconds(), data["comp_time"].total_seconds(), data["2d_time"].total_seconds())
    print "Their times per base: %f/%f/%f" % (
      data["temp_time"].total_seconds() / len(data["temp_events"]),
      data["comp_time"].total_seconds() / len(data["comp_events"]),
      data["2d_time"].total_seconds() / (len(data["comp_events"]) + len(data["temp_events"])))

