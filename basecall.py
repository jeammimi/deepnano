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

def load_read_data(read_file):
  h5 = h5py.File(read_file, "r")
  ret = {}
  
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
    # No timing
    pass

  base_loc = "Analyses/Basecall_2D_000"
  try:
    events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  except:
    base_loc = "Analyses/Basecall_1D_000"

  try:
    ret["called_template"] = h5[base_loc+"/BaseCalled_template/Fastq"][()].split('\n')[1]
    ret["called_complement"] = h5[base_loc+"/BaseCalled_complement/Fastq"][()].split('\n')[1]
    ret["called_2d"] = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Fastq"][()].split('\n')[1]
  except Exception as e:
    pass
  try:
    events = h5[base_loc+"/BaseCalled_template/Events"]
    ret["mp_template"] = []
    for e in events:
      if e["move"] == 1:
        ret["mp_template"].append(e["mp_state"][2])
      if e["move"] == 2:
        ret["mp_template"].append(e["mp_state"][1:3])
    ret["mp_template"] = "".join(ret["mp_template"])
    tscale = h5[base_loc+"/Summary/basecall_1d_template"].attrs["scale"]
    tscale_sd = h5[base_loc+"/Summary/basecall_1d_template"].attrs["scale_sd"]
    tshift = h5[base_loc+"/Summary/basecall_1d_template"].attrs["shift"]
    tdrift = h5[base_loc+"/Summary/basecall_1d_template"].attrs["drift"]
    index = 0.0
    ret["temp_events"] = []
    for e in events:
      mean = (e["mean"] - tshift - index * tdrift) / tscale
      stdv = e["stdv"] / tscale_sd
      length = e["length"]
      ret["temp_events"].append(preproc_event(mean, stdv, length))
      index += e["length"]
    ret["temp_events"] = np.array(ret["temp_events"], dtype=np.float32)
  except:
    pass

  try:
    events = h5[base_loc+"/BaseCalled_complement/Events"]
    cscale = h5[base_loc+"/Summary/basecall_1d_complement"].attrs["scale"]
    cscale_sd = h5[base_loc+"/Summary/basecall_1d_complement"].attrs["scale_sd"]
    cshift = h5[base_loc+"/Summary/basecall_1d_complement"].attrs["shift"]
    cdrift = h5[base_loc+"/Summary/basecall_1d_complement"].attrs["drift"]
    index = 0.0
    ret["comp_events"] = []
    for e in events:
      mean = (e["mean"] - cshift - index * cdrift) / cscale
      stdv = e["stdv"] / cscale_sd
      length = e["length"]
      ret["comp_events"].append(preproc_event(mean, stdv, length))
      index += e["length"]
    ret["comp_events"] = np.array(ret["comp_events"], dtype=np.float32)
  except:
    pass

  try:
    al = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Alignment"]
    temp_events = h5[base_loc+"/BaseCalled_template/Events"]
    comp_events = h5[base_loc+"/BaseCalled_complement/Events"]
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
  except:
    pass

  h5.close()
  return ret

parser = argparse.ArgumentParser()
parser.add_argument('--template_net', type=str, default="nets_data/map6temp.npz")
parser.add_argument('--complement_net', type=str, default="nets_data/map6comp.npz")
parser.add_argument('--big_net', type=str, default="nets_data/map6-2d-big.npz")
parser.add_argument('reads', type=str, nargs='*')
parser.add_argument('--timing', action='store_true', default=False)
parser.add_argument('--type', type=str, default="all", help="One of: template, complement, 2d, all, use comma to separate multiple options, eg.: template,complement")
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--output_orig', action='store_true', default=False)
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

if do_template:
  print "loading template net"
  temp_net = RnnPredictor(args.template_net)
  print "done"
if do_complement:
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

total_bases = [0, 0, 0]

files = args.reads
if len(args.directory):
  files += [os.path.join(args.directory, x) for x in os.listdir(args.directory)]  

for i, read in enumerate(files):
  basename = os.path.basename(read)
  try:
    data = load_read_data(read)
  except Exception as e:
    print "error at file", read
    print e
    continue
  if not data:  
    continue
  print "\rcalling read %d/%d %s" % (i, len(files), read),
  sys.stdout.flush()
  if args.output_orig:
    try:
      if "called_template" in data:
        print >>fo, ">%s_template" % basename
        print >>fo, data["called_template"]
      if "mp_template" in data:
        print >>fo, ">%s_mp_template" % basename
        print >>fo, data["mp_template"]
      if "called_complement" in data:
        print >>fo, ">%s_complement" % basename
        print >>fo, data["called_complement"]
      if "called_2d" in data:
        print >>fo, ">%s_2d" % basename
        print >>fo, data["called_2d"]
    except:
      pass

  temp_start = datetime.datetime.now()
  if do_template and "temp_events" in data:
    predict_and_write(data["temp_events"], temp_net, fo, "%s_template_rnn" % basename)
  temp_time = datetime.datetime.now() - temp_start

  comp_start = datetime.datetime.now()
  if do_complement and "comp_events" in data:
    predict_and_write(data["comp_events"], comp_net, fo, "%s_complement_rnn" % basename)
  comp_time = datetime.datetime.now() - comp_start

  start_2d = datetime.datetime.now()
  if do_2d and "2d_events" in data:
    predict_and_write(data["2d_events"], big_net, fo, "%s_2d_rnn" % basename) 
  time_2d = datetime.datetime.now() - start_2d

  if args.timing:
    try:
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
    except:
      # Don't let timing throw us out
      pass
  fo.flush()
fo.close()
