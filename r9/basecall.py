from rnnf import Rnn
import h5py
import argparse
import os
import datetime
import numpy as np
from extract_events import extract_events


def scale(X):
    m25 = np.percentile(X[:, 0], 25)
    m75 = np.percentile(X[:, 0], 75)
    s50 = np.median(X[:, 2])
    me25 = 0.07499809
    me75 = 0.26622871
    se50 = 0.6103758
    ret = np.array(X)
    scale = (me75 - me25) / (m75 - m25)
    m25 *= scale
    shift = me25 - m25
    ret[:, 0] = X[:, 0] * scale + shift
    ret[:, 1] = ret[:, 0]**2

    sscale = se50 / s50

    ret[:, 2] = X[:, 2] * sscale
    return ret


def get_events(h5):
    if not args.event_detect:
        try:
            e = h5["Analyses/Basecall_RNN_1D_000/BaseCalled_template/Events"]
            return e
        except:
            pass
        try:
            e = h5["Analyses/Basecall_1D_000/BaseCalled_template/Events"]
            return e
        except:
            pass

    return extract_events(h5, args.chemistry)


def basecall(filename, output_file):
    # try:
    h5 = h5py.File(filename, "r")
    events = get_events(h5)
    if events is None:
        print "No events in file %s" % filename
        h5.close()
        return 0

    if len(events) < 300:
        print "Read %s too short, not basecalling" % filename
        h5.close()
        return 0

    # print(len(events))
    events = events[1:-1]
    mean = events["mean"]
    std = events["stdv"]
    length = events["length"]
    X = scale(np.array(np.vstack([mean, mean * mean, std, length]).T, dtype=np.float32))
    try:
        o1 = ntwk.predict(np.array(X)[np.newaxis, ::, ::])
        o1 = o1[0]
        #o2 = o2[0]
        # for i in o2[:20]:
        #print(["%.2f"%ii for ii in i])
        # print(o2)
    except:
        o1, o2 = ntwk.predict(X)

    # print(o1[:20])
    om = np.argmax(o1, axis=-1)
    # print(o2[:20])
    # exit()
    """

    o1m = (np.argmax(o1, 1))
    o2m = (np.argmax(o2, 1))
    om = np.vstack((o1m, o2m)).reshape((-1,), order='F')"""
    # print(om)
    output = "".join(map(lambda x: alph[x], om)).replace("N", "")
    print(om.shape, len(output))

    print >>output_file, ">%s_template_deepnano" % filename
    print >>output_file, output
    output_file.flush()

    h5.close()
    return len(events)
    # except Exception as e:
    print "Read %s failed with %s" % (filename, e)
    return 0


parser = argparse.ArgumentParser()
parser.add_argument('--chemistry', choices=['r9', 'r9.4'], default='r9.4')
parser.add_argument("--weights", default="None")
parser.add_argument('--Nbases', choices=["4", "5"], default='4')
parser.add_argument('--output', type=str, default="output.fasta")
parser.add_argument('--directory', type=str, default='',
                    help="Directory where read files are stored")
parser.add_argument('--watch', type=str, default='', help='Watched directory')
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--no-debug', dest='debug', action='store_false')
parser.add_argument('--event-detect', dest='event_detect', action='store_true')
parser.add_argument('reads', type=str, nargs='*')

parser.set_defaults(debug=False)
parser.set_defaults(event_detect=False)

args = parser.parse_args()
assert len(args.reads) != 0 or len(args.directory) != 0 or len(
    args.watch) != 0, "Nothing to basecall"

ntwks = {"r9": os.path.join("networks", "r9.pkl"), "r9.4": os.path.join("networks", "r94.pkl")}

alph = "ACGTN"
if args.Nbases == "5":
    alph = "ACGTBN"
classical = False
if classical:
    ntwk = Rnn()
    if args.weights == "None":
        ntwk.load(ntwks[args.chemistry])
    else:
        ntwk.load(args.weights)
else:
    import sys
    sys.path.append("../training/")

    from keras.models import load_model
    from rnnbisbis import model as ntwk
    ntwk.load_weights(args.weights)
    print("loaded")


if len(args.reads) or len(args.directory) != 0:
    fo = open(args.output, "w")

    files = args.reads
    if len(args.directory):
        files += [os.path.join(args.directory, x) for x in os.listdir(args.directory)]

    total_events = 0
    start_time = datetime.datetime.now()
    for i, read in enumerate(files):
        current_events = basecall(read, fo)
        if args.debug:
            total_events += current_events
            time_diff = (datetime.datetime.now() - start_time).seconds + 0.000001
            print "Basecalled %d events in %f (%f ev/s)" % (total_events, time_diff, total_events / time_diff)

    fo.close()

if len(args.watch) != 0:
    try:
        from watchdog.observers import Observer
        from watchdog.events import PatternMatchingEventHandler
    except:
        print "Please install watchdog to watch directories"
        sys.exit()

    class Fast5Handler(PatternMatchingEventHandler):
        """Class for handling creation fo fast5-files"""
        patterns = ["*.fast5"]

        def on_created(self, event):
            print "Calling", event
            file_name = str(os.path.basename(event.src_path))
            fasta_file_name = os.path.splitext(event.src_path)[0] + '.fasta'
            with open(fasta_file_name, "w") as fo:
                basecall(event.src_path, fo)
    print('Watch dir: ' + args.watch)
    observer = Observer()
    print('Starting Observerer')
    # start watching directory for fast5-files
    observer.start()
    observer.schedule(Fast5Handler(), path=args.watch)
    try:
        while True:
            time.sleep(1)
    # quit script using ctrl+c
    except KeyboardInterrupt:
        observer.stop()

    observer.join()
