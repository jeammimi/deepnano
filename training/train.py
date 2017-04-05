from rnn import Rnn
import pickle
import sys
import numpy as np
import datetime
from collections import defaultdict
import os
from sklearn.metrics import confusion_matrix
import theano as th
from multiprocessing import Pool
import glob





def print_stats(o):
  stats = defaultdict(int)
  for x in o:
    stats[x] += 1
  print (stats)

def flatten2(x):
  return x.reshape((x.shape[0] * x.shape[1], -1))

def realign(s):
  ps = s
  o1, o2 = ntwk.tester(data_x[ps])
  o1m = (np.argmax(o1, 1))
  o2m = (np.argmax(o2, 1))
  f = open(base_dir+"tmpb-%s.in" % s, "w")
  print >>f, refs[ps]
  for a, b in zip(o1, o2):
    print >>f, " ".join(map(str, a))
    print >>f, " ".join(map(str, b))
  f.close()

  print "s", s
  if n_classes == 6:
      if os.system("./realign_five <%stmpb-%s.in >%stmpb-%s.out" % (base_dir, s, base_dir, s)) != 0:
        print "watwat", s
        sys.exit()
  elif n_classes == 5:
     if os.system("./realign_four <%stmpb-%s.in >%stmpb-%s.out" % (base_dir, s, base_dir, s)) != 0:
       print "watwat", s
       sys.exit()

  f = open(base_dir+"tmpb-%s.out" % s)
  for i, l in enumerate(f):
    data_y[ps][i] = mapping[l[0]]
    data_y2[ps][i] = mapping[l[1]]

  return data_y[ps], data_y2[ps]

if __name__ == '__main__':


  data_x = []
  data_y = []
  data_y2 = []
  refs = []
  names = []

  if sys.argv[2] == "4":
      mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4} #Modif
  elif sys.argv[2] == "5":
      mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B" : 4, "N": 5} #Modif
  else:
      print("Unkwon mapping length")
      exit()

  n_classes = len(mapping.keys())

  list_files = []
  for folder in sys.argv[3:]:
      list_files += glob.glob(folder + "/*")
  for fn in list_files:
    print(fn)
    f = open(fn)
    ref = f.readline()
    if len(ref) > 30000:
      print "out", len(ref)
      continue
    refs.append(ref.strip())
    names.append(fn)
    X = []
    Y = []
    Y2 = []
    for l in f:
      its = l.strip().split()
      X.append(map(float, its[:-1]))
      Y.append(mapping[its[-1][0]])
      Y2.append(mapping[its[-1][1]])
    data_x.append(np.array(X, dtype=np.float32))
    data_y.append(np.array(Y, dtype=np.int32))
    data_y2.append(np.array(Y2, dtype=np.int32))

  print ("done", sum(len(x) for x in refs))
  sys.stdout.flush()
  #print(len(refs[0]),len(data_x[0]),len(data_y[0]))
  #exit()
  ntwk = Rnn(sys.argv[1], n_classes=n_classes)

  print ("net rdy")

  s_arr = []
  p_arr = []
  subseq_size = 400
  for s in range(len(data_x)):
    s_arr += [s]
    p_arr += [len(data_x[s]) - subseq_size]

  sum_p = sum(p_arr)
  for i in range(len(p_arr)):
    p_arr[i] = 1.*p_arr[i] / sum_p

  base_dir = str(datetime.datetime.now())
  base_dir = base_dir.replace(' ', '_')

  os.mkdir(base_dir)
  base_dir += "/"
  batch_size = 1
  n_batches = len(data_x) / batch_size
  print len(data_x), batch_size, n_batches, datetime.datetime.now()

  for epoch in range(1000):
    print("Epoch",epoch)
    if (epoch % 20 == 0 and epoch > 0) or (epoch == 0):
      p = Pool(8)
      new_labels = p.map(realign, range(len(data_x)))
      for i in range(len(new_labels)):
        data_y[i] = new_labels[i][0]
        data_y2[i] = new_labels[i][1]

    taken_gc = []
    out_gc = []
    tc = 0
    tc2 = 0
    tc3 = 0
    o1mm = []
    y1mm = []
    o2mm = []
    y2mm = []
    for s in range(len(data_x)):
      s2 = np.random.choice(s_arr, p=p_arr)
      r = np.random.randint(0, data_x[s2].shape[0] - subseq_size)
      x = data_x[s2][r:r+subseq_size]
      #    x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) * np.random.normal(scale=0.01, size=x.shape[0])
      y = data_y[s2][r:r+subseq_size]
      y2 = data_y2[s2][r:r+subseq_size]

      lr = 1e-2
  #    if epoch >= 3:
  #      lr = 2e-1
  #    if epoch >= 50:
  #      lr = 2e-1
      #print(x.shape,y.shape,y2.shape)
      if epoch >= 970:
        lr = 1e-3
      if epoch < 100:
          cost, o1, o2 = ntwk.trainer_reduced(x, y, y2, np.array(lr,dtype=np.float32))
      else:
          lr = 5e-3
          cost, o1, o2 = ntwk.trainer(x, y, y2, np.array(lr,dtype=np.float32))


      tc += cost

      o1m = (np.argmax(o1, 1))
      o2m = (np.argmax(o2, 1))

      o1mm += list(o1m)
      o2mm += list(o2m)
      y1mm += list(y)
      y2mm += list(y2)

      tc2 += np.sum(np.equal(o1m, y))
      tc3 += np.sum(np.equal(o2m, y2))
      sys.stdout.write('\r%d' % s)
      sys.stdout.flush()

    print

    print epoch, tc / n_batches, 1.*tc2 / n_batches / batch_size, 1.*tc3 / n_batches / batch_size, datetime.datetime.now()
    print_stats(o1mm)
    print_stats(o2mm)
    print confusion_matrix(y1mm, o1mm)
    print confusion_matrix(y2mm, o2mm)

  #  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
    sys.stdout.flush()

    if epoch % 20 == 2:
      ntwk.save(base_dir+"dumpx-%d.npz" % epoch)
