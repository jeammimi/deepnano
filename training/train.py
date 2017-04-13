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
import keras
from Bio import pairwise2
from keras.utils.np_utils import to_categorical


def print_stats(o):
    stats = defaultdict(int)
    for x in o:
        stats[x] += 1
    print(stats)


def flatten2(x):
    return x.reshape((x.shape[0] * x.shape[1], -1))


def realign(s):
    ps = s
    try:
        o1, o2 = ntwk.tester(data_x[ps])
    except:
        o1, o2 = ntwk.predict(np.array([data_x[ps]]))
        o1 = o1[0]
        o2 = o2[0]
    o1m = (np.argmax(o1, 1))
    o2m = (np.argmax(o2, 1))
    f = open(base_dir + "tmpb-%s.in" % s, "w")
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

    f = open(base_dir + "tmpb-%s.out" % s)
    for i, l in enumerate(f):
        data_y[ps][i] = mapping[l[0]]
        data_y2[ps][i] = mapping[l[1]]

    return data_y[ps], data_y2[ps]

if __name__ == '__main__':

    data_x = []
    data_y = []
    data_y2 = []
    data_index = []
    data_alignment = []
    refs = []
    names = []

    if sys.argv[2] == "4":
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}  # Modif
    elif sys.argv[2] == "5":
        mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "B": 4, "N": 5}  # Modif
    else:
        print("Unkwon mapping length")
        exit()

    n_classes = len(mapping.keys())

    list_files = []
    subseq_size = 400

    for folder in sys.argv[3:]:
        list_files += glob.glob(folder + "/*")
    list_files = list_files

    load = True
    if load is None:
        for fn in list_files:
            print(fn)
            f = open(fn)
            ref = f.readline()
            ref = ref.replace("\n", "")
            if len(ref) > 30000:
                print "out", len(ref)
                continue

            X = []
            Y = []
            Y2 = []
            seq = []
            for l in f:
                its = l.strip().split()
                X.append(map(float, its[:-1]))
                Y.append(mapping[its[-1][0]])
                Y2.append(mapping[its[-1][1]])
                seq.append(its[-1])

            if len(X) < subseq_size:
                print "out (too small (to include must set a smaller subseq_size))", fn
                continue
            refs.append(ref.strip())
            names.append(fn)
            data_x.append(np.array(X, dtype=np.float32))
            data_y.append(np.array(Y, dtype=np.int32))
            data_y2.append(np.array(Y2, dtype=np.int32))
            seq = "".join(seq)
            data_index.append(np.arange(len(seq))[np.array([s for s in seq]) != "N"])
            seqs = seq.replace("N", "")
            data_alignment.append(pairwise2.align.globalxx(ref, seqs)[0][:2])
            print(len(seqs), len(ref))
        import cPickle
        with open("Allignements", "wb") as f:
            cPickle.dump([data_x, data_y, data_y2, data_index, data_alignment, refs, names], f)
    else:
        import cPickle
        with open("Allignements", "rb") as f:
            data_x, data_y, data_y2, data_index, data_alignment, refs, names = cPickle.load(f)

    print("done", sum(len(x) for x in refs))
    sys.stdout.flush()
    # print(len(refs[0]),len(data_x[0]),len(data_y[0]))
    # exit()

    s_arr = []
    p_arr = []
    for s in range(len(data_x)):
        s_arr += [s]
        p_arr += [len(data_x[s]) - subseq_size]

    sum_p = sum(p_arr)
    for i in range(len(p_arr)):
        p_arr[i] = 1. * p_arr[i] / sum_p

    base_dir = str(datetime.datetime.now())
    #base_dir = "compensate_bis"
    base_dir = base_dir.replace(' ', '_')

    os.mkdir(base_dir)
    base_dir += "/"
    batch_size = 1
    n_batches = len(data_x) / batch_size
    print len(data_x), batch_size, n_batches, datetime.datetime.now()

    classical = False
    if classical:
        ntwk = Rnn(sys.argv[1], n_classes=n_classes)

        print("net rdy")

        for epoch in range(1000):
            print("Epoch", epoch)
            if (epoch % 20 == 0 and epoch > 0):  # or (epoch == 0):
                p = Pool(5)
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
                x = data_x[s2][r:r + subseq_size]
                # x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) *
                # np.random.normal(scale=0.01, size=x.shape[0])
                y = data_y[s2][r:r + subseq_size]
                y2 = data_y2[s2][r:r + subseq_size]

                lr = 1e-2
        #    if epoch >= 3:
        #      lr = 2e-1
        #    if epoch >= 50:
        #      lr = 2e-1
                # print(x.shape,y.shape,y2.shape)

                if epoch >= 970:
                    lr = 1e-3

                if epoch < 0:
                    cost, o1, o2 = ntwk.trainer_reduced(x, y, y2, np.array(lr, dtype=np.float32))
                else:
                    # lr = 5e-3
                    cost, o1, o2 = ntwk.trainer(x, y, y2, np.array(lr, dtype=np.float32))

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

            print epoch, tc / n_batches, 1. * tc2 / n_batches / batch_size, 1. * tc3 / n_batches / batch_size, datetime.datetime.now()
            print_stats(o1mm)
            print_stats(o2mm)
            print confusion_matrix(y1mm, o1mm)
            print confusion_matrix(y2mm, o2mm)

        #  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
            sys.stdout.flush()

            if epoch % 20 == 2:
                ntwk.save(base_dir + "dumpx-%d.npz" % epoch)

    else:
        boring = False
        if boring:
            from rnnbis import model as ntwk
        else:
            from rnnbis import model2 as ntwk
            from rnnbis import model as predictor

        def find_closest(start, Index, factor=3.5):
            # Return the first element != N which correspond to the index of seqs
            start_index = min(int(start / factor), len(Index) - 1)
            # print(start,start_index,Index[start_index])
            if Index[start_index] >= start:
                while start_index >= 0 and Index[start_index] >= start:
                    start_index -= 1
                return max(0, start_index)

            if Index[start_index] < start:
                while start_index <= len(Index) - 1 and Index[start_index] < start:
                    start_index += 1
                if start_index <= len(Index) - 1 and start_index > 0:
                    if abs(Index[start_index] - start) > abs(Index[start_index - 1] - start):
                        start_index -= 1

                    # print(start_index,Index[start_index])
                # print(start_index,min(start_index,len(Index)-1),Index[min(start_index,len(Index)-1)])
                return min(start_index, len(Index) - 1)

        def get_segment(alignment, start_index_on_seqs, end_index_on_seqs):
            s1, s2 = alignment
            count = 0
            # print(s1,s2)
            startf = False
            for N, (c1, c2) in enumerate(zip(s1, s2)):
                # print(count)
                if count == start_index_on_seqs and not startf:
                    start = 0 + N
                    startf = True
                if count == end_index_on_seqs + 1:
                    end = 0 + N
                    break

                if c2 != "-":
                    count += 1
            # print(start,end)
            return s1[start:end].replace("-", "")

    # ntwk.load_weights("./my_model_weights.h5")
        for epoch in range(1000):
            print("Epoch", epoch)
            if (epoch % 4000 == 0 and epoch > 0):  # or (epoch == 0):
                p = Pool(5)
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
            X_new = []
            Y_new = []
            Y2_new = []
            Label = []
            Length = []
            stats = defaultdict(int)
            for s in range(len(data_x)):
                s2 = np.random.choice(s_arr, p=p_arr)
                r = np.random.randint(0, data_x[s2].shape[0] - subseq_size)
                x = data_x[s2][r:r + subseq_size]
                # x[:,0] += np.random.binomial(n=1, p=0.1, size=x.shape[0]) *
                # np.random.normal(scale=0.01, size=x.shape[0])

                def domap(base):
                    ret = [0 for b in range(n_classes)]
                    ret[base] = 1
                    return ret

                for xx in data_y2[s2][r:r + subseq_size]:
                    stats[xx] += 1

                y = [domap(base) for base in data_y[s2][r:r + subseq_size]]
                y2 = [domap(base) for base in data_y2[s2][r:r + subseq_size]]

                if not boring:
                    length = 2 * subseq_size
                    start = r
                    Index = data_index[s2]
                    alignment = data_alignment[s2]

                    start_index_on_seqs = find_closest(start, Index)
                    end_index_on_seqs = find_closest(start + length, Index)

                    s = get_segment(alignment, start_index_on_seqs, end_index_on_seqs)

                    maxi = 500
                    l = min(max(len(s), 1), maxi)
                    if l < 20:
                        continue
                    Length.append(l)

                    # print(len(s))
                    if len(s) > maxi:
                        s = s[:maxi]
                    s = s + "A" * (maxi - len(s))
                    if "B" in refs[s2]:
                        s = s.replace("T", "B")
                    # print(len(s))
                    # print(s)
                    # print([base for base in s])
                    Label.append([mapping[base] for base in s])
                X_new.append(x)
                Y_new.append(y)
                Y2_new.append(y2)

            X_new = np.array(X_new)
            Y_new = np.array(Y_new)
            Y2_new = np.array(Y2_new)
            Label = np.array(Label)
            Length = np.array(Length)
            print(X_new.shape, Y_new.shape)
            sum1 = 0
            for k in stats.keys():
                sum1 += stats[k]

            if epoch == 0:
                weight = [0 for k in stats.keys()]

                for k in stats.keys():
                    weight[k] = stats[k] / 1.0 / sum1
                    weight[k] = 1 / weight[k]
                weight = np.array(weight)
                weight = weight * len(stats.keys()) / np.sum(weight)
            # weight[4] *= 100

            w2 = []

            for y in Y2_new:
                w2.append([])
                for arr in y:
                    w2[-1].append(weight[np.argmax(arr)])

            w2 = np.array(w2)
            print(w2.shape)
            print(weight)
            print(Length)

            try:
                ntwk.fit(X_new, [Y_new, Y2_new], epochs=1, batch_size=10)
            except:
                # To balance class weight
                if boring:
                    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                                  patience=5, min_lr=0.1)
                    ntwk.fit(X_new, [Y_new, Y2_new], nb_epoch=1, batch_size=10, validation_split=0.05,
                             sample_weight={"out_layer2": w2}, callbacks=[reduce_lr])
                else:
                    print(Label)
                    print(X_new.shape, Label.shape, np.array(
                        [length] * len(Length)).shape, Length.shape)
                    if epoch == 0:
                        ntwk.load_weights("dumpx-22.npz")
                    if epoch > -1:
                        test = False
                        if test:
                            batch = 10
                            for i in range(0, 900, batch):
                                ntwk.fit([X_new[i:i + batch], Label[i:i + batch], np.array([2 * subseq_size] * len(Length))[i:i + batch], Length[i:i + batch]],
                                         Label[i:i + batch], nb_epoch=1, batch_size=batch)  # , validation_split=0.1)
                                p = predictor.evaluate(X_new[i:i + batch],
                                                       [Y_new[i:i + batch], Y2_new[i:i + batch]])

                                print(Length[i:i + batch], p)

                                if np.sum(np.isnan(p)) > 0:

                                    print("loading from old")
                                    # ntwk.load_weights("my_model_weights-1.h5")
                                    ntwk.load_weights("tmp_not_nan-%i.h5" % ((i - 40) % 900))
                                    predictor.load_weights("tmp_not_nan-%i.h5" % ((i - 40) % 900))
                                    #predictor.load_weights("tmp_not_nan-%i.h5" % ((i - 20) % 900))
                                    p = predictor.evaluate(X_new[i:i + batch],
                                                           [Y_new[i:i + batch], Y2_new[i:i + batch]])
                                    print(p)

                                    p = ntwk.evaluate([X_new[i:i + batch], Label[i:i + batch], np.array([length] * len(Length))[i:i + batch], Length[i:i + batch]],
                                                      Label[i:i + batch])  # , validation_split=0.1)
                                    print(p)

                                else:
                                    print("saving")
                                    ntwk.save_weights("tmp_not_nan-%i.h5" % i)
                            ntwk.save_weights(base_dir + '/my_model_weights-%i.h5' % epoch)
                        else:
                            print(len(data_x), np.mean(Length), np.max(Length))
                            ntwk.fit([X_new, Label, np.array([2 * subseq_size] * len(Length)), Length],
                                     Label, nb_epoch=1, batch_size=10)  # , validation_split=0.1)
                            p = predictor.evaluate(X_new, [Y_new, Y2_new])
                            print(p)
                            if np.sum(np.isnan(p)) > 0:
                                d = 0
                                while np.sum(np.isnan(p)) > 0 and epoch - d > 0:
                                    d -= 1
                                    ntwk.load_weights("my_model_weights-%i.h5" % (epoch - d))
                                    p = predictor.evaluate(X_new, [Y_new, Y2_new])

                                    print("my_model_weights-%i.h5" % (epoch - d))
                                    print(p)

                                if np.sum(np.isnan(p)) > 0:
                                    exit()

                                # exit()
                            else:
                                ntwk.save_weights(base_dir + '/my_model_weights-%i.h5' % epoch)

                    else:
                        reduce_lr = keras.callbacks.ReduceLROnPlateau(
                            monitor='val_loss', factor=0.2, patience=5, min_lr=0.1)

                        predictor.fit(X_new, [Y_new, Y2_new], nb_epoch=1, batch_size=10, validation_split=0.05,
                                      sample_weight={"out_layer2": w2}, callbacks=[reduce_lr])
                        if i % 10 == 0:
                            predictor.save_weights(
                                base_dir + '/pred-my_model_weights-%i.h5' % epoch)

                # if epoch == 0:
                #    ntwk.fit(X_new,[Y_new,Y2_new],nb_epoch=1, batch_size=10,validation_split=0.05)
            if i % 10 == 0:
                ntwk.save_weights(base_dir + '/my_model_weights-%i.h5' % epoch)

            print epoch, tc / n_batches, 1. * tc2 / n_batches / batch_size, 1. * tc3 / n_batches / batch_size, datetime.datetime.now()
            print_stats(o1mm)
            print_stats(o2mm)
            print confusion_matrix(y1mm, o1mm)
            print confusion_matrix(y2mm, o2mm)

        #  print "out", np.min(out_gc), np.median(out_gc), np.max(out_gc), len(out_gc)
            sys.stdout.flush()

            if epoch % 20 == 2:
                ntwk.save(base_dir + "dumpx-%d.npz" % epoch)
