#!/usr/bin/env python3
from time import localtime, strftime

import os
import sys
import tensorflow as tf
import numpy as np
import random

from model import Model


# cmd line params
def print_usage():
    print("Usage:")
    print(sys.argv[0] + "<vocabulary size> <code size> train")
    print(sys.argv[0] + "<vocabulary size> <code size> test <checkpoint file>")
    sys.exit(1)

if len(sys.argv) == 1:
    num_syms = 10
    code_width = 3
elif len(sys.argv) < 4:
    print_usage()
else:
    num_syms = int(sys.argv[1])
    code_width = int(sys.argv[2])

model = Model(num_syms=num_syms, code_width=code_width)

# some constants and initialization
eps = 1e-6
experiment = "exp-voc" + str(num_syms) + "-code" + str(code_width)

batch_size = 100
f32 = tf.float32


def id_pair(different):
    if different:
        a,b = random.sample(model.symbols,2)
    else:
        a = random.choice(model.symbols)
        b = random.choice(model.symbols)
    return model.sym_dict[a], model.sym_dict[b]


def id_pairs(batch, different):
    pairs = [id_pair(different) for x in range(batch)]
    t_pairs = list(zip(*pairs))
    ids1_v = np.transpose(t_pairs[0])
    ids2_v = np.transpose(t_pairs[1])
    return ids1_v, ids2_v


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def do_train():
    ensure_dir('checkpoints')
    ensure_dir('logs')

    # training equations (for tuple)
    ids = tf.placeholder(f32, [2, model.sym_width, None])

    cs = tf.matmul(model.Coder, tf.reshape(ids, [model.sym_width, -1] ))
    cs_lst = tf.split(1,2,cs)
    print(tf.shape(ids))
    print(tf.shape(cs))
    print(tf.shape(cs_lst[0]))

    tups = tf.matmul(model.Tuple, tf.concat(0,cs_lst))

    cs1_rev = tf.matmul(model.UnTuple1,tups)
    cs2_rev = tf.matmul(model.UnTuple2,tups)

    cs1_sd = tf.reduce_sum(tf.squared_difference(cs_lst[0],cs1_rev),0)
    cs2_sd = tf.reduce_sum(tf.squared_difference(cs_lst[1],cs2_rev),0)

    cs1_loss = tf.reduce_mean(cs1_sd)
    cs2_loss = tf.reduce_mean(cs2_sd)
    tf.summary.scalar('cs1_loss', cs1_loss)
    tf.summary.scalar('cs2_loss', cs2_loss)
    cs1_max = tf.sqrt(tf.reduce_max(cs1_sd))
    cs2_max = tf.sqrt(tf.reduce_max(cs2_sd))
    tf.summary.scalar('cs1_loss_max', cs1_max)
    tf.summary.scalar('cs2_loss_max', cs2_max)

    # for detector
    det_cs1 = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(model.TypeDetector, cs_lst[0]))))
    det_tups = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(model.TypeDetector, tups))))

    det_cross_ent = tf.reduce_mean( - tf.reduce_sum([[1],[0]] * tf.log(det_cs1), reduction_indices=0)) + tf.reduce_mean( - tf.reduce_sum([[0],[1]] * tf.log(det_tups), reduction_indices=0))
    tf.summary.scalar('det_cross_ent', det_cross_ent)
    det_cs1_max = tf.reduce_sum(tf.argmax(det_cs1,0))
    det_tups_max = tf.reduce_sum(tf.argmax(det_tups,0))
    det_cs1_acc = (batch_size - det_cs1_max) / batch_size
    det_tups_acc = (det_tups_max) / batch_size
    tf.summary.scalar('det_cs_acc', det_cs1_acc)
    tf.summary.scalar('det_tups_acc', det_tups_acc)

    # for coding
    diff_ids1 = tf.placeholder(f32, [model.sym_width, None])
    diff_ids2 = tf.placeholder(f32, [model.sym_width, None])

    c1 = tf.matmul(model.Coder, diff_ids1)
    c2 = tf.matmul(model.Coder, diff_ids2)

    code_sqrt = tf.sqrt(tf.reduce_sum(tf.squared_difference(c1, c2),0))
    code_loss = tf.reduce_mean(1 / code_sqrt)
    tf.summary.scalar('code_loss', code_loss)
    code_min = tf.reduce_min(code_sqrt)
    tf.summary.scalar('code_min', code_min)

    full_loss = cs1_loss + cs2_loss + code_loss + det_cross_ent
    step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(full_loss)

    experiment_date = experiment + "-" + strftime("%Y-%m-%d-%H%M%S", localtime())
    writer = tf.summary.FileWriter("logs/" + experiment_date, flush_secs=5)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        k = None
        for i in range(100000-1):
            k = i
            ids1_v, ids2_v = id_pairs(batch_size, False)
            diff_ids1_v, diff_ids2_v = id_pairs(num_syms, True)

            _, bin_logs, logs = sess.run([step, summaries, ((cs1_max, cs2_max, code_min), (cs1_loss, cs2_loss, code_loss), (det_cs1_acc, det_tups_acc, det_cross_ent))], feed_dict = {
                ids: [ids1_v, ids2_v],
                diff_ids1: diff_ids1_v, diff_ids2: diff_ids2_v
            })
            mlogs = logs[0]
            if mlogs[0] < eps and mlogs[1] < eps and mlogs[2] > 0.5:
                print("early stopping")
                break

            if i % 100 == 0:
                writer.add_summary(bin_logs, i)
                print(i, logs)

            if i % 1000 == 0:
                model.net_saver.save(sess, "checkpoints/" + experiment_date, global_step = k)
        model.net_saver.save(sess, "checkpoints/" + experiment_date, global_step = k)

ntop = 5
tries = 10
def do_test(snapshot):
    p_all_ids = tf.placeholder(f32, [sym_width, num_syms])

    p_id1 = tf.placeholder(f32, [sym_width, 1])
    p_id2 = tf.placeholder(f32, [sym_width, 1])

    c1 = tf.matmul(Coder, p_id1)
    c2 = tf.matmul(Coder, p_id2)

    tup = tf.matmul(Tuple, tf.concat(0,[c1,c2]))

    c1_rev = tf.matmul(UnTuple1,tup)
    c2_rev = tf.matmul(UnTuple2,tup)

    all_cs = tf.matmul(Coder, p_all_ids)

    diff1 = tf.sqrt(tf.squared_difference(all_cs, c1_rev))
    values, entries = tf.nn.top_k(-diff1, ntop )

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        net_saver.restore(sess, snapshot)

        successes1 = 0
        confidence_neg = 0
        confidence_pos = 0
        for k in range(tries):
            pair = id_pairs(1, False)
            v_values, v_entries = sess.run([values, entries], feed_dict = {
                p_id1 : pair[0],
                p_id2 : pair[1],
                p_all_ids : np.transpose(list(sym_dict.values()))
                }
            )
            if np.argmax(pair[0]) == v_entries[0,0]:
                successes1 += 1
                confidence_pos += v_values[0,1] / v_values[0,0]
            else:
                confidence_neg += v_values[0,1] / v_values[0,0]

            for i in range(ntop):
                e = v_entries[0,i]
                v = -v_values[0,i]
                print('sym', np.argmax(pair[0]), 'restored', e, 'value', v)

        print('successes: ', successes1, 'of', tries)
        if successes1 > 0:
            print('confidence_pos', confidence_pos / successes1)
        if successes1 < tries:
            print('confidence_neg', confidence_neg / (tries - successes1))




if len(sys.argv) == 1 or sys.argv[3] == 'train':
    do_train()
elif sys.argv[3] == 'test':
    checkpoint = sys.argv[4]
    do_test(checkpoint)
