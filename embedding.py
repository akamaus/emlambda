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

if True or len(sys.argv) == 1:
    num_syms = 10
    code_width = 10
elif len(sys.argv) < 4:
    print_usage()
else:
    num_syms = int(sys.argv[1])
    code_width = int(sys.argv[2])

batch_size = 50
seq_sz = 3


def ensure_dir(d):
    """Creates dir if it doesn't exist"""
    if not os.path.exists(d):
        os.makedirs(d)


model = Model(num_syms=num_syms, code_width=code_width)

# some constants and initialization
eps = 1e-6
experiment = "exp-voc" + str(num_syms) + "-code" + str(code_width) + "-seq" + str(seq_sz)

f32 = tf.float32


def sym_list(list_len, different=False):
    """Generates a list of identifiers (to be used, for example to combine into single code)"""
    if different:
        lst = random.sample(model.symbols, list_len)
    else:
        lst = [random.choice(model.symbols) for x in range(list_len)]
    return model.one_hot(lst)


def sym_list_batch(list_len, batch, different=False):
    """Generates a batch of identifier sequence"""
    batch = [list(sym_list(list_len, different)) for x in range(batch)]
    return batch


def seq_coder(symbol_codes):
    """Takes list of codes representing symbol-code sequence
    folds it into a single code. Returns summary code and list of accumulated results"""
    tuples = []
    b = tf.shape(symbol_codes[0])[0]
    c = tf.tile(tf.expand_dims(model.empty_code(), axis=0), [b, 1])
    for ci in symbol_codes:
        tuples.append(c)
        c = model.tuple(c,ci)
    return c, tuples


def seq_decoder(seq_code, seq_length):
    """Takes a code representing symbol sequence and unfolds it"""
    tuples = []
    symbols = []
    for i in range(seq_length):
        seq_code,symbol = model.untuple(seq_code)
        tuples.append(seq_code)
        symbols.append(symbol)
    tuples.reverse()
    symbols.reverse()
    return symbols, tuples


def do_train():
    ensure_dir('checkpoints')
    ensure_dir('logs')

    # training equations (for tuple)
    ids = tf.placeholder(f32, [None, seq_sz, model.sym_width])
    ids_2d = tf.reshape(ids, [-1, model.sym_width])
    sym_codes = tf.reshape(model.embed(ids_2d), [-1, seq_sz, model.code_width])

    seqs = tf.transpose(sym_codes, perm=[1, 0, 2])  # Seq x Batch x Code
    seq_list = tf.unpack(seqs)
    code, tup_codes = seq_coder(seq_list)
    rev_seqs, rev_tup_codes = seq_decoder(code, seq_sz)
    seq_sqr_dist = tf.squared_difference(seqs, rev_seqs)
    tup_sqr_dist = tf.squared_difference(tup_codes, rev_tup_codes)
    seq_loss = tf.reduce_mean(seq_sqr_dist)
    tup_loss = tf.reduce_mean(tup_sqr_dist)
    tf.summary.scalar('seq_loss', seq_loss)
    tf.summary.scalar('tup_loss', tup_loss)
    tup_max = tf.sqrt(tf.reduce_max(tup_sqr_dist))
    seq_max = tf.sqrt(tf.reduce_max(seq_sqr_dist))
    tf.summary.scalar('seq_max', seq_max)
    tf.summary.scalar('tup_max', tup_max)

    # for coder
    pair_sz = 2
    diff_ids = tf.placeholder(f32, [None, pair_sz, model.sym_width])
    diff_codes = tf.reshape(model.embed(one_hot=tf.reshape(diff_ids, [-1, model.sym_width])), [-1, pair_sz, model.code_width])
    diff_pairs = tf.transpose(diff_codes, perm=[1,0,2])
    diff_cs = tf.unpack(diff_pairs)
    code_dist = tf.reduce_sum(tf.squared_difference(diff_cs[0], diff_cs[1]), 1)
    code_loss = tf.reduce_mean(1 / code_dist)
    tf.summary.scalar('code_loss', code_loss)
    code_min = tf.reduce_min(code_dist)
    tf.summary.scalar('code_min', code_min)

    # final loss, step and loop
    full_loss = seq_loss + tup_loss + code_loss  #  + det_cross_ent
    step = tf.train.MomentumOptimizer(0.01, 0.1).minimize(full_loss)

    experiment_date = experiment + "-" + strftime("%Y-%m-%d-%H%M%S", localtime())
    writer = tf.summary.FileWriter("logs/" + experiment_date, flush_secs=5)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        k = None
        for i in range(100000):
            k = i
            ids_v = sym_list_batch(seq_sz, batch_size, False)
            diff_ids_v = sym_list_batch(pair_sz, model.num_syms, True)

            _, bin_logs, logs = sess.run([step, summaries, (
            (seq_max, tup_max, code_min), (seq_loss, tup_loss, code_loss))],  # (det_cs1_acc, det_tups_acc, det_cross_ent)
                                         feed_dict={
                                             ids: ids_v,
                                             diff_ids: diff_ids_v
                                         })
            mlogs = logs[0]
            if mlogs[0] < eps and mlogs[1] < eps and mlogs[2] > 0.5:
                print("early stopping")
                break

            if i % 100 == 0:
                writer.add_summary(bin_logs, i)
                print(i, logs)

            if i % 1000 == 0:
                model.net_saver.save(sess, "checkpoints/" + experiment_date, global_step=k)
        model.net_saver.save(sess, "checkpoints/" + experiment_date, global_step=k)


def rest():
    cs = tf.matmul(model.Coder, tf.reshape(ids, [model.sym_width, -1] ))
    cs_lst = tf.split(1, 2, cs)
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


if __name__ != '__main__':
    print('doing nothing, runned in', sys.argv[0])
if len(sys.argv) == 1 or sys.argv[3] == 'train':
    do_train()
elif sys.argv[3] == 'test':
    checkpoint = sys.argv[4]
    do_test(checkpoint)
