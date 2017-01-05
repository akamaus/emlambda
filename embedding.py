#!/usr/bin/env python3
from time import localtime, strftime

import os
import tensorflow as tf
import random

from model import Model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('steps', 100000, 'Number of steps until stop')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Number of examples in mini-batch')

tf.app.flags.DEFINE_integer('num_symbols', 8, 'Atomic symbols number')
tf.app.flags.DEFINE_integer('code_width', 5, 'Number of embedding dimensions')
tf.app.flags.DEFINE_integer('seq_len', 2, 'Maximal length of symbol sequences to learn')

# Model
model = Model(num_symbols=FLAGS.num_symbols, code_width=FLAGS.code_width)

# some constants and initialization
f32 = tf.float32
pair_sz = 2
eps = 1e-6
experiment = "exp-voc" + str(FLAGS.num_symbols) + "-code" + str(FLAGS.code_width) + "-seq" + str(FLAGS.seq_len)


def ensure_dir(d):
    """Creates dir if it doesn't exist"""
    if not os.path.exists(d):
        os.makedirs(d)


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


def seq_coder_l(symbol_codes):
    """Takes list of codes representing symbol-code sequence
    folds it into a single code. Returns summary code and list of accumulated results"""
    tuples = []
    with tf.name_scope('seq_coder_l'):
        b = tf.shape(symbol_codes[0])[0]
        c = tf.tile(tf.expand_dims(model.empty_code(), axis=0), [b, 1])
        for ci in symbol_codes:
            tuples.append(c)
            c = model.tuple(c,ci)
        return c, tuples


def seq_decoder_l(seq_code, seq_length):
    """Takes a code representing symbol sequence and unfolds it"""
    tuples = []
    symbols = []
    with tf.name_scope('seq_decoder_l'):
        for i in range(seq_length):
            seq_code,symbol = model.untuple(seq_code)
            tuples.append(seq_code)
            symbols.append(symbol)
    tuples.reverse()
    symbols.reverse()
    return symbols, tuples


def seq_coder_r(symbol_codes):
    """Foldr. Returns summary code and list of accumulated results"""
    tuples = []
    with tf.name_scope('seq_coder_r'):
        b = tf.shape(symbol_codes[0])[0]
        c = tf.tile(tf.expand_dims(model.empty_code(), axis=0), [b, 1])
        cs = list(symbol_codes)
        cs.reverse()
        for ci in cs:
            tuples.append(c)
            c = model.tuple(ci,c)
        return c, tuples


def seq_decoder_r(seq_code, seq_length):
    """Unflodr. Takes a code representing symbol sequence and unfolds it"""
    tuples = []
    symbols = []
    with tf.name_scope('seq_decoder_r'):
        for i in range(seq_length):
            symbol, seq_code = model.untuple(seq_code)
            tuples.append(seq_code)
            symbols.append(symbol)
    tuples.reverse()
    return symbols, tuples


# Learners
def learn_coder(p_diff_ids):
    """Subgraph for learning coder"""
    with tf.name_scope('coder_learner') as scope:
        diff_codes = tf.reshape(model.embed(one_hot=tf.reshape(p_diff_ids, [-1, model.sym_width])),
                                [-1, pair_sz, model.code_width])
        diff_pairs = tf.transpose(diff_codes, perm=[1, 0, 2])
        diff_cs = tf.unpack(diff_pairs)
        code_dist = tf.reduce_sum(tf.squared_difference(diff_cs[0], diff_cs[1]), 1)
        code_loss = tf.reduce_mean(1 / code_dist)
        tf.summary.scalar('code_loss', code_loss)
        code_min = tf.reduce_min(code_dist)
        tf.summary.scalar('code_min', code_min)
        return code_loss, code_min


def learn_tuple(seqs):
    """Subgraph for learning tuple/untuple modules"""
    with tf.name_scope('learn_tuple'):
        seq_list = tf.unpack(seqs)
        tup_codes = model.tuple(seq_list[0], seq_list[1])
        rev_seqs = model.untuple(tup_codes)
        tuple_sqr_dist = tf.squared_difference(seqs, rev_seqs)
        tuple_loss = tf.reduce_mean(tuple_sqr_dist)
        tf.summary.scalar('tuple_loss', tuple_loss)
        tuple_max = tf.sqrt(tf.reduce_max(tuple_sqr_dist))
        tf.summary.scalar('tuple_max', tuple_max)
        return tuple_loss, tuple_max, tup_codes, rev_seqs


def learn_fold(seqs, assoc):
    """Subgraph for learning folds consisting of repeating tuple applications (left or right associativity)"""
    with tf.name_scope('learn_fold') as scope:
        if assoc == 'Left':
            params = {'coder': seq_coder_l,
                      'decoder': seq_decoder_l,
                      'suffix': '_l'}
        elif assoc == 'Right':
            params = {'coder': seq_coder_r,
                      'decoder': seq_decoder_r,
                      'suffix': '_r'}
        else:
            raise Exception('unknown dir')

        seq_list = tf.unpack(seqs)
        code, tup_codes = params['coder'](seq_list)
        rev_seqs, rev_tup_codes = params['decoder'](code, FLAGS.seq_len)
        seq_sqr_dist = tf.squared_difference(seqs, rev_seqs)
        tup_sqr_dist = tf.squared_difference(tup_codes, rev_tup_codes)
        seq_loss = tf.reduce_mean(seq_sqr_dist)
        tup_loss = tf.reduce_mean(tup_sqr_dist)
        tf.summary.scalar('seq_loss' + params['suffix'], seq_loss)
        tf.summary.scalar('tup_loss' + params['suffix'], tup_loss)
        tup_max = tf.sqrt(tf.reduce_max(tup_sqr_dist))
        seq_max = tf.sqrt(tf.reduce_max(seq_sqr_dist))
        tf.summary.scalar('seq_max' + params['suffix'], seq_max)
        tf.summary.scalar('tup_max' + params['suffix'], tup_max)
        return code, seq_max, tup_max, seq_loss, tup_loss, rev_seqs


def learn_morphisms(code_l, code_r):
    """Subgraph for learning morphisms"""
    with tf.name_scope('learn_morphisms'):
        code_lr = model.left_to_right(code_l)
        code_rl = model.right_to_left(code_r)

        code_dist_lr_loss = tf.reduce_mean(tf.squared_difference(code_lr, code_r))
        code_dist_rl_loss = tf.reduce_mean(tf.squared_difference(code_rl, code_l))
        tf.summary.scalar('code_dist_lr', code_dist_lr_loss)
        tf.summary.scalar('code_dist_rl', code_dist_rl_loss)
        return code_dist_lr_loss, code_dist_rl_loss


def restoration_precision(seqs, rev_seqs, all_codes):
    """Subgraph for restoration stats for symbols"""
    with tf.name_scope('restoration_stats'):
        def codes_to_ids(codes):
            codes_1 = tf.expand_dims(codes, 2)
            dists = tf.reduce_sum(tf.squared_difference(codes_1, all_codes), 3)
            ids = tf.arg_min(dists, 2)
            return ids

        orig_ids = codes_to_ids(seqs)
        rev_ids = codes_to_ids(rev_seqs)

        restorations = tf.equal(orig_ids, rev_ids)
        num_restored = tf.reduce_sum(tf.cast(restorations, dtype=tf.float32), 0)
        # stats[i] is number of sequences with i-th element restored
        elem_restoration_stats = tf.reduce_sum(tf.cast(restorations, dtype=tf.int32), 1)
        # hist[i] - is number of sequences with i properly restored elements
        num_proper_restorations_hist = tf.histogram_fixed_width(num_restored,
                                                    [0.0, FLAGS.seq_len+1.0], FLAGS.seq_len+1, dtype=tf.int32)

        tf.summary.histogram('num_restored', num_restored)
        return elem_restoration_stats, num_proper_restorations_hist


def vis_tuple(seqs):
    """Subgraph for visualizing tuples and restored symbols"""
    with tf.name_scope('vis_tuple'):
        seq_list = tf.unpack(seqs)
        tup_codes = model.tuple(seq_list[0], seq_list[1])
        rev_seqs = model.untuple(tup_codes)
        return tup_codes, rev_seqs


def learn_type_detector(codes, tuples):
    """Subgraph for type detector learning"""
    code_labels = tf.tile(tf.constant([0], dtype=tf.int64), [tf.shape(codes)[0]])
    tuple_labels = tf.tile(tf.constant([1], dtype=tf.int64), [tf.shape(tuples)[0]])
    labels = tf.concat(0, [code_labels, tuple_labels])
    one_hot_labels = tf.one_hot(labels, 2)

    data = tf.concat(0, [codes, tuples])
    logits = model.type_detector(data)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels))
    prec = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), dtype=f32))
    return loss, prec


def do_train():
    print('vocabulary {}, code_width {}, sequence_len {}'.format(FLAGS.num_symbols, FLAGS.code_width, FLAGS.seq_len))

    ensure_dir('checkpoints')
    ensure_dir('logs')

    p_ids = tf.placeholder(f32, [None, FLAGS.seq_len, model.sym_width], name='ids')  # Batch x Seq x SymWidth
    ids_2d = tf.reshape(p_ids, [-1, model.sym_width])
    sym_codes = tf.reshape(model.embed(ids_2d), [-1, FLAGS.seq_len, model.code_width])  # Batch x Seq x Code
    seqs = tf.transpose(sym_codes, perm=[1, 0, 2])  # Seq x Batch x Code

    # for coder
    p_diff_ids = tf.placeholder(f32, [None, pair_sz, model.sym_width], name='diff_ids')
    code_loss, code_min = learn_coder(p_diff_ids)

    # Tuple/Untuple
    tuple_loss, tuple_max, tuple_codes, rev_seqs = learn_tuple(seqs)

    # Folds
    code_l, seq_max_l, tup_max_l, seq_loss_l, tup_loss_l, rev_seqs_l = learn_fold(seqs, 'Left')
    code_r, seq_max_r, tup_max_r, seq_loss_r, tup_loss_r, _ = learn_fold(seqs, 'Right')

    # Left-to right morphism
    code_dist_lr_loss, code_dist_rl_loss = learn_morphisms(code_l, code_r)

    # restoration accuracy
    p_all_ids = tf.placeholder(f32, [model.sym_width, model.num_syms], name='all_ids')
    all_codes = model.embed(p_all_ids)

    elem_restoration_stats, num_proper_restorations_hist = restoration_precision(seqs, rev_seqs, all_codes) # rev_seqs_l for folds

    # Visualization
    nc = tf.shape(all_codes)[0]
    all_code_stacks_1 = tf.reshape(tf.tile(all_codes, [nc, 1]), [nc, nc, -1])
    all_code_stacks_2 = tf.transpose(all_code_stacks_1, perm=[1, 0, 2])
    all_code_pairs_1 = tf.reshape(all_code_stacks_1, [nc * nc, -1])
    all_code_pairs_2 = tf.reshape(all_code_stacks_2, [nc * nc, -1])
    all_code_pairs = tf.stack([all_code_pairs_1, all_code_pairs_2])
    all_tuples, all_rev_sym = vis_tuple(all_code_pairs)

    # Type Detector
    type_det_loss, type_det_prec = learn_type_detector(all_codes, all_tuples)

    # loss for folds
    # full_loss = seq_loss_l + tup_loss_l + seq_loss_r + tup_loss_r + code_loss + code_dist_lr_loss + code_dist_rl_loss  #+ det_cross_ent
    # loss for tuple/untuple
    full_loss = tuple_loss + code_loss + type_det_loss
    step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(full_loss)

    experiment_date = experiment + "-" + strftime("%Y-%m-%d-%H%M%S", localtime())
    writer = tf.summary.FileWriter("logs/" + experiment_date, flush_secs=5)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)
        k = None
        all_perfect = 0

        all_ids_v = list(model.one_hot(model.symbols))

        for i in range(FLAGS.steps):
            k = i
            ids_v = sym_list_batch(FLAGS.seq_len, FLAGS.batch_size, False)
            diff_ids_v = sym_list_batch(pair_sz, model.num_syms, True)

  #          logs_ = ((seq_max_l, tup_max_l, code_min), (seq_max_r, tup_max_r),
  #           (seq_loss_l, tup_loss_l, code_loss, code_dist_lr_loss, code_dist_rl_loss))

            logs_ = ((tuple_loss, tuple_max), (code_min,), (type_det_loss, type_det_prec))
            vis_data_ = {"coder": model.Coder.parameters(), "tuple_codes": all_tuples, "rev_seqs": all_rev_sym}

            _, bin_summary, logs, restoration_stats, vis_data = \
                sess.run([step, summaries, logs_,
                          (num_proper_restorations_hist, elem_restoration_stats),
                          vis_data_],  # (det_cs1_acc, det_tups_acc, det_cross_ent)
                                         feed_dict={
                                             p_ids: ids_v,
                                             p_diff_ids: diff_ids_v,
                                             p_all_ids: all_ids_v
                                         })
            tuple_logs, coder_logs, type_det_logs = logs

            if restoration_stats[0][FLAGS.seq_len] == FLAGS.batch_size:
                all_perfect += 1
            else:
                all_perfect = 0

            if tuple_logs[0] < eps and coder_logs[0] > 0.5 and type_det_logs[1] > 0.99 : # or all_perfect >= 10000:
                print("early stopping")
                break

            if i % 100 == 0:
                writer.add_summary(bin_summary, i)
                print(i, list(restoration_stats[0]), list(restoration_stats[1]), logs)
                model.draw_matrices(vis_data)

            if i % 1000 == 0:
                model.net_saver.save(sess, "checkpoints/" + experiment_date, global_step=k)

            if i % 5000 == 0:
                model.print_matrices(sess)


do_train()
#elif sys.argv[3] == 'test':
#    checkpoint = sys.argv[4]
#    do_test(checkpoint)
