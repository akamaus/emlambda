
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

