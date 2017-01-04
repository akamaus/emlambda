import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model:
    """A model for learning symbol-hierarchy embedding.
    Symbols initially represented as one-hot rows (of size sym_width)
    which got embedded into Code vector-space (rows of size code_width)"""
    def __init__(self, num_symbols, code_width):
        self.code_width = code_width
        self.num_syms = num_symbols
        self.sym_width = num_symbols
        # symbol tables
        self.symbols = [i for i in range(1, num_symbols + 1)]
        self.sym_dict = {}

        for i, c in enumerate(self.symbols):
            self.sym_dict[c] = np.zeros(self.sym_width)
            self.sym_dict[c][i] = 1

        # embeds symbol
        self.Coder = Model.matrix([num_symbols, code_width], 'Coder')
        self.EmptyCode = Model.matrix([code_width], 'EmptyCode')
        tf.summary.histogram("Coder_weights", self.Coder)
        tf.summary.histogram("EmptyCode_weights", self.EmptyCode)
        # merges two embeddings to produce a tuple
        self.Tuple = Model.matrix([code_width * 2, code_width], 'Tuple')
        tf.summary.histogram("Tuple_weights", self.Tuple)
        # deconstruct tuple
        self.UnTuple = Model.matrix([code_width, code_width*2], 'UnTuple')
        tf.summary.histogram("UnTuple_weights", self.UnTuple)
        # detects if its a symbol or a Tuple
        self.TypeDetector = Model.matrix([code_width,2], 'TypeDetector')
        tf.summary.histogram("TypeDetector_weights", self.TypeDetector)
        # morphisms
        self.LR = Model.matrix([code_width, code_width], 'LR')
        self.RL = Model.matrix([code_width, code_width], 'RL')
        tf.summary.histogram("LR_weights", self.LR)
        tf.summary.histogram("RL_weights", self.RL)

        self.net_saver = tf.train.Saver([self.Coder, self.Tuple, self.UnTuple])

        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.first_draw = True

    def one_hot(self, symbols):
        """one-hot encoding of symbol or list of symbols"""
        if type(symbols) is int:
            res = self.sym_dict[symbols]
        elif type(symbols) is list:
            res = map(lambda s: self.sym_dict[s], symbols)
        else: raise Exception("Unknown type passed to model.one_hot" + str(type(symbols)))
        return res

    def embed(self, one_hot):
        """Embeds a symbol (or list of symbols) into vector-space"""
        return tf.matmul(one_hot, self.Coder)

    def empty_code(self):
        """Returns a code for empty symbol"""
        return self.EmptyCode

    def tuple(self, c1, c2):
        """Makes a tuple of two args"""
        return tf.matmul(tf.concat(1, [c1, c2]), self.Tuple)

    def untuple(self, c):
        """Splits tuple code into two subcomponents"""
        res = tf.matmul(c, self.UnTuple)
        return tf.split(1, 2, res)

    def left_to_right(self, c):
        return tf.matmul(c, self.LR)

    def right_to_left(self, c):
        return tf.matmul(c, self.RL)

    def draw_matrices(self, vis_data):
        xs = vis_data['coder'][:, 0]
        ys = vis_data['coder'][:, 1]

        rev_xs = vis_data['rev_seqs'][0][:, 0]
        rev_ys = vis_data['rev_seqs'][0][:, 1]

        tup_xs = vis_data['tuple_codes'][:, 0]
        tup_ys = vis_data['tuple_codes'][:, 1]

        s = 20
        if self.first_draw:
            self.seqs_plot, self.rev_plot, self.tuple_plot = self.ax.plot(xs, ys, '+', rev_xs, rev_ys, 'go', tup_xs, tup_ys, 'rp')
            self.ax.set_xlim(-s, s)
            self.ax.set_ylim(-s, s)
            self.first_draw = False
#            self.fig.show()
        else:
            self.seqs_plot.set_xdata(xs)
            self.seqs_plot.set_ydata(ys)
            self.rev_plot.set_xdata(rev_xs)
            self.rev_plot.set_ydata(rev_ys)
            self.tuple_plot.set_xdata(tup_xs)
            self.tuple_plot.set_ydata(tup_ys)

            #self.ax.relim()
            #self.ax.autoscale_view()
            self.fig.canvas.draw()

    def print_matrices(self, sess):
        print('Coder:')
        print(sess.run(self.Coder))

        print('Tuple:')
        print(sess.run(self.Tuple))

        print('UnTuple:')
        print(sess.run(self.UnTuple))

        print('EmptyCode')
        print(sess.run(self.EmptyCode))

        print('LR')
        print(sess.run(self.LR))

        print('RL')
        print(sess.run(self.RL))

    @staticmethod
    def matrix(shape, name = None):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
