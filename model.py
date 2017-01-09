import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Linear:
    def __init__(self, inp_size, out_size, name = None):
        self.weights = tf.Variable(tf.truncated_normal([inp_size, out_size], stddev=1), name=name + "_weights")
        tf.summary.histogram(name + '_weights', self.weights)

    def apply(self, inp):
        return tf.matmul(inp, self.weights)

    def parameters(self):
        return {'weights': self.weights}

class Affine:
        def __init__(self, inp_size, out_size, name=None):
            self.weights = tf.Variable(tf.truncated_normal([inp_size, out_size], stddev=1), name=name + "_weights")
            self.biases = tf.Variable(tf.truncated_normal([out_size], stddev=1), name=name + "_biases")
            tf.summary.histogram(name + '_weights', self.weights)
            tf.summary.histogram(name + '_biases', self.biases)

        def apply(self, inp):
            return tf.matmul(inp, self.weights) + self.biases

        def parameters(self):
            return {'weights': self.weights, 'biases': self.biases}


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

        # Null Symbol
        self.EmptyCode = Model.matrix([code_width], 'EmptyCode')
        tf.summary.histogram("EmptyCode_weights", self.EmptyCode)

        # embeds symbol
        self.Coder = Linear(num_symbols, code_width, 'Coder')
        # merges two embeddings to produce a tuple
        self.Tuple = Affine(code_width * 2, code_width, 'Tuple')
        # deconstruct tuple
        self.UnTuple = Affine(code_width, code_width*2, 'UnTuple')
        # detects if its a symbol or a Tuple
        self.TypeDetector = Affine(code_width, 2, 'TypeDetector')
        # morphisms
        self.LR = Linear(code_width, code_width, 'LR')
        self.RL = Linear(code_width, code_width, 'RL')

        all_params = []
        for m in [self.Coder, self.Tuple, self.UnTuple, self.TypeDetector, self.LR, self.RL]:
            all_params += m.parameters().items()
        self.net_saver = tf.train.Saver(dict(all_params))

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
        return self.Coder.apply(one_hot)

    def type_detector(self, code):
        """Returns logits corresponding to code belong to different classes"""
        return self.TypeDetector.apply(code)

    def empty_code(self):
        """Returns a code for empty symbol"""
        return self.EmptyCode

    def tuple(self, c1, c2):
        """Makes a tuple of two args"""
        return self.Tuple.apply(tf.concat(1, [c1, c2]))

    def untuple(self, c):
        """Splits tuple code into two subcomponents"""
        res = self.UnTuple.apply(c)
        return tf.split(1, 2, res)

    def left_to_right(self, c):
        return self.LR.apply(c)

    def right_to_left(self, c):
        return self.RL.apply(c)

    def draw_matrices(self, vis_data):
        xs = vis_data['coder']['weights'][:, 0]
        ys = vis_data['coder']['weights'][:, 1]

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

            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()

    def print_matrices(self, sess):
        print('EmptyCode')
        print(sess.run(self.EmptyCode))

        print('Coder:')
        print(sess.run(self.Coder.parameters()))

        print('Tuple:')
        print(sess.run(self.Tuple.parameters()))

        print('UnTuple:')
        print(sess.run(self.UnTuple.parameters()))

        print('LR')
        print(sess.run(self.LR.parameters()))

        print('RL')
        print(sess.run(self.RL.parameters()))

    @staticmethod
    def matrix(shape, name = None):
        return tf.Variable(tf.truncated_normal(shape, stddev=1), name=name)

    def affine(inp, out, name = None):

        return tf.Variable()
