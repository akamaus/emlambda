import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, num_syms, code_width):
        self.code_width = code_width
        self.num_syms = num_syms
        self.sym_width = num_syms
        # symbol tables
        self.symbols = [i for i in range(1, num_syms + 1)]
        self.sym_dict = {}

        for i, c in enumerate(self.symbols):
            self.sym_dict[c] = np.zeros(self.sym_width)
            self.sym_dict[c][i] = 1

        # embeds symbol
        self.Coder = Model.matrix([code_width, num_syms])
        # merges two embeddings to produce a tuple
        self.Tuple = Model.matrix([code_width, code_width * 2])
        # deconstruct tuple
        self.UnTuple1 = Model.matrix([code_width, code_width])
        self.UnTuple2 = Model.matrix([code_width, code_width])
        # detects if its a symbol or a Tuple
        self.TypeDetector = Model.matrix([2, code_width])

        tf.summary.histogram("Coder_weights", self.Coder)
        tf.summary.histogram("Tuple_weights", self.Tuple)
        tf.summary.histogram("UnTuple1_weights", self.UnTuple1)
        tf.summary.histogram("UnTuple2_weights", self.UnTuple2)
        tf.summary.histogram("TypeDetector_weights", self.TypeDetector)

        self.net_saver = tf.train.Saver([self.Coder, self.Tuple, self.UnTuple1, self.UnTuple2])

    @staticmethod
    def matrix(shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
