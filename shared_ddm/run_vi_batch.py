import numpy as np
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg_batch/'])
import vcl
import coreset
import utils
from copy import deepcopy
import pickle

class SplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        #self.sets_0 = [0, 2, 4, 6, 8]
        #self.sets_1 = [1, 3, 5, 7, 9]
        self.sets_0 = [0, 2, 8]
        self.sets_1 = [1, 3, 9]

        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


hidden_size = [256]
batch_size = 256
no_epochs = 20

# 10  epochs - [0.9990543735224586, 0.9794319294809011, 0.9798285426122038]
# 20  epochs - [0.9995271867612293, 0.9779627815866797, 0.9798285426122038]
# 80  epochs - [0.9995271867612293, 0.975024485798237,  0.9803328290468987]
# 300 epochs - [0.9995271867612293, 0.9760039177277179, 0.983358547655068]
# 300 epochs - swap order of tasks to 8 0 2 - [0.9813414019162885, 0.9995271867612293, 0.9769833496571988]

# Run vanilla VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)


coreset_size = 0
data_gen = SplitMnistGenerator()
vcl_result = vcl.run_vcl_shared_vi(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size)