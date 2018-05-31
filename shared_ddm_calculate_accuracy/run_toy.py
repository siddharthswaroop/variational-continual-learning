import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy
import pickle


def softmax(x):
    x_t = np.transpose(x)
    e_x = np.exp(x_t - np.amax(x_t, axis=0))
    return np.transpose(e_x / e_x.sum(axis=0))


def sample(x):
    idx = []
    for i in range(x.shape[0]):
        idx.append( np.random.choice(2, 1, p=x[i])[0] )
    return np.eye(2)[idx]


class ToyGenerator():
    def __init__(self, n_samples, in_dims, hidden_size):
        np.random.seed(1)

        self.X_train, self.Y_train, self.X_test, self.Y_test = [], [], [], []

        # First task
        W1 = np.random.randn(in_dims, hidden_size)
        b1 = np.random.randn(hidden_size)

        self.X_train.append( np.random.uniform(size=(n_samples, in_dims)) )
        h1_0 = np.dot(self.X_train[0], W1) + b1
        np.maximum(h1_0, 0, h1_0) # ReLU

        W2_0 = np.random.randn(hidden_size, 2)
        b2_0 = np.random.randn(2)
        out_0 = softmax(np.dot(h1_0, W2_0) + b2_0)
        self.Y_train.append( sample(out_0) )

        self.X_test.append( np.random.uniform(size=(1000, in_dims)) )
        h1_0 = np.dot(self.X_test[0], W1) + b1
        np.maximum(h1_0, 0, h1_0) # ReLU
        out_0 = softmax(np.dot(h1_0, W2_0) + b2_0)
        self.Y_test.append( sample(out_0) )


        # Second task
        self.X_train.append( np.random.uniform(size=(n_samples, in_dims)) )
        h1_1 = np.dot(self.X_train[1], W1) + b1
        np.maximum(h1_1, 0, h1_1) # ReLU

        W2_1 = np.random.randn(hidden_size, 2)
        b2_1 = np.random.randn(2)
        out_1 = softmax(np.dot(h1_1, W2_1) + b2_1)
        self.Y_train.append( sample(out_1) )

        self.X_test.append( np.random.uniform(size=(1000, in_dims)) )
        h1_1 = np.dot(self.X_test[1], W1) + b1
        np.maximum(h1_1, 0, h1_1) # ReLU
        out_1 = softmax(np.dot(h1_1, W2_1) + b2_1)
        self.Y_test.append( sample(out_1) )

        self.max_iter = 2
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train[0].shape[1], 2

    def next_task(self):
        next_x_train = self.X_train[self.cur_iter]
        next_y_train = self.Y_train[self.cur_iter]
        next_x_test = self.X_test[self.cur_iter]
        next_y_test = self.Y_test[self.cur_iter]
        self.cur_iter += 1
        return next_x_train, next_y_train, next_x_test, next_y_test


class SplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        # self.sets_0 = [0, 2, 4, 6, 8]
        # self.sets_1 = [1, 3, 5, 7, 9]
        self.sets_0 = [2, 8]
        self.sets_1 = [3, 9]
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


hidden_size_toy = 100
hidden_size = [100]
batch_size = 256
no_epochs = 100
no_iters = 1
coreset_size = 40

# Run vanilla VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

option = 1

#if len(sys.argv) == 2:
#    option = int(sys.argv[1])
#else:
#    option = 4

if option == 1:
    coreset_size = 0
    #data_gen = SplitMnistGenerator()
    data_gen = ToyGenerator(10000, 784, hidden_size_toy)
    vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, no_iters=no_iters)
    print vcl_result
    #pickle.dump(vcl_result, open('results/vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

elif option == 2:
    # Run random coreset VCL
    data_gen = SplitMnistGenerator()
    rand_vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, no_iters=no_iters)
    print rand_vcl_result
    #pickle.dump(rand_vcl_result, open('results/rand_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)


elif option == 3:
    # Run k-center coreset VCL
    data_gen = SplitMnistGenerator()
    kcen_vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
        coreset.k_center, coreset_size, batch_size, no_iters=no_iters)
    print kcen_vcl_result
    #pickle.dump(kcen_vcl_result, open('results/kcen_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)


# # Plot average accuracy
# vcl_avg = np.nanmean(vcl_result, 1)
# rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
# kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
# utils.plot('results/split.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)