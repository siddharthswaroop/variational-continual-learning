import numpy as np
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

class SplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        #self.sets_0 = [0, 2, 8]
        #self.sets_1 = [1, 3, 9]
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

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            np.random.seed(self.cur_iter)
            perm_inds = range(self.X_train.shape[1])
            if self.cur_iter > 0:
                np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SingleHeadSplitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]


        task1 = [0, 1]
        task2 = [2, 3]
        task3 = [4, 5]
        task4 = [6, 7]
        task5 = [8, 9]

        #sets_0 = [0, 2, 4, 6, 8]
        #sets_1 = [1, 3, 5, 7, 9]
        #self.sets_0 = [0, 2, 8]
        #self.sets_1 = [1, 3, 9]

        self.sets = [task1, task2, task3, task4, task5]

        # Currently assuming classes are not repeated in self.sets for out_dim
        self.out_dim = np.size(self.sets)
        self.max_iter = len(self.sets)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets[self.cur_iter][0])[0]
            train_1_id = np.where(self.train_label == self.sets[self.cur_iter][1])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            # Only works for 2 classes per task
            next_y_train_interm = np.hstack((np.ones((1, train_0_id.shape[0])), np.zeros((1, train_1_id.shape[0]))))
            next_y_train = np.zeros((np.size(next_y_train_interm,1), self.out_dim))
            next_y_train[:, self.sets[self.cur_iter][0]] = next_y_train_interm
            next_y_train[:, self.sets[self.cur_iter][1]] = 1-next_y_train_interm

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets[self.cur_iter][0])[0]
            test_1_id = np.where(self.test_label == self.sets[self.cur_iter][1])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test_interm = np.hstack((np.ones((1, test_0_id.shape[0])), np.zeros((1, test_1_id.shape[0]))))
            next_y_test = np.zeros((np.size(next_y_test_interm,1), self.out_dim))
            next_y_test[:, self.sets[self.cur_iter][0]] = next_y_test_interm
            next_y_test[:, self.sets[self.cur_iter][1]] = 1-next_y_test_interm

            ## Retrieve test data
            #next_x_test = deepcopy(self.X_test)
            #next_y_test = np.eye(10)[self.test_label]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


vcl_avg = None
no_repeats = 1
for i in range(no_repeats):

    # Set to true to load weights from 'path' and calculate accuracy (no training)
    calculate_acc = True
    path = 'sandbox/singlehead/coreset_200/'

    hidden_size = [256]
    batch_size = 256
    no_epochs = 300
    no_iters = 1

    ml_init = False

    # Run vanilla VCL
    tf.reset_default_graph()
    tf.set_random_seed(10+i)
    np.random.seed(10+i)

    option = 2

    #if len(sys.argv) == 2:
    #    option = int(sys.argv[1])
    #else:
    #    option = 4

    if option == 1:
        single_head = False
        coreset_size = 0
        data_gen = SplitMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, coreset_size, batch_size, ml_init, path, calculate_acc, no_iters=no_iters, single_head=single_head)
        # print vcl_result
        #pickle.dump(vcl_result, open('results/vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

    elif option == 2:
        single_head = True
        coreset_size = 200
        data_gen = SingleHeadSplitMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, coreset_size, batch_size, ml_init, path, calculate_acc, no_iters=no_iters, single_head=single_head)
        # print vcl_result
        #pickle.dump(vcl_result, open('results/vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)

    elif option == 3:
        hidden_size = [16, 256]
        path = 'sandbox/overpruning_tests/small_big_layers/'
        #hidden_size = [256, 256]
        #path = 'sandbox/overpruning_tests/full/'
        calculate_acc = True
        batch_size = 256
        no_epochs = 2000
        single_head = False
        coreset_size = 0
        data_gen = PermutedMnistGenerator(max_iter=1)
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, coreset_size, batch_size, ml_init, path, calculate_acc, no_iters=no_iters, single_head=single_head)

    """
    elif option == 2:
        # Run random coreset VCL
        coreset_size = 40
        data_gen = SplitMnistGenerator()
        vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, coreset_size, batch_size, ml_init, no_iters=no_iters)
        #print rand_vcl_result
        #pickle.dump(rand_vcl_result, open('results/rand_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)


    elif option == 3:
        # Run k-center coreset VCL
        coreset_size = 40
        data_gen = SplitMnistGenerator()
        kcen_vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.k_center, coreset_size, batch_size, ml_init, no_iters=no_iters)
        print kcen_vcl_result
        #pickle.dump(kcen_vcl_result, open('results/kcen_vcl_split_result_%d.pkl'%no_iters, 'wb'), pickle.HIGHEST_PROTOCOL)
    """

    if vcl_avg is None:
        vcl_avg = vcl_result
    else:
        vcl_avg = vcl_avg+vcl_result

vcl_result = vcl_avg/(1.0*no_repeats)

# Print in a suitable format
if data_gen.max_iter > 1:
    for task_id in range(data_gen.max_iter):
        for i in range(task_id + 1):
            print vcl_result[task_id][i],
        print ''

#np.savez('sandbox/smallinitalways/accuracy.npz', acc=acc_result, ind=[x+1 for x in epoch_pause])


# # Plot average accuracy
# vcl_avg = np.nanmean(vcl_result, 1)
# rand_vcl_avg = np.nanmean(rand_vcl_result, 1)
# kcen_vcl_avg = np.nanmean(kcen_vcl_result, 1)
# utils.plot('results/split.jpg', vcl_avg, rand_vcl_avg, kcen_vcl_avg)