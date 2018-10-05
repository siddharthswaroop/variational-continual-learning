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


class MnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        # Define each task's MNIST digits
        task1 = [0, 1]
        task2 = [2, 3]
        task3 = [4, 5]
        task4 = [6, 7]
        task5 = [8, 9]

        self.sets = [task1, task2, task3, task4, task5]

        self.max_iter = len(self.sets)

        self.out_dim = 0 # Total number of unique classes
        self.class_list = [] # List of unique classes being considered, in the order they appear
        for task_id in range(self.max_iter):
            for class_index in range(len(self.sets[task_id])):
                if self.sets[task_id][class_index] not in self.class_list:
                    # Convert from MNIST digit numbers to class index number by using self.class_list.index(), which is done in self.classes
                    self.class_list.append(self.sets[task_id][class_index])
                    self.out_dim = self.out_dim + 1

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for task_id in range(self.max_iter):
            class_idx = []
            for i in range(len(self.sets[task_id])):
                class_idx.append(self.class_list.index(self.sets[task_id][i]))
            self.classes.append(class_idx)

        print 'MNIST task classes', self.classes

        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            next_x_train = []
            next_y_train = []
            next_x_test = []
            next_y_test = []

            # Loop over all classes in current iteration
            for class_index in range(np.size(self.sets[self.cur_iter])):

                # Find the correct set of training inputs
                train_id = np.where(self.train_label == self.sets[self.cur_iter][class_index])[0]
                # Stack the training inputs
                if class_index == 0:
                    next_x_train = self.X_train[train_id]
                else:
                    next_x_train = np.vstack((next_x_train, self.X_train[train_id]))

                # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                next_y_train_interm = np.zeros((len(train_id), self.out_dim))
                next_y_train_interm[:, self.classes[self.cur_iter][class_index]] = 1
                if class_index == 0:
                    next_y_train = next_y_train_interm
                else:
                    next_y_train = np.vstack((next_y_train, next_y_train_interm))

                # Repeat above process for test inputs
                test_id = np.where(self.test_label == self.sets[self.cur_iter][class_index])[0]
                if class_index == 0:
                    next_x_test = self.X_test[test_id]
                else:
                    next_x_test = np.vstack((next_x_test, self.X_test[test_id]))

                next_y_test_interm = np.zeros((len(test_id), self.out_dim))
                next_y_test_interm[:, self.classes[self.cur_iter][class_index]] = 1
                if class_index == 0:
                    next_y_test = next_y_test_interm
                else:
                    next_y_test = np.vstack((next_y_test, next_y_test_interm))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


vcl_avg = None
no_repeats = 1
for i in range(no_repeats):

    calculate_acc = False
    store_weights = False
    path = 'sandbox/singlehead/one_hidden_layer/300epochs/observed_classes/'

    hidden_size = [256]
    batch_size = 256
    no_epochs = 3
    no_iters = 1
    option = 1

    tf.reset_default_graph()
    random_seed = 10 + i
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    # Set up all the tasks
    # 1: multi-head
    # 2: like 1, but over all classes at test time
    # 3: like 2, but over classes observed so far at test time
    # 4: single-head
    # 5: single-head, but only for classes observed so far
    setting = 5


    if option == 1:
        coreset_size = 0
        data_gen = MnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    elif option == 2:
        coreset_size = 200
        data_gen = MnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)


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