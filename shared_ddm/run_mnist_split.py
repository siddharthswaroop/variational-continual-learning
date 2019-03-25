import numpy as np
import tensorflow as tf
import gzip
import cPickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
from utils import load_mnist
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


        # split MNIST
        task1 = [0, 1]
        task2 = [2, 3]
        task3 = [4, 5]
        task4 = [6, 7]
        task5 = [8, 9]
        self.sets = [task1, task2, task3, task4, task5]

        # Full MNIST
        self.sets = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

        # # Odd/even MNIST
        # task1 = [0, 2, 4, 6, 8]
        # task2 = [1, 3, 5, 7, 9]
        # self.sets = [task1, task2]

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

class FullMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]


        # Full MNIST
        self.sets = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
        #self.sets = [[0,1,2,3,4]]

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

                # print self.sets[self.cur_iter][class_index], len(next_x_train)

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class PermutedMnistGeneratorOld():
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


        self.out_dim = 10*self.max_iter # Total number of unique classes
        self.class_list = range(10*self.max_iter) # List of unique classes being considered, in the order they appear
        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0+10*iter,10+10*iter))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

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

            # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
            next_y_train = np.zeros((len(next_x_train), self.out_dim))
            next_y_train[:,0+10*self.cur_iter:10+10*self.cur_iter] = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]

            next_y_test = np.zeros((len(next_x_test), self.out_dim))
            next_y_test[:,0+10*self.cur_iter:10+10*self.cur_iter] = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class PermutedMnistGenerator():
    def __init__(self, max_iter=10, random_seed=0):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.cur_iter = 0


        self.out_dim = 10 # Total number of unique classes
        self.class_list = range(10) # List of unique classes being considered, in the order they appear
        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0,10))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter+self.random_seed)
            perm_inds = range(self.X_train.shape[1])
            if self.cur_iter > 0:
                np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]

            # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
            next_y_train = np.zeros((len(next_x_train), 10))
            next_y_train[:,0:10] = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]

            next_y_test = np.zeros((len(next_x_test), 10))
            next_y_test[:,0:10] = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class BatchPermutedMnistGenerator():
    def __init__(self, max_iter=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.Y_train = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.Y_test = test_set[1]
        self.max_iter = 1
        self.batch_number = max_iter
        self.cur_iter = 0


        self.out_dim = 10 # Total number of unique classes
        self.class_list = range(10) # List of unique classes being considered, in the order they appear
        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0,10))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:

            for cur_iter in range(self.batch_number):

                np.random.seed(cur_iter)
                perm_inds = range(self.X_train.shape[1])
                if cur_iter > 0:
                    np.random.shuffle(perm_inds)

                # Retrieve train data
                next_x_train_interm = deepcopy(self.X_train)
                next_x_train_interm = next_x_train_interm[:,perm_inds]

                # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                next_y_train_interm = np.zeros((len(next_x_train_interm), 10))
                next_y_train_interm[:,0:10] = np.eye(10)[self.Y_train]

                # Retrieve test data
                next_x_test_interm = deepcopy(self.X_test)
                next_x_test_interm = next_x_test_interm[:,perm_inds]

                next_y_test_interm = np.zeros((len(next_x_test_interm), 10))
                next_y_test_interm[:,0:10] = np.eye(10)[self.Y_test]

                if cur_iter == 0:
                    next_x_test = next_x_test_interm
                    next_y_test = next_y_test_interm
                    next_x_train = next_x_train_interm
                    next_y_train = next_y_train_interm
                else:
                    next_x_test = np.vstack((next_x_test, next_x_test_interm))
                    next_y_test = np.vstack((next_y_test, next_y_test_interm))
                    next_x_train = np.vstack((next_x_train, next_x_train_interm))
                    next_y_train = np.vstack((next_y_train, next_y_train_interm))


            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitOddEvenMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]


        task1 = [0,1]
        self.sets = [task1, task1, task1, task1, task1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

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

        # For VI Separate (not continual, not batch)
        #self.max_iter = 1

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_0_id = []
            train_1_id = []
            #self.cur_iter = 1 # For VI Separate (not continual, not batch)

            option = 3

            # 20% of task 0v1 training data at all times, 25% of task 2v3, 33% of task 4v5, etc.
            if option == 1:
                for task_id in range(self.cur_iter+1):
                    # Retrieve train data
                    train_0_id_full = np.where(self.train_label == self.sets_0[task_id])[0]
                    start_index = np.int(np.floor((1.0*self.cur_iter-task_id)/((self.max_iter-task_id))*train_0_id_full.size))
                    end_index = np.int(np.floor((1.0*self.cur_iter-task_id+1.0)/((self.max_iter-task_id))*train_0_id_full.size-1.0))
                    if task_id == 0:
                        train_0_id = train_0_id_full[start_index:end_index]
                    else:
                        train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                    train_1_id_full = np.where(self.train_label == self.sets_1[task_id])[0]
                    start_index = np.int(np.floor((1.0*self.cur_iter-task_id)/((self.max_iter-task_id))*train_1_id_full.size))
                    end_index = np.int(np.floor((1.0*self.cur_iter-task_id+1.0)/((self.max_iter-task_id))*train_1_id_full.size-1.0))
                    if task_id == 0:
                        train_1_id = train_1_id_full[start_index:end_index]
                    else:
                        train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])


                next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
                next_y_train = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # 50% of task 0v1 first, and then 10% from then on. Test data is just pairs of digits.
            if option == 2:
                train_0_id_full = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
                start_index = np.int(np.floor(0.5 * train_0_id_full.size))
                train_0_id = train_0_id_full[start_index:]

                train_1_id_full = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
                start_index = np.int(np.floor(0.5 * train_1_id_full.size))
                train_1_id = train_1_id_full[start_index:]

                for task_id in range(self.cur_iter):
                    # Retrieve train data
                    train_0_id_full = np.where(self.train_label == self.sets_0[task_id])[0]
                    start_index = np.int(np.floor((0.1 * self.cur_iter) * train_0_id_full.size))
                    end_index = np.int(np.floor((0.1 * (self.cur_iter + 1.0)) * train_0_id_full.size))
                    train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                    train_1_id_full = np.where(self.train_label == self.sets_1[task_id])[0]
                    start_index = np.int(np.floor((0.1 * self.cur_iter) * train_1_id_full.size))
                    end_index = np.int(np.floor((0.1 * (self.cur_iter + 1.0)) * train_1_id_full.size))
                    train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
                next_y_train = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                next_y_train = np.hstack((next_y_train, 1 - next_y_train))

            # 50% of task 0v1 first, and then 1% from then on. Test data is just pairs of digits.
            if option == 3:
                train_0_id_full = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
                start_index = np.int(np.floor(0.5 * train_0_id_full.size))
                train_0_id = train_0_id_full[start_index:]

                train_1_id_full = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
                start_index = np.int(np.floor(0.5 * train_1_id_full.size))
                train_1_id = train_1_id_full[start_index:]

                for task_id in range(self.cur_iter):
                    # Retrieve train data
                    train_0_id_full = np.where(self.train_label == self.sets_0[task_id])[0]
                    start_index = np.int(np.floor((0.01 * self.cur_iter) * train_0_id_full.size))
                    end_index = np.int(np.floor((0.01 * (self.cur_iter + 1.0)) * train_0_id_full.size))
                    train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                    train_1_id_full = np.where(self.train_label == self.sets_1[task_id])[0]
                    start_index = np.int(np.floor((0.01 * self.cur_iter) * train_1_id_full.size))
                    end_index = np.int(np.floor((0.01 * (self.cur_iter + 1.0)) * train_1_id_full.size))
                    train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
                next_y_train = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                next_y_train = np.hstack((next_y_train, 1 - next_y_train))



            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))
            next_y_test = np.vstack((np.ones((np.size(test_0_id), 1)), np.zeros((np.size(test_1_id), 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class BatchSplitOddEvenMnistGenerator():
    def __init__(self, task_max=5):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]

        self.task_max = task_max


        task1 = [0,1]
        self.sets = [task1]

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]

        self.max_iter = 1

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
            train_0_id = []
            train_1_id = []

            option = 3

            for cur_iter in range(self.task_max):

                # 50% of task 0v1 first, and then 10% from then on
                if option == 2:
                    train_0_id_full = np.where(self.train_label == self.sets_0[cur_iter])[0]
                    start_index = np.int(np.floor(0.5 * train_0_id_full.size))
                    train_0_id = train_0_id_full[start_index:]

                    train_1_id_full = np.where(self.train_label == self.sets_1[cur_iter])[0]
                    start_index = np.int(np.floor(0.5 * train_1_id_full.size))
                    train_1_id = train_1_id_full[start_index:]

                    for task_id in range(cur_iter):
                        # Retrieve train data
                        train_0_id_full = np.where(self.train_label == self.sets_0[task_id])[0]
                        start_index = np.int(np.floor((0.1 * cur_iter) * train_0_id_full.size))
                        end_index = np.int(np.floor((0.1 * (cur_iter + 1.0)) * train_0_id_full.size))
                        train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                        train_1_id_full = np.where(self.train_label == self.sets_1[task_id])[0]
                        start_index = np.int(np.floor((0.1 * cur_iter) * train_1_id_full.size))
                        end_index = np.int(np.floor((0.1 * (cur_iter + 1.0)) * train_1_id_full.size))
                        train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                    next_x_train_interm = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
                    next_y_train_interm = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                    next_y_train_interm = np.hstack((next_y_train_interm, 1 - next_y_train_interm))

                # 50% of task 0v1 first, and then 1% from then on. Test data is just pairs of digits.
                if option == 3:
                    train_0_id_full = np.where(self.train_label == self.sets_0[cur_iter])[0]
                    start_index = np.int(np.floor(0.5 * train_0_id_full.size))
                    train_0_id = train_0_id_full[start_index:]

                    train_1_id_full = np.where(self.train_label == self.sets_1[cur_iter])[0]
                    start_index = np.int(np.floor(0.5 * train_1_id_full.size))
                    train_1_id = train_1_id_full[start_index:]

                    for task_id in range(cur_iter):
                        # Retrieve train data
                        train_0_id_full = np.where(self.train_label == self.sets_0[task_id])[0]
                        start_index = np.int(np.floor((0.01 * cur_iter) * train_0_id_full.size))
                        end_index = np.int(np.floor((0.01 * (cur_iter + 1.0)) * train_0_id_full.size))
                        train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                        train_1_id_full = np.where(self.train_label == self.sets_1[task_id])[0]
                        start_index = np.int(np.floor((0.01 * cur_iter) * train_1_id_full.size))
                        end_index = np.int(np.floor((0.01 * (cur_iter + 1.0)) * train_1_id_full.size))
                        train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                    next_x_train_interm = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
                    next_y_train_interm = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                    next_y_train_interm = np.hstack((next_y_train_interm, 1 - next_y_train_interm))

                # Retrieve test data
                test_0_id = np.where(self.test_label == self.sets_0[cur_iter])[0]
                test_1_id = np.where(self.test_label == self.sets_1[cur_iter])[0]
                next_x_test_interm = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))
                next_y_test_interm = np.vstack((np.ones((np.size(test_0_id), 1)), np.zeros((np.size(test_1_id), 1))))
                next_y_test_interm = np.hstack((next_y_test_interm, 1-next_y_test_interm))

                if cur_iter == 0:
                    next_x_test = next_x_test_interm
                    next_y_test = next_y_test_interm
                    next_x_train = next_x_train_interm
                    next_y_train = next_y_train_interm
                else:
                    next_x_test = np.vstack((next_x_test, next_x_test_interm))
                    next_y_test = np.vstack((next_y_test, next_y_test_interm))
                    next_x_train = np.vstack((next_x_train, next_x_train_interm))
                    next_y_train = np.vstack((next_y_train, next_y_train_interm))




            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class FashionMnistGenerator():
    def __init__(self):
        #f = gzip.open('data/mnist.pkl.gz', 'rb')
        #train_set, valid_set, test_set = cPickle.load(f)
        #f.close()

        # self.X_train = np.vstack((train_set[0], valid_set[0]))
        # self.X_test = test_set[0]
        # self.train_label = np.hstack((train_set[1], valid_set[1]))
        # self.test_label = test_set[1]

        # From https://github.com/zalandoresearch/fashion-mnist

        self.X_train, self.train_label = load_mnist('data', kind='train')
        self.X_test, self.test_label = load_mnist('data', kind='t10k')


        # split MNIST
        task1 = [6,7]
        self.sets = [task1]

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

                print self.sets[self.cur_iter][class_index], len(next_x_train)

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class FashionDigitMnistGenerator():
    def __init__(self):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train_digit = np.vstack((train_set[0], valid_set[0]))
        self.X_test_digit = test_set[0]
        self.train_label_digit = np.hstack((train_set[1], valid_set[1]))
        self.test_label_digit = test_set[1]

        # From https://github.com/zalandoresearch/fashion-mnist
        self.X_train_fashion, self.train_label_fashion = load_mnist('data', kind='train')
        self.X_test_fashion, self.test_label_fashion = load_mnist('data', kind='t10k')


        # Each task corresponds to two classes from fashion MNIST and digit MNIST being compared
        task = [0,1]
        self.sets = [task, task, task, task, task, task, task, task, task, task]

        self.class_sets_per_task = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        self.class_sets_per_task_random_digit = [[7], [1], [3], [8], [9], [4], [2], [5], [6], [0]]
        self.class_sets_per_task_random_fashion = [[9], [4], [0], [2], [3], [8], [6], [1], [5], [7]]

        # # For VI separate
        # self.sets = [task]
        # self.class_sets_per_task = [[7]]

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
        return self.X_train_digit.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_0_id = []
            train_1_id = []
            #self.cur_iter = 1 # For VI Separate (not continual, not batch)

            option = 1

            # number_main_examples of task [0,1 vs fashion0,1] first, and then number_carried_over_examples from then on. Test data is just pairs of digits.
            # Only ever see 0s and 1s together, just like only see fashion MNIST class 0 and class 1 together
            if option == 1:

                number_main_examples = 3000
                number_main_examples = -1
                number_carried_over_examples = 0 # 100

                for class_id in range(len(self.class_sets_per_task[self.cur_iter])):
                    # Digit MNIST
                    train_0_id_full = np.where(self.train_label_digit == self.class_sets_per_task_random_digit[self.cur_iter][class_id])[0]
                    train_0_id = np.append(train_0_id, train_0_id_full[:number_main_examples])

                    # Fashion MNIST
                    train_1_id_full = np.where(self.train_label_fashion == self.class_sets_per_task_random_fashion[self.cur_iter][class_id])[0]
                    train_1_id = np.append(train_1_id, train_1_id_full[:number_main_examples])

                for task_id in range(self.cur_iter):
                    for class_id in range(len(self.class_sets_per_task[task_id])):
                        train_0_id_full = np.where(self.train_label_digit == self.class_sets_per_task_random_digit[task_id][class_id])[0]
                        start_index = number_main_examples + self.cur_iter * number_carried_over_examples
                        end_index = number_main_examples + (self.cur_iter + 1) * number_carried_over_examples
                        train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                        train_1_id_full = np.where(self.train_label_fashion == self.class_sets_per_task_random_fashion[task_id][class_id])[0]
                        train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                next_x_train = np.vstack((self.X_train_digit[train_0_id.astype(int)], self.X_train_fashion[train_1_id.astype(int)]))
                next_y_train = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                next_y_train = np.hstack((next_y_train, 1 - next_y_train))


            test_0_id = []
            test_1_id = []
            # Retrieve test data
            for class_id in range(len(self.class_sets_per_task[self.cur_iter])):
                test_0_id_interm = np.where(self.test_label_digit == self.class_sets_per_task_random_digit[self.cur_iter][class_id])[0]
                test_1_id_interm = np.where(self.test_label_fashion == self.class_sets_per_task_random_fashion[self.cur_iter][class_id])[0]
                test_0_id = np.append(test_0_id, test_0_id_interm)
                test_1_id = np.append(test_1_id, test_1_id_interm)

            next_x_test = np.vstack((self.X_test_digit[test_0_id.astype(int)], self.X_test_fashion[test_1_id.astype(int)]))
            next_y_test = np.vstack((np.ones((np.size(test_0_id), 1)), np.zeros((np.size(test_1_id), 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class BatchFashionDigitMnistGenerator():
    def __init__(self, task_max=10):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train_digit = np.vstack((train_set[0], valid_set[0]))
        self.X_test_digit = test_set[0]
        self.train_label_digit = np.hstack((train_set[1], valid_set[1]))
        self.test_label_digit = test_set[1]

        # From https://github.com/zalandoresearch/fashion-mnist
        self.X_train_fashion, self.train_label_fashion = load_mnist('data', kind='train')
        self.X_test_fashion, self.test_label_fashion = load_mnist('data', kind='t10k')


        # Each task corresponds to two classes from fashion MNIST and digit MNIST being compared
        task = [0,1]
        self.sets = [task]
        self.task_max = 10

        self.class_sets_per_task = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        self.class_sets_per_task_random_digit = [[7], [1], [3], [8], [9], [4], [2], [5], [6], [0]]
        self.class_sets_per_task_random_fashion = [[9], [4], [0], [2], [3], [8], [6], [1], [5], [7]]

        # # For VI separate
        # self.sets = [task]
        # self.class_sets_per_task = [[7]]

        self.max_iter = len(self.sets)
        self.max_iter = 1

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
        return self.X_train_digit.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            train_0_id = []
            train_1_id = []
            #self.cur_iter = 1 # For VI Separate (not continual, not batch)

            option = 1

            for cur_iter in range(self.task_max):

                # number_main_examples of task [0,1 vs fashion0,1] first, and then number_carried_over_examples from then on. Test data is just pairs of digits.
                # Only ever see 0s and 1s together, just like only see fashion MNIST class 0 and class 1 together
                if option == 1:

                    number_main_examples = 3000
                    number_main_examples = -1
                    number_carried_over_examples = 0 # 100

                    for class_id in range(len(self.class_sets_per_task[cur_iter])):
                        # Digit MNIST
                        train_0_id_full = np.where(self.train_label_digit == self.class_sets_per_task_random_digit[cur_iter][class_id])[0]
                        train_0_id = np.append(train_0_id, train_0_id_full[:number_main_examples])

                        # Fashion MNIST
                        train_1_id_full = np.where(self.train_label_fashion == self.class_sets_per_task_random_fashion[cur_iter][class_id])[0]
                        train_1_id = np.append(train_1_id, train_1_id_full[:number_main_examples])

                    for task_id in range(cur_iter):
                        for class_id in range(len(self.class_sets_per_task[task_id])):
                            train_0_id_full = np.where(self.train_label_digit == self.class_sets_per_task_random_digit[task_id][class_id])[0]
                            start_index = number_main_examples + cur_iter * number_carried_over_examples
                            end_index = number_main_examples + (cur_iter + 1) * number_carried_over_examples
                            train_0_id = np.append(train_0_id, train_0_id_full[start_index:end_index])

                            train_1_id_full = np.where(self.train_label_fashion == self.class_sets_per_task_random_fashion[task_id][class_id])[0]
                            train_1_id = np.append(train_1_id, train_1_id_full[start_index:end_index])

                    next_x_train = np.vstack((self.X_train_digit[train_0_id.astype(int)], self.X_train_fashion[train_1_id.astype(int)]))
                    next_y_train = np.vstack((np.ones((np.size(train_0_id), 1)), np.zeros((np.size(train_1_id), 1))))
                    next_y_train = np.hstack((next_y_train, 1 - next_y_train))


                test_0_id = []
                test_1_id = []
                # Retrieve test data
                for class_id in range(len(self.class_sets_per_task[cur_iter])):
                    test_0_id_interm = np.where(self.test_label_digit == self.class_sets_per_task_random_digit[cur_iter][class_id])[0]
                    test_1_id_interm = np.where(self.test_label_fashion == self.class_sets_per_task_random_fashion[cur_iter][class_id])[0]
                    test_0_id = np.append(test_0_id, test_0_id_interm)
                    test_1_id = np.append(test_1_id, test_1_id_interm)

                next_x_test = np.vstack((self.X_test_digit[test_0_id.astype(int)], self.X_test_fashion[test_1_id.astype(int)]))
                next_y_test = np.vstack((np.ones((np.size(test_0_id), 1)), np.zeros((np.size(test_1_id), 1))))
                next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitPermutedMnistGenerator():
    def __init__(self, max_iter=10, random_seed=1):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.X_test = test_set[0]
        self.test_label = test_set[1]
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.cur_iter = 0


        self.out_dim = 10 # Total number of unique classes
        self.class_list = range(10) # List of unique classes being considered, in the order they appear
        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(range(0,10))

        self.sets = self.classes

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.cur_iter + self.random_seed)
            perm_inds = range(self.X_train.shape[1])
            if self.cur_iter > 0:
                np.random.shuffle(perm_inds)

            X_train_interm = deepcopy(self.X_train)
            X_test_interm = deepcopy(self.X_test)

            X_train_interm = X_train_interm[:,perm_inds]
            X_test_interm = X_test_interm[:,perm_inds]

            next_x_train = []
            next_y_train = []
            next_x_test = []
            next_y_test = []

            option = 1

            for digit in range(10):


                # Random fraction_of_images of each digit per training task
                fraction_of_images = 0.5
                if option == 1:
                    # Find the correct set of training inputs
                    train_id_full = np.where(self.train_label == digit)[0]
                    #rand_order = range(train_id_full.size)
                    np.random.shuffle(train_id_full)
                    #train_id_full = train_id_full[rand_order]
                    end_index = np.int(np.floor(fraction_of_images * train_id_full.size))
                    train_id = train_id_full[:end_index]

                    # Stack the training inputs
                    if digit == 0:
                        next_x_train = X_train_interm[train_id]
                    else:
                        next_x_train = np.vstack((next_x_train, X_train_interm[train_id]))

                    # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                    next_y_train_interm = np.zeros((len(train_id), self.out_dim))
                    next_y_train_interm[:, digit] = 1
                    if digit == 0:
                        next_y_train = next_y_train_interm
                    else:
                        next_y_train = np.vstack((next_y_train, next_y_train_interm))

                    # Repeat above process for test inputs
                    test_id = np.where(self.test_label == digit)[0]
                    if digit == 0:
                        next_x_test = X_test_interm[test_id]
                    else:
                        next_x_test = np.vstack((next_x_test, X_test_interm[test_id]))

                    next_y_test_interm = np.zeros((len(test_id), self.out_dim))
                    next_y_test_interm[:, digit] = 1
                    if digit == 0:
                        next_y_test = next_y_test_interm
                    else:
                        next_y_test = np.vstack((next_y_test, next_y_test_interm))

                # Different 10% of MNIST images per task
                if option == 2:
                    # Find the correct set of training inputs
                    train_id_full = np.where(self.train_label == digit)[0]
                    start_index = np.int(np.floor(0.1 * self.cur_iter * train_id_full.size))
                    end_index = np.int(np.floor(0.1 * (self.cur_iter + 1) * train_id_full.size))
                    train_id = train_id_full[start_index:end_index]

                    # Stack the training inputs
                    if digit == 0:
                        next_x_train = X_train_interm[train_id]
                    else:
                        next_x_train = np.vstack((next_x_train, X_train_interm[train_id]))

                    # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                    next_y_train_interm = np.zeros((len(train_id), self.out_dim))
                    next_y_train_interm[:, digit] = 1
                    if digit == 0:
                        next_y_train = next_y_train_interm
                    else:
                        next_y_train = np.vstack((next_y_train, next_y_train_interm))

                    # Repeat above process for test inputs
                    test_id = np.where(self.test_label == digit)[0]
                    if digit == 0:
                        next_x_test = X_test_interm[test_id]
                    else:
                        next_x_test = np.vstack((next_x_test, X_test_interm[test_id]))

                    next_y_test_interm = np.zeros((len(test_id), self.out_dim))
                    next_y_test_interm[:, digit] = 1
                    if digit == 0:
                        next_y_test = next_y_test_interm
                    else:
                        next_y_test = np.vstack((next_y_test, next_y_test_interm))

            # Add some examples from previous images
            # Need to add perm_inds information!!
            if False:
                # Coreset equivalent
                for digit in range(self.cur_iter):
                    # Find the correct set of training inputs
                    train_id_full = np.where(self.train_label == digit)[0]
                    start_index = np.int(np.floor(0.1 * self.cur_iter * train_id_full.size))
                    end_index = np.int(np.floor(((0.1 * self.cur_iter)+0.01) * train_id_full.size))
                    train_id = train_id_full[start_index:end_index]

                    # Stack the training inputs
                    next_x_train = np.vstack((next_x_train, X_train_interm[train_id]))

                    # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
                    next_y_train_interm = np.zeros((len(train_id), self.out_dim))
                    next_y_train_interm[:, digit] = 1
                    next_y_train = np.vstack((next_y_train, next_y_train_interm))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test


vcl_avg = None
no_repeats = 1
for i in range(no_repeats):


    calculate_acc = False
    store_weights = True

    # if i == 0:
    #     path = 'sandbox/full_MNIST/two_hidden_layers/run1/'
    # if i == 1:
    #     path = 'sandbox/full_MNIST/two_hidden_layers/run2/'
    # if i == 2:
    #     path = 'sandbox/full_MNIST/two_hidden_layers/run3/'
    # if i == 3:
    #     path = 'sandbox/full_MNIST/two_hidden_layers/run4/'
    # if i == 4:
    #     path = 'sandbox/full_MNIST/two_hidden_layers/run5/'

    #path = 'sandbox/full_MNIST/two_hidden_layers/run1/'
    if i == 0:
        path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run6/'
    if i == 1:
        path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run2/'
    if i == 2:
        path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run3/'
    if i == 3:
        path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run4/'
    if i == 4:
        path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run5/'

    path = 'sandbox/permuted_MNIST/test/'
    hidden_size = [100, 100]
    batch_size = 1024
    no_epochs = 100
    no_iters = 1#50
    option = 4
    # batch_split_oddeven_task_max = 5
    permuted_max_iter = 1

    tf.reset_default_graph()
    random_seed = 10 + i + 5
    tf.set_random_seed(random_seed+1)
    np.random.seed(random_seed)

    # # Need to run permuted MNIST two more times
    # calculate_acc = False
    # store_weights = True
    # #path = 'sandbox/splitMNIST/multihead/one_hidden_layer_600_epochs/200neurons/run6/'
    #
    # hidden_size = [100, 100]
    # batch_size = 1024
    # no_epochs = 800
    # no_iters = 1
    # option = 4
    # # batch_split_oddeven_task_max = 5
    # permuted_max_iter = 10
    #
    # tf.reset_default_graph()
    # random_seed = 15 + i
    # tf.set_random_seed(random_seed)
    # np.random.seed(random_seed)

    # elif (i == 3):
    #     path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run9/'
    #
    # elif (i == 4):
    #     path = 'sandbox/permuted_MNIST/two_hidden_layers/800epochs/run10/'




    # Set up all the tasks
    # 1: multi-head
    # 2: like 1, but over all classes at test time
    # 3: like 2, but over classes observed so far at test time
    # 4: single-head
    # 5: single-head, but only for classes observed so far
    # 6: single-head [observed classes] during training, but multi-head during testing
    # 7: like 4, but for split odd/even MNIST (more epochs for task 1, fewer epochs for task 5)
    setting = 4

    # Split MNIST
    if option == 1:
        coreset_size = 0
        data_gen = MnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Coreset
    elif option == 2:
        coreset_size = 200
        data_gen = MnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Full MNIST
    if option == 3:
        coreset_size = 0
        data_gen = FullMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Permuted MNIST
    if option == 4:
        setting = 4
        coreset_size = 0
        data_gen = PermutedMnistGenerator(max_iter=permuted_max_iter, random_seed=random_seed)
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Split odd/even MNIST
    if option == 5:
        coreset_size = 0
        data_gen = SplitOddEvenMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Batch split odd/even MNIST
    if option == 6:
        coreset_size = 0
        data_gen = BatchSplitOddEvenMnistGenerator(task_max=batch_split_oddeven_task_max)
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Fashion MNIST
    if option == 7:
        coreset_size = 0
        data_gen = FashionMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # split Fashion/Digit MNIST
    if option == 8:
        coreset_size = 0
        data_gen = FashionDigitMnistGenerator()
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Batch split Fasion/Digit MNIST
    if option == 9:
        coreset_size = 0
        data_gen = BatchFashionDigitMnistGenerator(task_max=10)
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Split Permuted MNIST
    if option == 10:
        coreset_size = 0
        data_gen = SplitPermutedMnistGenerator(max_iter=permuted_max_iter, random_seed=random_seed)
        vcl_result, _ = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
            coreset.rand_from_batch, setting, coreset_size, batch_size, path, calculate_acc, no_iters=no_iters, store_weights=store_weights)

    # Batch Permuted MNIST
    if option == 11:
        coreset_size = 0
        data_gen = BatchPermutedMnistGenerator(max_iter=permuted_max_iter)
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