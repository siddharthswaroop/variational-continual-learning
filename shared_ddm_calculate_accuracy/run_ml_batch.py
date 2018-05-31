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

class PermutedMnistGenerator():
    def __init__(self, max_iter=3):
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

hidden_size = [100, 100]
batch_size = None
no_epochs = 1000
Adam_learning_rate = 0.001

option = 2

## splitMNIST

# 0v1 2v3 8v9, batch_size = None, hidden_size = [256], Adam learning rate = 0.0005
# 20  epochs - [0.9971631205673759, 0.9427032321253673, 0.9389813414019162]
# 200 epochs - [0.9990543735224586, 0.9769833496571988, 0.9798285426122038]
# 500 epochs - [0.9995271867612293, 0.9789422135161606, 0.9853756933938477]
# 1000epochs - [0.9990543735224586, 0.9813907933398629, 0.9808371154815936]

# 0v1 2v3 8v9, batch_size = None, hidden_size = [256], Adam learning rate = 0.005
# 20  epochs - [0.9995271867612293, 0.9632713026444663, 0.961674230963187]
# 200 epochs - [0.9995271867612293, 0.9809010773751224, 0.9823499747856783]
# 500 epochs - [0.9995271867612293, 0.9760039177277179, 0.9747856782652546]

# 0v1 2v3 8v9, batch_size = None, hidden_size = [256], Adam learning rate = 0.001
# 20  epochs - [0.9990543735224586, 0.9559255631733594, 0.9520927887039838]
# 200 epochs - [0.9995271867612293, 0.9799216454456415, 0.9838628340897629]
# 500 epochs - [0.9995271867612293, 0.9809010773751224, 0.9828542612203732] ## Used this
# 1000epochs - [0.9995271867612293, 0.9760039177277179, 0.9757942511346445]

# 8v9 0v1 2v3, batch_size = None, hidden_size = [256], Adam learning rate = 0.001
# 20  epochs - [0.9641956631366616, 0.9971631205673759, 0.9564152791380999]
# 200 epochs - [0.983358547655068, 0.9995271867612293, 0.9755142017629774]

# 0v1 2v3 8v9, batch_size = None, hidden_size = [256, 256], Adam learning rate = 0.001
# 20  epochs - [0.9985815602836879, 0.9642507345739472, 0.9646999495713565]
# 200 epochs - [0.9995271867612293, 0.9906953966699314, 0.9884014120020171]
# 500 epochs - [0.9995271867612293, 0.9936336924583742, 0.9899142713061019]


## permutedMNIST, hidden_size = [100, 100]
# 1 hidden layer -> ~92-93% performance on all tasks
# 500 epochs - [0.9661, 0.9662, 0.9665]
# 1000epochs -


# Run vanilla VCL
tf.reset_default_graph()
tf.set_random_seed(12)
np.random.seed(1)

# splitMNIST
if option == 1:
    coreset_size = 0
    data_gen = SplitMnistGenerator()
    vcl_result = vcl.run_vcl_shared_ml(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, learning_rate=Adam_learning_rate)

# permutedMNIST
elif option == 2:
    coreset_size = 0
    data_gen = PermutedMnistGenerator(max_iter=3)
    vcl_result = vcl.run_vcl_shared_ml(hidden_size, no_epochs, data_gen,
        coreset.rand_from_batch, coreset_size, batch_size, learning_rate=Adam_learning_rate)

