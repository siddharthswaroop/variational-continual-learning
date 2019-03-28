import numpy as np
#import tensorflow as tf
import utils
from cla_models_multihead import MFVI_NN
from copy import deepcopy
#import pdb
import time

ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))


# Stores model weights (previous posterior weights = new prior weights)
class WeightsStorage():
    def __init__(self, no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0):
        # Initial mean and variance for lower network and upper network
        self.lower_mean = np.ones([no_lower_weights]) * prior_mean
        self.lower_log_var = np.ones([no_lower_weights]) * log_func(prior_var)
        self.upper_mean = [np.ones(no_weights) * prior_mean for no_weights in no_upper_weights]
        self.upper_log_var = [np.ones(no_weights) * log_func(prior_var) for no_weights in no_upper_weights]

    def return_weights(self):
        # Returns lower and upper weights that are currently stored (the previous posterior)
        upper_mv = []
        for class_ind in range(len(self.upper_mean)):
            upper_mv.append([deepcopy(self.upper_mean[class_ind]), deepcopy(self.upper_log_var[class_ind])])

        return (deepcopy(self.lower_mean), deepcopy(self.lower_log_var)), upper_mv

    def store_weights(self, post_l_mv, post_u_mv):
        # Store model weights
        self.lower_mean = deepcopy(post_l_mv[0])
        self.lower_log_var = deepcopy(post_l_mv[1])

        for class_ind in range(len(post_u_mv)):
            self.upper_mean[class_ind] = deepcopy(post_u_mv[class_ind][0])
            self.upper_log_var[class_ind] = deepcopy(post_u_mv[class_ind][1])

# Initialise model weights before training on new data, using small random means and small variances
def initialise_weights(weights):
    weights_mean_init = np.random.normal(size=weights[0].shape, scale=0.1)
    weights_log_var_init = np.ones_like(weights[1]) * (-6.0)
    return [weights_mean_init, weights_log_var_init]

# Run VCL on model; returns accuracies on each task after training on each task
def run_vcl_shared(hidden_size, no_epochs, data_gen, coreset_method, setting,
                   coreset_size=0, batch_size=None, path='sandbox/', calculate_acc=False, learning_rate=0.005, epoch_pause=[], store_weights=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []

    all_acc = np.array([])
    no_tasks = data_gen.max_iter
    for i in range(no_tasks):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)
        x_testsets.append(x_test)
        y_testsets.append(y_test)

    # Set up all the tasks
    # 1: multi-head
    # 2: like 1, but over all classes at test time
    # 3: like 2, but over classes observed so far at test time
    # 4: single-head
    # 5: single-head, but only for classes observed so far
    # 6: single-head [observed classes] during training, but multi-head during testing
    # setting = 5

    all_classes = range(data_gen.out_dim)
    training_loss_classes = [] # Training loss function depends on these classes
    training_classes = [] # Which classes' heads' weights change during training
    test_classes = [] # Which classes to compare between at test time
    observed_classes = [] # Which classes we have observed so far
    multi_head = False # True if test_classes is data_classes, indicating that we are dealing with multi-head at test-time
    # Toggling 'multi_head' in other situations can lead to more advanced settings during testing
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]
        observed_classes = observed_classes + data_classes

        if setting == 1:
            # Multi-head
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(data_classes)
            multi_head = True
        elif setting == 2:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(all_classes) # All classes
        elif setting == 3:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(observed_classes)  # Observed classes
        elif setting == 4 or 7:
            # Single-head
            training_loss_classes.append(all_classes)
            training_classes.append(all_classes)
            test_classes.append(all_classes)
        elif setting == 5:
            training_loss_classes.append(observed_classes) # Observed classes
            training_classes.append(observed_classes)
            test_classes.append(observed_classes)
        elif setting == 6:
            training_loss_classes.append(observed_classes) # Observed classes
            training_classes.append(observed_classes)
            test_classes.append(data_classes)
            multi_head = True

    # creating model
    no_heads = out_dim
    lower_size = [in_dim] + deepcopy(hidden_size)
    upper_sizes = [[hidden_size[-1], 1] for i in range(no_heads)]

    model = MFVI_NN(lower_size, upper_sizes, training_loss_classes=training_loss_classes, data_classes=data_gen.classes, use_float64=False) # TODO: Change use_float64 for split vs permuted MNIST
    no_lower_weights = model.lower_net.no_weights
    no_upper_weights = [net.no_weights for net in model.upper_nets]

    # Set up model weights at initial prior
    weights_storage = WeightsStorage(no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0)

    accuracies = []
    avg_cost = []
    avg_lik_cost = []
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]

        # init model
        model.init_session(task_id, learning_rate, training_classes[task_id])

        # get data
        x_train, y_train = x_trainsets[task_id], y_trainsets[task_id]

        # Set the readout head to train
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Prior of weights is previous posterior (or, if first task, already in weights_storage)
        lower_weights_prior, upper_weights_prior = weights_storage.return_weights()


        # # Initialise weights using the prior (previous posterior)
        # upper_weights = deepcopy(upper_weights_prior)

        # Initialise using random means + small variances
        lower_weights = initialise_weights(lower_weights_prior)
        upper_weights = np.empty_like(upper_weights_prior)
        for class_id in training_classes[task_id]:
            upper_weights[class_id] = deepcopy(initialise_weights(upper_weights_prior[class_id]))

        # # init using the prior or cavity
        # lower_post = init_post(lower_mv, init_using_cav=True)
        # upper_post = init_post(upper_mv, init_using_cav=True)


        # TODO: Rename lower/upper_post, factory, etc
        load_only_weight_0 = False
        if calculate_acc:
            no_epochs = 0
            no_digits = data_gen.out_dim

            if i == 0:
                load_weight = 0 if load_only_weight_0 else task_id
                # load_weight = 9
                res = np.load(path + 'weights_%d.npz' % load_weight)
                lower_post = res['lower']
                upper_post = res['upper']


                # Permuted MNIST - 2 hidden layers - [100, 100]
                if False:
                    if task_id == 0:
                        m_upper_new = upper_post[:,0]
                        var_upper_new = np.exp(upper_post[:,1])
                        #m_upper_new = m_upper_new.reshape([hidden_size[-1] + 1, no_digits])
                        #var_upper_new = var_upper_new.reshape([hidden_size[-1] + 1, no_digits])

                        in_dim = 784
                        no_params = 0
                        m_low_1_new = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[0]]
                        var_low_1_new = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[0]])
                        m_low_1_new = m_low_1_new.reshape([in_dim + 1, hidden_size[0]])
                        var_low_1_new = var_low_1_new.reshape([in_dim + 1, hidden_size[0]])

                        no_params = no_params + (in_dim + 1) * hidden_size[0]
                        in_dim = hidden_size[0]
                        m_low_2_new = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[1]]
                        var_low_2_new = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[1]])
                        m_low_2_new = m_low_2_new.reshape([in_dim + 1, hidden_size[1]])
                        var_low_2_new = var_low_2_new.reshape([in_dim + 1, hidden_size[1]])


                        #m_lower_new = lower_post[0, :]
                        #var_lower_new = np.exp(lower_post[1, :])
                        #m_lower_new = m_lower_new.reshape([in_dim + 1, hidden_size[0]])
                        #var_lower_new = var_lower_new.reshape([in_dim + 1, hidden_size[0]])

                    # Store upper/lower_post in intermediate values, which we will change
                    m_upper = upper_post[:,0]
                    var_upper = np.exp(upper_post[:,1])
                    #m_upper = m_upper.reshape([hidden_size[-1] + 1, no_digits])
                    #var_upper = var_upper.reshape([hidden_size[-1] + 1, no_digits])

                    in_dim = 784
                    no_params = 0
                    m_low_1 = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[0]]
                    var_low_1 = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[0]])
                    m_low_1 = m_low_1.reshape([in_dim + 1, hidden_size[0]])
                    var_low_1 = var_low_1.reshape([in_dim + 1, hidden_size[0]])

                    no_params = no_params + (in_dim + 1) * hidden_size[0]
                    in_dim = hidden_size[0]
                    m_low_2 = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[1]]
                    var_low_2 = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[1]])
                    m_low_2 = m_low_2.reshape([in_dim + 1, hidden_size[1]])
                    var_low_2 = var_low_2.reshape([in_dim + 1, hidden_size[1]])

                    ## This uses all weights as normal
                    #m_upper_old = m_upper
                    #var_upper_old = var_upper
                    #m_lower_old = m_lower
                    #var_lower_old = var_lower

                    # Set all weights to be pruned
                    m_upper_new = np.zeros_like(m_upper_new)
                    var_upper_new = 0.000001*np.ones_like(var_upper_new)
                    m_low_1_new = np.zeros_like(m_low_1_new)
                    var_low_1_new = 0.000001*np.ones_like(var_low_1_new)
                    m_low_2_new = np.zeros_like(m_low_2_new)
                    var_low_2_new = 0.000001*np.ones_like(var_low_2_new)

                    # Lower net neurons
                    #neuron_ids_1 = [0,2,5,7,8,9,14,22,24,29,30,32,34,37,38,41,46,49,59,63,66,70,71,77,78,83,84,85,88]
                    neuron_ids_1 = range(100) # Change all neurons' lower level weights
                    m_low_1_new[:,neuron_ids_1] = m_low_1[:,neuron_ids_1]
                    var_low_1_new[:,neuron_ids_1] = var_low_1[:,neuron_ids_1]

                    #neuron_ids_2 = [6,7,21,22,27,36,51,82,94,96,99]
                    neuron_ids_2 = range(100)
                    m_low_2_new[:,neuron_ids_2] = m_low_2[:,neuron_ids_2]
                    var_low_2_new[:,neuron_ids_2] = var_low_2[:,neuron_ids_2]

                    # Upper weight neurons
                    upper_weight_ids = range(10)  # Change all 10 classes' upper level weights
                    neuron_ids_upper = range(101)
                    neuron_ids_upper = [6,7,21,22,27,36,51,82,94,96,99,4,5,19,32,34,37,41,48,56,60,61,64,81]
                    for upper_weight in upper_weight_ids:
                        m_upper_new[upper_weight, neuron_ids_upper] = m_upper[upper_weight, neuron_ids_upper]
                        var_upper_new[upper_weight, neuron_ids_upper] = var_upper[upper_weight, neuron_ids_upper]

                    # Update weights back into upper/lower_post according to new values
                    m_lower_new = np.append(m_low_1_new.reshape([-1]), m_low_2_new.reshape([-1]))
                    var_lower_new = np.append(var_low_1_new.reshape([-1]),var_low_2_new.reshape([-1]))

                    upper_post[:,0] = m_upper_new
                    upper_post[:,1] = np.log(var_upper_new)
                    lower_post[0, :] = m_lower_new.reshape([-1])
                    lower_post[1, :] = np.log(var_lower_new.reshape([-1]))

                # Split MNIST - 1 hidden layer - [200]
                if False:
                    if task_id == 0:
                        m_upper_new = upper_post[:, 0]
                        var_upper_new = np.exp(upper_post[:, 1])

                        in_dim = 784
                        no_params = 0
                        m_low_1_new = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[0]]
                        var_low_1_new = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[0]])
                        m_low_1_new = m_low_1_new.reshape([in_dim + 1, hidden_size[0]])
                        var_low_1_new = var_low_1_new.reshape([in_dim + 1, hidden_size[0]])



                    # Store upper/lower_post in intermediate values, which we will change
                    m_upper = upper_post[:, 0]
                    var_upper = np.exp(upper_post[:, 1])

                    in_dim = 784
                    no_params = 0
                    m_low_1 = lower_post[0, no_params:no_params + (in_dim + 1) * hidden_size[0]]
                    var_low_1 = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * hidden_size[0]])
                    m_low_1 = m_low_1.reshape([in_dim + 1, hidden_size[0]])
                    var_low_1 = var_low_1.reshape([in_dim + 1, hidden_size[0]])

                    ## This uses all weights as normal
                    # m_upper_old = m_upper
                    # var_upper_old = var_upper
                    # m_lower_old = m_lower
                    # var_lower_old = var_lower

                    # Set all weights to be pruned
                    m_upper_new = np.zeros_like(m_upper_new)
                    var_upper_new = 0.000000001 * np.ones_like(var_upper_new)
                    m_low_1_new = np.zeros_like(m_low_1_new)
                    var_low_1_new = 0.000000001 * np.ones_like(var_low_1_new)

                    # Lower net neurons
                    neuron_ids_1 = [126,183,67,63,179]
                    #neuron_ids_1 = range(200)  # Change all neurons' lower level weights
                    m_low_1_new[:, neuron_ids_1] = m_low_1[:, neuron_ids_1]
                    var_low_1_new[:, neuron_ids_1] = var_low_1[:, neuron_ids_1]

                    # Upper weight neurons
                    upper_weight_ids = range(10)  # Change all 10 classes' upper level weights
                    neuron_ids_upper = range(201)
                    #neuron_ids_upper = [6, 7, 21, 22, 27, 36, 51, 82, 94, 96, 99, 4, 5, 19, 32, 34, 37, 41, 48, 56, 60, 61, 64, 81]
                    for upper_weight in upper_weight_ids:
                        m_upper_new[upper_weight, neuron_ids_upper] = m_upper[upper_weight, neuron_ids_upper]
                        var_upper_new[upper_weight, neuron_ids_upper] = var_upper[upper_weight, neuron_ids_upper]

                    # Update weights back into upper/lower_post according to new values
                    #m_lower_new = np.append(m_low_1_new.reshape([-1]), m_low_2_new.reshape([-1]))
                    #var_lower_new = np.append(var_low_1_new.reshape([-1]), var_low_2_new.reshape([-1]))
                    m_lower_new = m_low_1_new.reshape([-1])
                    var_lower_new = var_low_1_new.reshape([-1])

                    upper_post[:, 0] = m_upper_new
                    upper_post[:, 1] = np.log(var_upper_new)
                    lower_post[0, :] = m_lower_new.reshape([-1])
                    lower_post[1, :] = np.log(var_lower_new.reshape([-1]))

        # Assign initial weights to the model
        model.assign_weights(range(no_heads), lower_weights, upper_weights)

        # Train on non-coreset data
        model.reset_optimiser()

        if setting == 7:
            if task_id == 0:
                no_epochs = 500
            elif task_id == 1:
                no_epochs = 400
            elif task_id == 2:
                no_epochs = 300

        start_time = time.time()
        avg_cost, avg_lik_cost = model.train(x_train, y_train, data_classes, task_id,
                        lower_weights_prior, upper_weights_prior, no_epochs, bsize)
        end_time = time.time()
        print 'Time taken to train (s):', end_time - start_time

        # Get weights from model, and store in weights_storage
        lower_weights, upper_weights = model.get_weights(range(no_heads))
        weights_storage.store_weights(lower_weights, upper_weights)

        # Loop through the coresets, for each find coreset factor # TODO: Rename upper/lower_post, factory, etc
        if coreset_size > 0:
            no_coresets = len(x_coresets)
            no_coresets=1
            for k in range(no_coresets):
                # pdb.set_trace()
                x_coreset_k = x_coresets[k]
                y_coreset_k = y_coresets[k]
                # compute cavity and this is now the prior
                lower_data_idx = range(task_id + 1)
                lower_core_idx = range(task_id + 1)
                lower_core_idx.remove(k)
                lower_cav, upper_cav = factory.compute_dist(
                    lower_data_idx, lower_core_idx,
                    [task_id], remove_core=True, remove_data=False)
                lower_mv = [lower_cav[0], lower_cav[1]]
                upper_mv = [upper_cav[0][0], upper_cav[1][0]]
                lower_n = [lower_cav[2], lower_cav[3]]
                upper_n = [upper_cav[2][0], upper_cav[3][0]]
                model.assign_weights(head, lower_post, upper_post, ide_func, ide_func)
                # train on coreset data
                model.reset_optimiser()
                model.train(x_coreset_k, y_coreset_k, head, lower_mv, upper_mv, no_epochs, bsize)
                # get params and update factor
                lower_post, upper_post = model.get_weights(head)
                factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id,
                                      data_factor=False, core_factor=True)

        # Save model weights after training
        if store_weights:
            np.savez(path + 'weights_%d.npz' % i, lower=lower_weights, upper=upper_weights, classes=data_gen.classes,
                     MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list)
            np.savez(path + 'costs_%d.npz' % i, avg_cost=avg_cost, avg_lik_cost=avg_lik_cost)

        # Test-time: load weights, and calculate test accuracy
        lower_weights, upper_weights = weights_storage.return_weights()

        if not calculate_acc:
            model.assign_weights(range(no_heads), lower_weights, upper_weights)

        #acc, pred_vec, pred_vec_true, pred_vec_total = utils.get_scores_output_pred(model, x_testsets, y_testsets, test_classes, task_id=task_id, multi_head=multi_head)
        acc = utils.get_scores_output_pred(model, x_testsets, y_testsets,test_classes, task_id=task_id, multi_head=multi_head)

        if task_id == 0:
            all_acc = np.array(acc)
        else:
            all_acc = np.vstack([all_acc, acc])
        print all_acc

        # Store things
        # lower_weights, upper_weights = model.get_weights(range(no_heads))
        # store_weights = True
        # store_pred_values = False
        # store_cost_values = False
        # if store_weights and not calculate_acc:
        #     store_cost_values = True
        # if store_weights:
        #     np.savez(path + 'weights_%d.npz' % task_id, lower=lower_weights, upper=upper_weights,
        #              classes=data_gen.classes, MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list)
        # if store_pred_values:
        #     np.savez(path + 'pred_%d.npz' % task_id, pred_true=pred_vec_true, pred=pred_vec, pred_total=pred_vec_total)
        # if store_cost_values:
        #     np.savez(path + 'costs_%d.npz' % task_id, avg_cost=avg_cost, avg_lik_cost=avg_lik_cost)
        #np.savez('sandbox/smallinitalways/accuracy_%d.npz' % task_id, acc=acc_interm, ind=[x+1 for x in epoch_pause])

        model.close_session()

    return all_acc


"""
def run_vcl_shared_calculate_acc(hidden_size, no_epochs, data_gen, coreset_method, setting,
                   coreset_size=0, batch_size=None, path='sandbox/', calculate_acc=False, no_iters=1, learning_rate=0.005, epoch_pause=[], store_weights=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []
    x_trainsets, y_trainsets = [], []

    all_acc = np.array([])
    no_tasks = data_gen.max_iter
    for i in range(no_tasks):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_trainsets.append(x_train)
        y_trainsets.append(y_train)
        x_testsets.append(x_test)
        y_testsets.append(y_test)

    # Set up all the tasks
    # 1: multi-head
    # 2: like 1, but over all classes at test time
    # 3: like 2, but over classes observed so far at test time
    # 4: single-head
    # 5: single-head, but only for classes observed so far
    # 6: single-head [observed classes] during training, but multi-head during testing
    # setting = 5

    all_classes = range(data_gen.out_dim)
    training_loss_classes = [] # Training loss function depends on these classes
    training_classes = [] # Which classes' heads' weights change during training
    test_classes = [] # Which classes to compare between at test time
    observed_classes = [] # Which classes we have observed so far
    multi_head = False # True if test_classes is data_classes, indicating that we are dealing with multi-head at test-time
    # Toggling 'multi_head' in other situations can lead to more advanced settings during testing
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]
        observed_classes = observed_classes + data_classes

        if setting == 1:
            # Multi-head
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(data_classes)
            multi_head = True
        elif setting == 2:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(all_classes) # All classes
        elif setting == 3:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(observed_classes)  # Observed classes
        elif setting == 4 or 7:
            # Single-head
            training_loss_classes.append(all_classes)
            training_classes.append(all_classes)
            test_classes.append(all_classes)
        elif setting == 5:
            training_loss_classes.append(observed_classes) # Observed classes
            training_classes.append(observed_classes)
            test_classes.append(observed_classes)
        elif setting == 6:
            training_loss_classes.append(observed_classes) # Observed classes
            training_classes.append(observed_classes)
            test_classes.append(data_classes)
            multi_head = True

    # creating model
    no_heads = out_dim
    lower_size = [in_dim] + deepcopy(hidden_size)
    upper_sizes = [[hidden_size[-1], 1] for i in range(no_heads)]

    model = MFVI_NN(lower_size, upper_sizes, training_loss_classes=training_loss_classes, data_classes=data_gen.classes)
    no_lower_weights = model.lower_net.no_weights
    no_upper_weights = [net.no_weights for net in model.upper_nets]

    factory = FactorManager(no_tasks, no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0)

    lower_post_epoch = []
    upper_post_epoch = []
    accuracies = []
    avg_cost = []
    avg_lik_cost = []
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]

        # init model
        model.init_session(task_id, learning_rate, training_classes[task_id])

        # get data
        x_train, y_train = x_trainsets[task_id], y_trainsets[task_id]

        # Set the readout head to train
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(
                x_coresets, y_coresets, x_train, y_train, coreset_size)

        acc_interm = [] # Stores intermediate values of accuracy for plotting

        for i in range(no_iters):
            #print 'sizeof', np.size(upper_mv), np.size(upper_mv[0]), np.size(upper_mv[0][0]), np.size(upper_mv[0][0][0])

            no_epochs = 0
            no_digits = data_gen.out_dim
            load_weight = task_id ### For Permuted MNIST
            #load_weight = i ### For FullMNIST
            res = np.load(path + 'weights_%d.npz' % load_weight)
            lower_post = res['lower']
            upper_post = res['upper']

            model.assign_weights(range(no_heads), lower_post, upper_post) ####################################


            #model.assign_weights(range(no_heads), lower_post, upper_post) ####################################

            # train on non-coreset data
            model.reset_optimiser()

            if setting == 7:
                if task_id == 0:
                    no_epochs = 500
                elif task_id == 1:
                    no_epochs = 400
                elif task_id == 2:
                    no_epochs = 300

            # start_time = time.time()
            # avg_cost, avg_lik_cost, lower_post_epoch, upper_post_epoch = model.train(x_train, y_train, data_classes, task_id, lower_mv, upper_mv, no_epochs, bsize, epoch_pause=epoch_pause)
            # end_time = time.time()
            #
            # print 'Time taken to train (s):', end_time - start_time

            # get params and update factor
            #lower_post, upper_post = model.get_weights(range(no_heads)) ###################################

            #factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id, data_factor=True, core_factor=False) #############################

            # np.savez(path + 'weights_%d.npz' % i, lower=lower_post, upper=upper_post, classes=data_gen.classes, MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list)  ##################
            # np.savez(path + 'costs_%d.npz' % i, avg_cost=avg_cost, avg_lik_cost=avg_lik_cost) ############################


            #np.savez(path + 'const_upper/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)
            #np.savez('sandbox/weights_%d_epoch.npz' % task_id, lower=lower_post_epoch, upper=upper_post_epoch)

            # Make prediction ######################
            # lower_post, upper_post = factory.compute_dist(range(no_tasks), range(no_tasks), range(no_tasks), False, False)
            # lower_mv = deepcopy([lower_post[0], lower_post[1]])
            # upper_mv = deepcopy(upper_post)

            acc, _, _, _ = utils.get_scores_output_pred(model, x_testsets, y_testsets, test_classes, no_repeats=1, task_id=task_id, multi_head=multi_head)

            all_acc = np.append(np.array(all_acc), acc)
            print task_id, all_acc


        # lower_post, upper_post = model.get_weights(range(no_heads))
        # #store_weights = True
        # store_pred_values = False
        # store_cost_values = False
        # if store_weights and not calculate_acc:
        #     store_cost_values = True
        # if store_weights:
        #     np.savez(path + 'weights_%d.npz' % task_id, lower=lower_post, upper=upper_post, classes=data_gen.classes, MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list) ##############
        # if store_pred_values:
        #     np.savez(path + 'pred_%d.npz' % task_id, pred_true=pred_vec_true, pred=pred_vec, pred_total=pred_vec_total)
        # if store_cost_values:
        #     np.savez(path + 'costs_%d.npz' % task_id, avg_cost=avg_cost, avg_lik_cost=avg_lik_cost)
        # #np.savez('sandbox/smallinitalways/accuracy_%d.npz' % task_id, acc=acc_interm, ind=[x+1 for x in epoch_pause])

        model.close_session()

    np.savez(path + 'test_acc.npz', acc=all_acc)

    return all_acc, accuracies

"""