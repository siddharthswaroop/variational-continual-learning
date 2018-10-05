import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import MFVI_NN
from copy import deepcopy
import pdb

ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))


class FactorManager():
    def __init__(self, no_tasks, no_lower_weights, no_upper_weights, prior_mean=0.0, prior_var=1.0):
        # natural parameters
        # non-coreset factors
        self.dl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.dl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.du_n1 = [np.zeros(no_weights) for no_weights in no_upper_weights]
        self.du_n2 = [np.zeros(no_weights) for no_weights in no_upper_weights]
        # coreset factors
        self.cl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.cl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.cu_n1 = [np.zeros(no_weights) for no_weights in no_upper_weights]
        self.cu_n2 = [np.zeros(no_weights) for no_weights in no_upper_weights]
        # prior factors
        self.pl_n1 = np.ones([no_lower_weights])*prior_mean/prior_var
        self.pl_n2 = np.ones([no_lower_weights])/prior_var
        self.pu_n1 = [np.ones(no_weights)*prior_mean/prior_var for no_weights in no_upper_weights]
        self.pu_n2 = [np.ones(no_weights)/prior_var for no_weights in no_upper_weights]

        # Mean and variance parameters
        self.dl_m = np.ones([no_lower_weights]) * prior_mean
        self.dl_v = np.ones([no_lower_weights]) * log_func(prior_var)
        self.du_m = [np.ones(no_weights) * prior_mean for no_weights in no_upper_weights]
        self.du_v = [np.ones(no_weights) * log_func(prior_var) for no_weights in no_upper_weights]

        #print 'factor manager du_m', np.size(self.du_m[0]), no_upper_weights

    def compute_dist(self, dl_idx, cl_idx, task_idx, remove_data, remove_core):
        # dl_n1 = np.sum(self.dl_n1[dl_idx, :], axis=0)
        # dl_n2 = np.sum(self.dl_n2[dl_idx, :], axis=0)
        # cl_n1 = np.sum(self.cl_n1[cl_idx, :], axis=0)
        # cl_n2 = np.sum(self.cl_n2[cl_idx, :], axis=0)
        # l_n1 = self.pl_n1 + dl_n1 + cl_n1
        # l_n2 = self.pl_n2 + dl_n2 + cl_n2
        # l_v = 1.0 / l_n2
        # l_m = l_v * l_n1
        #
        # # l_v[np.where(l_v<0)[0]] = 1.0
        # # l_n2[np.where(l_n2<0)[0]] = 1.0
        #
        # u_n1, u_n2, u_m, u_v = [], [], [], []
        # no_heads_idx = [0] if single_head else task_idx
        # for i in no_heads_idx:
        #     du_n1 = self.du_n1[i]
        #     du_n2 = self.du_n2[i]
        #     cu_n1 = self.cu_n1[i]
        #     cu_n2 = self.cu_n2[i]
        #     pu_n1 = self.pu_n1[i]
        #     pu_n2 = self.pu_n2[i]
        #     u_n1_i = pu_n1
        #     u_n2_i = pu_n2
        #     if not remove_core:
        #         u_n1_i += cu_n1
        #         u_n2_i += cu_n2
        #     if not remove_data:
        #         u_n1_i += du_n1
        #         u_n2_i += du_n2
        #     u_v_i = 1.0 / u_n2_i
        #     u_m_i = u_v_i * u_n1_i
        #
        #     u_v_i[np.where(u_v_i < 0)[0]] = 1.0
        #     u_n2_i[np.where(u_n2_i < 0)[0]] = 1.0
        #
        #     u_n1.append(u_n1_i)
        #     u_n2.append(u_n2_i)
        #     u_m.append(u_m_i)
        #     u_v.append(u_v_i)
        #return (l_m, l_v, l_n1, l_n2), (u_m, u_v, u_n1, u_n2)

        upper_mv = []
        for class_ind in range(len(self.du_m)):
            upper_mv.append([self.du_m[class_ind], self.du_v[class_ind]])

        return (self.dl_m, self.dl_v), upper_mv

    def update_factor(self, post_l_mv, post_u_mv, cav_l_n, cav_u_n,
                      task_idx, data_factor, core_factor, transform_func=np.exp):

        #post_u_m = np.zeros(np.size(post_u_mv[0]))
        #post_u_v = np.zeros(np.size(post_u_mv[0]))
        #no_params = np.size(post_u_mv[0][0])
        #for i in range(len(post_u_mv[0])):
        #    post_u_m[i*no_params:(i+1)*no_params] = post_u_mv[0][i]
        #    post_u_v[i*no_params:(i+1)*no_params] = transform_func(post_u_mv[0][i])

        # post_l_m, post_l_v = post_l_mv[0], transform_func(post_l_mv[1])
        # post_u_m, post_u_v = post_u_mv[0], transform_func(post_u_mv[1])
        # post_l_n1, post_l_n2 = post_l_m / post_l_v, 1.0 / post_l_v
        # post_u_n1, post_u_n2 = post_u_m / post_u_v, 1.0 / post_u_v
        # f_l_n1 = post_l_n1 - cav_l_n[0]
        # f_l_n2 = post_l_n2 - cav_l_n[1]
        #
        # f_u_n1 = post_u_n1 - cav_u_n[0]
        # f_u_n2 = post_u_n2 - cav_u_n[1]
        # head_idx = 0 if single_head else task_idx
        # if data_factor:
        #     self.dl_n1[task_idx, :] = f_l_n1
        #     self.dl_n2[task_idx, :] = f_l_n2
        #     self.du_n1[head_idx] = f_u_n1
        #     self.du_n2[head_idx] = f_u_n2
        # else:
        #     self.cl_n1[task_idx, :] = f_l_n1
        #     self.cl_n2[task_idx, :] = f_l_n2
        #     self.cu_n1[head_idx] = f_u_n1
        #     self.cu_n2[head_idx] = f_u_n2

        #head = 0 if single_head else task_idx

        self.dl_m = deepcopy(post_l_mv[0])
        self.dl_v = deepcopy(post_l_mv[1])

        for class_id in range(len(post_u_mv)):
            self.du_m[class_id] = deepcopy(post_u_mv[class_id][0])
            self.du_v[class_id] = deepcopy(post_u_mv[class_id][1])


def init_post(cav_info, init_using_cav, ml_weights=None):
    if init_using_cav:
        return cav_info
    else:
        cav_mean = np.array(cav_info[0])
        cav_var = cav_info[1]
        if ml_weights is not None:
            post_mean = ml_weights[0]
        else:
            post_mean = np.random.normal(size=cav_mean.shape, scale=0.1)
        post_var = np.ones_like(cav_var) * (-6.0)
        return [post_mean, post_var]


def run_vcl_shared(hidden_size, no_epochs, data_gen, coreset_method, setting,
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
    # setting = 5

    all_classes = range(data_gen.out_dim)
    training_loss_classes = [] # Training loss function depends on these classes
    training_classes = [] # Which classes' heads' weights change during training
    test_classes = [] # Which classes to compare between at test time
    observed_classes = [] # Which classes we have observed so far
    for task_id in range(no_tasks):
        # The data input classes for this task
        data_classes = data_gen.classes[task_id]
        observed_classes = observed_classes + data_classes

        if setting == 1:
            # Multi-head
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(data_classes)
        elif setting == 2:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(all_classes) # All classes
        elif setting == 3:
            training_loss_classes.append(data_classes)
            training_classes.append(data_classes)
            test_classes.append(observed_classes)  # Observed classes
        elif setting == 4:
            # Single-head
            training_loss_classes.append(all_classes)
            training_classes.append(all_classes)
            test_classes.append(all_classes)
        elif setting == 5:
            training_loss_classes.append(observed_classes) # Observed classes
            training_classes.append(observed_classes)
            test_classes.append(observed_classes)

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
            # finding data factor
            # compute cavity
            lower_data_idx = range(task_id)
            lower_core_idx = range(task_id + 1)
            lower_cav, upper_cav = factory.compute_dist(lower_data_idx, lower_core_idx, [task_id], remove_core=False, remove_data=True)
            lower_mv = deepcopy([lower_cav[0], lower_cav[1]])
            upper_mv = deepcopy(upper_cav)
            lower_n = []
            upper_n = []

            #print 'sizeof', np.size(upper_mv), np.size(upper_mv[0]), np.size(upper_mv[0][0]), np.size(upper_mv[0][0][0])

            # Small init or prior init only (removed ml init option)
            if i == 0:
                # init using the prior (previous posterior)
                upper_post = deepcopy(upper_mv)

                # init using random means + small variances
                lower_post = init_post(lower_mv, init_using_cav=False)
                for class_id in training_classes[task_id]:
                    upper_post[class_id] = deepcopy(init_post(upper_mv[class_id], init_using_cav=False))

                ## init using the prior or cavity
                # lower_post = init_post(lower_mv, init_using_cav=True)
                # upper_post = init_post(upper_mv, init_using_cav=True)


            if calculate_acc:
                no_epochs = 0
                no_digits = data_gen.out_dim

                if i == 0:
                    res = np.load(path + 'weights_%d.npz' % task_id)
                    lower_post = res['lower']
                    upper_post = res['upper']


                    # if task_id == 0:
                    #     m_upper_new = upper_post[0, :]
                    #     var_upper_new = np.exp(upper_post[1, :])
                    #     m_upper_new = m_upper_new.reshape([hidden_size[-1] + 1, no_digits])
                    #     var_upper_new = var_upper_new.reshape([hidden_size[-1] + 1, no_digits])
                    #
                    #     in_dim = 784
                    #     m_lower_new = lower_post[0, :]
                    #     var_lower_new = np.exp(lower_post[1, :])
                    #     m_lower_new = m_lower_new.reshape([in_dim + 1, hidden_size[0]])
                    #     var_lower_new = var_lower_new.reshape([in_dim + 1, hidden_size[0]])
                    #
                    # # Store upper/lower_post in intermediate values, which we will change
                    # m_upper = upper_post[0, :]
                    # var_upper = np.exp(upper_post[1, :])
                    # m_upper = m_upper.reshape([hidden_size[-1] + 1, no_digits])
                    # var_upper = var_upper.reshape([hidden_size[-1] + 1, no_digits])
                    # m_lower = lower_post[0, :]
                    # var_lower = np.exp(lower_post[1, :])
                    # m_lower = m_lower.reshape([in_dim + 1, hidden_size[0]])
                    # var_lower = var_lower.reshape([in_dim + 1, hidden_size[0]])
                    #
                    # ## This uses all weights as normal
                    # #m_upper_old = m_upper
                    # #var_upper_old = var_upper
                    # #m_lower_old = m_lower
                    # #var_lower_old = var_lower
                    #
                    # ## Set all weights to be pruned
                    # #m_upper_new = np.zeros_like(m_upper_new)
                    # #var_upper_new = 0.000001*np.ones_like(var_upper_new)
                    # #m_lower_new = np.zeros_like(m_lower_new)
                    # #var_lower_new = 0.000001*np.ones_like(var_lower_new)
                    #
                    # # Lower net neurons
                    # #neuron_ids = [114, 251, 7, 220, 157]
                    # neuron_ids = range(256) # Change all 256 neurons' lower level weights
                    # m_lower_new[:,neuron_ids] = m_lower[:,neuron_ids]
                    # var_lower_new[:,neuron_ids] = var_lower[:,neuron_ids]
                    #
                    # # Upper level neurons
                    # if upper_weights_const:
                    #     upper_weight_ids = [task_id*2,task_id*2+1] # Only relevant classes' upper level weights will be changed/updated
                    # else:
                    #     upper_weight_ids = range(10)  # Change all 10 classes' upper level weights
                    # for upper_weight in upper_weight_ids:
                    #     m_upper_new[neuron_ids,upper_weight] = m_upper[neuron_ids,upper_weight]
                    #     var_upper_new[neuron_ids,upper_weight] = var_upper[neuron_ids,upper_weight]
                    #
                    # # Update weights back into upper/lower_post according to new values
                    # upper_post[0, :] = m_upper_new.reshape([-1])
                    # upper_post[1, :] = np.log(var_upper_new.reshape([-1]))
                    # lower_post[0, :] = m_lower_new.reshape([-1])
                    # lower_post[1, :] = np.log(var_lower_new.reshape([-1]))


            model.assign_weights(range(no_heads), lower_post, upper_post)
            # train on non-coreset data
            model.reset_optimiser()
            _, lower_post_epoch, upper_post_epoch = model.train(x_train, y_train, data_classes, task_id, lower_mv, upper_mv, no_epochs, bsize, epoch_pause=epoch_pause)

            # get params and update factor
            lower_post, upper_post = model.get_weights(range(no_heads))

            factory.update_factor(lower_post, upper_post, lower_n, upper_n, task_id,
                                  data_factor=True, core_factor=False)

            # loop through the coresets, for each find coreset factor
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


        #np.savez(path + 'const_upper/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)
        #np.savez('sandbox/weights_%d_epoch.npz' % task_id, lower=lower_post_epoch, upper=upper_post_epoch)

        # Make prediction
        lower_post, upper_post = factory.compute_dist(range(no_tasks), range(no_tasks), range(no_tasks), False, False)
        lower_mv = deepcopy([lower_post[0], lower_post[1]])
        upper_mv = deepcopy(upper_post)

        model.assign_weights(range(no_heads), lower_mv, upper_mv)

        acc, pred_vec, pred_vec_true, pred_vec_total = utils.get_scores_output_pred(model, x_testsets, y_testsets, test_classes, task_id=task_id)
        if task_id == 0:
            all_acc = np.array(acc)
        else:
            all_acc = np.vstack([all_acc, acc])
        print all_acc


        for epoch_ind in range(len(lower_post_epoch)):

            lower_post = lower_post_epoch[epoch_ind]
            upper_post = upper_post_epoch[epoch_ind]
            lower_mv1 = [lower_post[0], lower_post[1]]
            upper_mv1 = [upper_post[0], upper_post[1]]
            model.assign_weights(task_id, lower_mv1, upper_mv1, ide_func, ide_func)
            acc = utils.get_scores(model, x_testsets, y_testsets)
            if epoch_ind == 0:
                acc_interm = np.array(acc)
            else:
                acc_interm = np.vstack([acc_interm, acc])
            #acc_interm.append(acc)
            #print acc

        if task_id == 0:
            accuracies = np.array(acc_interm)
        else:
            accuracies = np.vstack([accuracies, acc_interm])

        lower_post, upper_post = model.get_weights(range(no_heads))
        #store_weights = True
        store_pred_values = False
        if store_weights:
            np.savez(path + 'weights_%d.npz' % task_id, lower=lower_post, upper=upper_post, classes=data_gen.classes, MNISTdigits=data_gen.sets, class_index_conversion=data_gen.class_list)
        if store_pred_values:
            np.savez(path + 'pred_%d.npz' % task_id, pred_true=pred_vec_true, pred=pred_vec, pred_total=pred_vec_total)
        #np.savez('sandbox/smallinitalways/accuracy_%d.npz' % task_id, acc=acc_interm, ind=[x+1 for x in epoch_pause])

        model.close_session()

    return all_acc, accuracies