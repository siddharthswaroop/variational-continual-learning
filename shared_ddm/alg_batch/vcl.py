import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import MFVI_NN, ML_NN
from copy import deepcopy
import pdb

ide_func = lambda x: np.float32(x)
log_func = lambda x: np.float32(np.log(x))
exp_func = lambda x: np.float32(np.exp(x))


class FactorManager():
    def __init__(self, no_tasks, no_lower_weights, no_upper_weights):
        # natural parameters
        # non-coreset factors
        self.dl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.dl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.du_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.du_n2 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        # coreset factors
        self.cl_n1 = np.zeros([no_tasks, no_lower_weights])
        self.cl_n2 = np.zeros([no_tasks, no_lower_weights])
        self.cu_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.cu_n2 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        # prior factors
        self.pl_n1 = np.zeros([no_lower_weights])
        self.pl_n2 = np.ones([no_lower_weights])
        self.pu_n1 = [np.zeros([no_weights]) for no_weights in no_upper_weights]
        self.pu_n2 = [np.ones([no_weights]) for no_weights in no_upper_weights]

    def compute_dist(self, dl_idx, cl_idx, task_idx, remove_data, remove_core):
        dl_n1 = np.sum(self.dl_n1[dl_idx, :], axis=0)
        dl_n2 = np.sum(self.dl_n2[dl_idx, :], axis=0)
        cl_n1 = np.sum(self.cl_n1[cl_idx, :], axis=0)
        cl_n2 = np.sum(self.cl_n2[cl_idx, :], axis=0)
        l_n1 = self.pl_n1 + dl_n1 + cl_n1
        l_n2 = self.pl_n2 + dl_n2 + cl_n2
        l_v = 1.0 / l_n2
        l_m = l_v * l_n1

        # l_v[np.where(l_v<0)[0]] = 1.0
        # l_n2[np.where(l_n2<0)[0]] = 1.0

        u_n1, u_n2, u_m, u_v = [], [], [], []
        for i in task_idx:
            du_n1 = self.du_n1[i]
            du_n2 = self.du_n2[i]
            cu_n1 = self.cu_n1[i]
            cu_n2 = self.cu_n2[i]
            pu_n1 = self.pu_n1[i]
            pu_n2 = self.pu_n2[i]
            u_n1_i = pu_n1
            u_n2_i = pu_n2
            if not remove_core:
                u_n1_i += cu_n1
                u_n2_i += cu_n2
            if not remove_data:
                u_n1_i += du_n1
                u_n2_i += du_n2
            u_v_i = 1.0 / u_n2_i
            u_m_i = u_v_i * u_n1_i

            u_v_i[np.where(u_v_i < 0)[0]] = 1.0
            u_n2_i[np.where(u_n2_i < 0)[0]] = 1.0

            u_n1.append(u_n1_i)
            u_n2.append(u_n2_i)
            u_m.append(u_m_i)
            u_v.append(u_v_i)
        return (l_m, l_v, l_n1, l_n2), (u_m, u_v, u_n1, u_n2)

    def update_factor(self, post_l_mv, post_u_mv, cav_l_n, cav_u_n,
                      task_idx, data_factor, core_factor, transform_func=np.exp):
        post_l_m, post_l_v = post_l_mv[0], transform_func(post_l_mv[1])
        post_u_m, post_u_v = post_u_mv[0], transform_func(post_u_mv[1])
        post_l_n1, post_l_n2 = post_l_m / post_l_v, 1.0 / post_l_v
        post_u_n1, post_u_n2 = post_u_m / post_u_v, 1.0 / post_u_v
        f_l_n1 = post_l_n1 - cav_l_n[0]
        f_l_n2 = post_l_n2 - cav_l_n[1]
        f_u_n1 = post_u_n1 - cav_u_n[0]
        f_u_n2 = post_u_n2 - cav_u_n[1]
        if data_factor:
            self.dl_n1[task_idx, :] = f_l_n1
            self.dl_n2[task_idx, :] = f_l_n2
            self.du_n1[task_idx] = f_u_n1
            self.du_n2[task_idx] = f_u_n2
        else:
            self.cl_n1[task_idx, :] = f_l_n1
            self.cl_n2[task_idx, :] = f_l_n2
            self.cu_n1[task_idx] = f_u_n1
            self.cu_n2[task_idx] = f_u_n2


def init_post(cav_info, init_using_cav, ml_weights=None):
    if init_using_cav:
        return cav_info
    else:
        cav_mean = cav_info[0]
        cav_var = cav_info[1]
        if ml_weights:
            post_mean = ml_weights[0]
        else:
            post_mean = np.random.normal(size=cav_mean.shape, scale=0.1)
        post_var = np.ones_like(cav_var) * np.exp(-6.0)
        return [post_mean, post_var]


def run_vcl_shared_ml(hidden_size, no_epochs, data_gen, coreset_method,
                   coreset_size=0, batch_size=None, no_iters=1, learning_rate=0.005):
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

    # creating model
    lower_size = [in_dim] + deepcopy(hidden_size)
    upper_sizes = [[hidden_size[-1], out_dim] for i in range(no_tasks)]
    # model = MFVI_NN(lower_size, upper_sizes)
    # we also create a model trained using maximum likelihood
    ml_model = ML_NN(lower_size, upper_sizes)
    no_lower_weights = ml_model.lower_net.no_weights
    no_upper_weights = [net.no_weights for net in ml_model.upper_nets]

    # get data
    x_train, y_train = x_trainsets[0], y_trainsets[0]
    x_test, y_test = x_testsets[0], y_testsets[0]
    bsize = x_train.shape[0] if (batch_size is None) else batch_size

    ## init using the maximum likeihood solution + small variances
    ml_model.init_session(learning_rate=0.0005)
    ml_model.train(x_trainsets, y_trainsets, no_tasks,
                   no_epochs, batch_size=bsize)

    upper_post = []
    for task_id in range(no_tasks):
        lower_post, upper_post_interm = ml_model.get_weights(task_id)
        upper_post.append(upper_post_interm)

    np.savez('sandbox/weights_ml_batch.npz', lower=lower_post, upper=upper_post)

    # Make prediction
    lower_mv = lower_post
    upper_mv = [upper_post[i] for i in range(no_tasks)]
    ml_model.assign_weights(range(no_tasks), lower_mv, upper_mv)
    # pdb.set_trace()
    acc = utils.get_scores_ml(ml_model, x_testsets, y_testsets)
    print acc

    # all_acc = utils.concatenate_results(acc, all_acc)
    # pdb.set_trace()
    ml_model.close_session()

    return acc