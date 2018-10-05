import tensorflow as tf
import numpy as np
from copy import deepcopy
import pdb

np.random.seed(0)
tf.set_random_seed(0)


def _create_weights(size):
    no_layers = len(size) - 1
    no_weights = 0
    for i in range(no_layers):
        no_weights += size[i] * size[i + 1] + size[i + 1]

    m = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
    v = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
    return no_weights, m, v

def _create_weights_fix_upper_weights(size):

    m_vec = []
    #m = []
    v_vec = []
    #v = []
    new_m = []
    new_v = []
    # Upper layer, so no_layers = 1, len(size) = 2
    # Go over all output classes
    for j in range(size[1]):
        m_vec_interm = tf.Variable(tf.constant(np.zeros([size[0] + 1]), dtype=tf.float32), name='mean_%d' % (j))
        m_vec.append(m_vec_interm)
        #m = tf.concat([m, m_vec_interm], 0)

        v_vec_interm = tf.Variable(tf.constant(np.zeros([size[0] + 1]), dtype=tf.float32), name='var_%d' % (j))
        v_vec.append(v_vec_interm)
        #v = tf.concat([v, v_vec_interm], 0)

        new_m_interm = tf.placeholder(tf.float32, [size[0] + 1])
        new_m.append(new_m_interm)

        new_v_interm = tf.placeholder(tf.float32, [size[0] + 1])
        new_v.append(new_v_interm)

    no_weights = (size[0]+1) * size[1]

    #m = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))
    #v = tf.Variable(tf.constant(np.zeros([no_weights]), dtype=tf.float32))

    #m = tf.reshape(m_vec, [no_weights])
    #v = tf.reshape(v_vec, [no_weights])
    #new_m1 = tf.reshape(new_m, [size[1], size[0]+1])
    #new_v1 = tf.reshape(new_v, [size[1], size[0]+1])

    return no_weights, m_vec, v_vec#, new_m1, new_v1

def _unpack_weights_fix_upper_weights(m, v, size):
    start_ind = 0
    end_ind = 0
    m_weights = []
    m_biases = []
    v_weights = []
    v_biases = []
    no_layers = len(size) - 1

    for i in range(size[1]):
        m_weights.append(m[i][:size[0]])
        v_weights.append(v[i][:size[0]])
        m_biases.append(m[i][size[0]:])
        v_biases.append(v[i][size[0]:])

        #m_weights = tf.concat([m_weights, m[i][:size[0]]], 0)
        #v_weights = tf.concat([v_weights, v[i][:size[0]]], 0)
        #m_biases = tf.concat([m_biases, m[i][size[0]:]], 0)
        #v_biases = tf.concat([v_biases, v[i][size[0]:]], 0)

    mw = tf.reshape(m_weights, [size[1], size[0]])
    vw = tf.reshape(v_weights, [size[1], size[0]])

    mw = tf.transpose(mw)
    vw = tf.transpose(vw)

    mb = tf.reshape(m_biases, [size[1]])
    vb = tf.reshape(v_biases, [size[1]])

    return mw, vw, mb, vb


def _unpack_weights(m, v, size):
    start_ind = 0
    end_ind = 0
    m_weights = []
    m_biases = []
    v_weights = []
    v_biases = []
    no_layers = len(size) - 1
    for i in range(no_layers):
        Din = size[i]
        Dout = size[i + 1]
        end_ind += Din * Dout
        m_weights.append(tf.reshape(m[start_ind:end_ind], [Din, Dout]))
        v_weights.append(tf.reshape(v[start_ind:end_ind], [Din, Dout]))
        start_ind = end_ind
        end_ind += Dout
        m_biases.append(m[start_ind:end_ind])
        v_biases.append(v[start_ind:end_ind])
        start_ind = end_ind
    return m_weights, v_weights, m_biases, v_biases


def _create_point_weights(size):
    no_layers = len(size) - 1
    no_weights = 0
    for i in range(no_layers):
        no_weights += size[i] * size[i + 1] + size[i + 1]

    m = tf.Variable(tf.constant(np.random.normal(size=[no_weights], scale=0.1),
                                dtype=tf.float32))
    return no_weights, m


def _unpack_point_weights(m, size):
    start_ind = 0
    end_ind = 0
    m_weights = []
    m_biases = []
    no_layers = len(size) - 1
    for i in range(no_layers):
        Din = size[i]
        Dout = size[i + 1]
        end_ind += Din * Dout
        m_weights.append(tf.reshape(m[start_ind:end_ind], [Din, Dout]))
        start_ind = end_ind
        end_ind += Dout
        m_biases.append(m[start_ind:end_ind])
        start_ind = end_ind
    return m_weights, m_biases


class MFVI_NN(object):
    def __init__(
            self, lower_size, upper_sizes,
            no_train_samples=10, no_test_samples=100, training_loss_classes=[], data_classes=[]):
        self.lower_size = lower_size
        self.no_tasks = len(training_loss_classes)
        self.data_classes = data_classes
        self.training_loss_classes = training_loss_classes
        self.upper_sizes = upper_sizes
        self.no_train_samples = no_train_samples
        self.no_test_samples = no_test_samples
        # input and output placeholders
        self.x = tf.placeholder(tf.float32, [None, lower_size[0]])
        self.ys = [
            tf.placeholder(tf.float32, [None, upper_size[-1]])
            for upper_size in upper_sizes]
        self.training_size = tf.placeholder(tf.int32)

        self.lower_net = HalfNet(lower_size)
        self.upper_nets = []
        for t, upper_size in enumerate(self.upper_sizes):
            ############# For singlehead fix upper weights, use UpperHalfNet
            #if fix_upper_weights:
            #    self.upper_nets.append(UpperHalfNet(upper_size))
            #else:
            #    self.upper_nets.append(HalfNet(upper_size))
            self.upper_nets.append(HalfNet(upper_size))

        self.training_loss = self._build_training_loss()
        self.costs = self._build_costs()
        self.preds = self._build_preds()
        self.preds_hist_plot = self._build_preds_hist_plot()


    # def _build_costs_new(self):
    #     kl_lower = self.lower_net.KL_term()
    #     kl_lower_prior = self.lower_net.KL_prior_term()
    #     costs = []
    #     N = tf.cast(self.training_size, tf.float32)
    #     for t, upper_net in enumerate(self.upper_nets):
    #         kl_upper = upper_net.KL_term()
    #         kl_upper_prior = upper_net.KL_prior_term()
    #         log_pred = self.log_prediction_fn(
    #             self.x, self.ys[t], t, self.no_train_samples)
    #         kl_prior = tf.div(kl_lower_prior + kl_upper_prior, N)
    #         kl_post = tf.div(kl_lower + kl_upper, N)
    #         cost = tf.div(kl_post, kl_prior) - log_pred
    #         costs.append(cost)
    #     return costs
    #
    def _build_training_loss(self):
        # Go over each task, and only append after adding the costs from the relevant classes in each task. This is then used in AdamOptimizer
        kl_lower = self.lower_net.KL_term()
        costs = []
        N = tf.cast(self.training_size, tf.float32)
        for task_id in range(self.no_tasks):
            log_pred = self.log_prediction_fn_training_loss(
                self.x, self.ys, self.training_loss_classes[task_id], self.no_train_samples)
            for ind, class_id in enumerate(self.training_loss_classes[task_id]):
                kl_upper = self.upper_nets[class_id].KL_term()
                if ind == 0:
                    kl_upper_total = kl_upper
                else:
                    kl_upper_total += kl_upper
            cost = tf.div(kl_lower + kl_upper, N) - log_pred
            costs.append(cost)
        return costs

    def _build_costs(self):
        kl_lower = self.lower_net.KL_term()
        costs = []
        N = tf.cast(self.training_size, tf.float32)
        for t, upper_net in enumerate(self.upper_nets):
            kl_upper = upper_net.KL_term()
            log_pred = self.log_prediction_fn(
                self.x, self.ys[t], t, self.no_train_samples)
            cost = tf.div(kl_lower + kl_upper, N) - log_pred
            costs.append(cost)
        return costs

    def _build_preds(self):
        preds = []
        for t, upper_net in enumerate(self.upper_nets):
            pred = self.prediction_fn(self.x, t, self.no_test_samples)
            preds.append(pred)
        return preds

    def _build_preds_hist_plot(self):
        preds = []
        for t, upper_net in enumerate(self.upper_nets):
            pred = self.prediction_fn_hist_plot(self.x, t, self.no_test_samples)
            preds.append(pred)
        return preds

    def prediction_fn(self, inputs, task_idx, no_samples):
        K = no_samples
        inputs_3d = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        lower_output = self.lower_net.prediction(inputs_3d, K)
        upper_output = self.upper_nets[task_idx].prediction(lower_output, K)
        return upper_output

    def prediction_fn_hist_plot(self, inputs, task_idx, no_samples):
        K = no_samples
        inputs_3d = tf.tile(tf.expand_dims(inputs, 0), [K, 1, 1])
        lower_output = self.lower_net.prediction(inputs_3d, K)
        upper_output = self.upper_nets[task_idx].prediction_hist_plot(lower_output, K)
        return upper_output

    def log_prediction_fn(self, inputs, targets, task_idx, no_samples):
        pred = self.prediction_fn(inputs, task_idx, no_samples)
        targets = tf.tile(tf.expand_dims(targets, 0), [self.no_train_samples, 1, 1])
        log_lik = - tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
        return log_lik

    def log_prediction_fn_training_loss(self, inputs, targets_input, training_loss_class_idx, no_samples):
        for ind, class_id in enumerate(training_loss_class_idx):
            pred_interm = self.prediction_fn(inputs, class_id, no_samples)
            targets_interm = tf.tile(tf.expand_dims(targets_input[class_id], 0), [self.no_train_samples, 1, 1])
            if ind == 0:
                pred = pred_interm
                targets = targets_interm
            else:
                pred = tf.concat([pred, pred_interm], axis=2)
                targets = tf.concat([targets, targets_interm], axis=2)
        log_lik = - tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
        return log_lik

    def init_session(self, task_idx, learning_rate, training_classes = []):

        #classes = [0]
        vars_to_optimise = [self.lower_net.m, self.lower_net.v]
        for class_id in training_classes:
            #print 'task', task_idx, 'class', class_id
            vars_to_optimise.append(self.upper_nets[class_id].m)
            vars_to_optimise.append(self.upper_nets[class_id].v)

        #vars_to_optimise = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #print 'variables to optimise:', vars_to_optimise

        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.training_loss[task_idx], var_list=vars_to_optimise)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # launch a session
        self.sess = tf.Session()
        self.sess.run(init)

    def close_session(self):
        self.sess.close()

    def train(self, x_train, y_train, class_idx, task_id, prior_lower, prior_upper,
              no_epochs=1000, batch_size=100, display_epoch=10, epoch_pause=[]):

        N = x_train.shape[0]
        if batch_size > N:
            batch_size = N
        sess = self.sess
        costs = []
        feed_dict = {
            self.lower_net.m0: prior_lower[0],
            self.lower_net.v0: prior_lower[1],
            self.training_size: N}

        for class_id in self.training_loss_classes[task_id]:
            feed_dict[self.upper_nets[class_id].m0] = prior_upper[class_id][0]
            feed_dict[self.upper_nets[class_id].v0] = prior_upper[class_id][1]

        # For visualising how weights change during training
        #epoch_pause = range(1,no_epochs,3)
        #epoch_pause = []
        lower_post_epoch = []
        upper_post_epoch = []

        # Training cycle
        for epoch in range(no_epochs):
            perm_inds = range(x_train.shape[0])
            np.random.shuffle(perm_inds)
            cur_x_train = x_train[perm_inds]
            cur_y_train = y_train[perm_inds]

            avg_cost = 0.
            total_batch = int(np.ceil(N * 1.0 / batch_size))
            # Loop over all batches
            for i in range(total_batch):
                start_ind = i * batch_size
                end_ind = np.min([(i + 1) * batch_size, N])
                batch_x = cur_x_train[start_ind:end_ind, :]
                batch_y = cur_y_train[start_ind:end_ind, :]
                feed_dict[self.x] = batch_x

                for class_id in self.training_loss_classes[task_id]:
                    batch_input = np.zeros([end_ind-start_ind, 1])
                    batch_input[:, 0] = batch_y[:, class_id]
                    feed_dict[self.ys[class_id]] = batch_input

                # Run optimization op (backprop) and cost op (to get loss value)
                _, c_total = sess.run(
                    [self.train_step, self.training_loss[task_id]],
                    feed_dict=feed_dict)

                # Compute average loss
                avg_cost += c_total / total_batch
                # print i, total_batch, c

            # Display logs per epoch step
            if epoch % display_epoch == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
            costs.append(avg_cost)

            if epoch in epoch_pause:
                lower_post_epoch1, upper_post_epoch1 = self.get_weights(class_idx)
                lower_post_epoch.append(lower_post_epoch1)
                upper_post_epoch.append(upper_post_epoch1)

        print("Optimisation Finished!")
        return costs, lower_post_epoch, upper_post_epoch

    def prediction(self, x_test, task_idx=0, batch_size=1000):
        # Test model
        N = x_test.shape[0]
        batch_size = N if batch_size > N else batch_size
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            #prediction = self.sess.run(
            #    [self.preds[task_idx]],
            #    feed_dict={self.x: batch_x})[0]

            prediction = self.sess.run(
                [self.preds],
                feed_dict={self.x: batch_x})[0]

            if i == 0:
                predictions = prediction
            else:
                predictions = np.concatenate((predictions, prediction), axis=2)

        return predictions

    def prediction_hist_plot(self, x_test, task_idx, batch_size=1000):
        # Test model
        N = x_test.shape[0]
        batch_size = N if batch_size > N else batch_size
        total_batch = int(np.ceil(N * 1.0 / batch_size))
        for i in range(total_batch):
            start_ind = i * batch_size
            end_ind = np.min([(i + 1) * batch_size, N])
            batch_x = x_test[start_ind:end_ind, :]
            prediction = self.sess.run(
                [self.preds_hist_plot[task_idx]],
                feed_dict={self.x: batch_x})[0]
            if i == 0:
                predictions = prediction
            else:
                predictions = np.concatenate((predictions, prediction), axis=1)
        return predictions

    def prediction_prob(self, x_test, task_idx, batch_size=1000):
        prob = self.sess.run(
            [tf.nn.softmax(self.prediction(x_test, task_idx, batch_size))],
            feed_dict={self.x: x_test})[0]
        return prob

    def get_weights(self, class_idx):
        lower = self.sess.run(self.lower_net.params)
        upper = []
        for class_id in class_idx:
            upper_interm = self.sess.run(self.upper_nets[class_id].params)
            upper.append(upper_interm)
        #res = self.sess.run(
        #    [self.lower_net.params, self.upper_nets[task_idx].params])
        return (lower, upper)

    def assign_weights(self, class_idx, lower_weights, upper_weights):
        lower_net = self.lower_net
        self.sess.run(
            [lower_net.assign_m_op, lower_net.assign_v_op],
            feed_dict={
                lower_net.new_m: lower_weights[0],
                lower_net.new_v: lower_weights[1]})

        if not isinstance(class_idx, (list,)):
            class_idx = [class_idx]
            #upper_weights = [upper_weights]

        for i, idx in enumerate(class_idx):
            upper_net = self.upper_nets[idx]

            assign_m_op = tf.assign(upper_net.m, upper_net.new_m)
            self.sess.run(
                assign_m_op, feed_dict={
                    upper_net.new_m: upper_weights[i][0]})

            assign_v_op = tf.assign(upper_net.v, upper_net.new_v)
            self.sess.run(
                assign_v_op, feed_dict={
                    upper_net.new_v: upper_weights[i][1]})

    def reset_optimiser(self):
        optimizer_scope = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            "scope/prefix/for/optimizer")
        self.sess.run(tf.initialize_variables(optimizer_scope))


class HalfNet():
    def __init__(self, size, act_func=tf.nn.relu):
        self.size = size
        self.no_layers = len(size) - 1
        self.act_func = act_func

        # creating weights
        self.no_weights, self.m, self.v = _create_weights(self.size)
        self.mw, self.vw, self.mb, self.vb = _unpack_weights(self.m, self.v, self.size)
        self.params = [self.m, self.v]

        self.new_m = tf.placeholder(tf.float32, [self.no_weights])
        self.new_v = tf.placeholder(tf.float32, [self.no_weights])
        self.assign_m_op = tf.assign(self.m, self.new_m)
        self.assign_v_op = tf.assign(self.v, self.new_v)

        # prior as place holder as these can change
        self.m0 = tf.placeholder(tf.float32, [self.no_weights])
        self.v0 = tf.placeholder(tf.float32, [self.no_weights])

    def prediction(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[1]
        Din = self.size[0]
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb
        act = inputs
        for i in range(self.no_layers):
            m_pre = tf.einsum('kni,io->kno', act, mw[i])
            m_pre = m_pre + mb[i]
            v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(vw[i]))
            v_pre = v_pre + tf.exp(vb[i])
            eps = tf.random_normal([K, 1, self.size[i + 1]], 0.0, 1.0, dtype=tf.float32)
            pre = eps * tf.sqrt(1e-9 + v_pre) + m_pre
            act = self.act_func(pre)
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    def prediction_hist_plot(self, inputs, no_samples):
        #neuron = [58,128,90,101,224]
        neuron = [77]
        #neuron = range(5)
        K = no_samples
        N = tf.shape(inputs)[1]
        Din = self.size[0]
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb
        act = inputs
        for i in range(self.no_layers):
            m_pre = 0
            v_pre = 0
            for j in range(np.size(neuron)):
                m_pre = m_pre + tf.einsum('kn,o->kno', act[:,:,neuron[j]], mw[i][neuron[j],:])
                #m_pre = m_pre + mb[i]
                v_pre = v_pre + tf.einsum('kn,o->kno', act[:,:,neuron[j]] ** 2.0, tf.exp(vw[i][neuron[j],:]))
                #v_pre = v_pre + tf.exp(vb[i])
            eps = tf.random_normal([K, N, self.size[i + 1]], 0.0, 1.0, dtype=tf.float32)
            pre = eps * tf.sqrt(1e-9 + v_pre) + m_pre
            act = self.act_func(pre)
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    def KL_term(self):
        const_term = -0.5 * self.no_weights
        log_std_diff = 0.5 * tf.reduce_sum(self.v0 - self.v)
        ## ignore log prior for now
        # log_std_diff = 0.5 * tf.reduce_sum(- self.v)
        mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(self.v) + (self.m0 - self.m) ** 2) / tf.exp(self.v0))
        kl = const_term + log_std_diff + mu_diff_term
        return kl

    # def KL_prior_term(self):
    #     v0 = 1.0
    #     m0 = 0.0
    #     const_term = -0.5 * self.no_weights
    #     log_std_diff = 0.5 * tf.reduce_sum(tf.log(v0) - self.v)
    #     ## ignore log prior for now
    #     # log_std_diff = 0.5 * tf.reduce_sum(- self.v)
    #     mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(self.v) + (m0 - self.m) ** 2) / v0)
    #     kl = const_term + log_std_diff + mu_diff_term
    #     return kl

class UpperHalfNet():
    def __init__(self, size, act_func=tf.nn.relu):
        self.size = size
        self.no_layers = len(size) - 1
        self.act_func = act_func

        # creating weights
        self.no_weights, self.m, self.v = _create_weights_fix_upper_weights(self.size)
        self.mw, self.vw, self.mb, self.vb = _unpack_weights_fix_upper_weights(self.m, self.v, self.size)
        self.params = [self.m, self.v]

        #self.new_m = tf.placeholder(tf.float32, [self.size[1], self.size[0]+1])
        #self.new_v = tf.placeholder(tf.float32, [self.size[1], self.size[0]+1])
        self.new_m = tf.placeholder(tf.float32, [10, 257])
        self.new_v = tf.placeholder(tf.float32, [10, 257])
        #self.assign_m_op = tf.assign(self.m, self.new_m)
        #self.assign_v_op = tf.assign(self.v, self.new_v)

        # prior as place holder as these can change
        self.m0 = tf.placeholder(tf.float32, [10, 257])
        self.v0 = tf.placeholder(tf.float32, [10, 257])

    def prediction(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[1]
        Din = self.size[0]
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb

        act = inputs
        for i in range(self.no_layers):
            m_pre = tf.einsum('kni,io->kno', act, mw)
            m_pre = m_pre + mb
            v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(vw))
            v_pre = v_pre + tf.exp(vb)
            eps = tf.random_normal([K, 1, self.size[i + 1]], 0.0, 1.0, dtype=tf.float32)
            pre = eps * tf.sqrt(1e-9 + v_pre) + m_pre
            act = self.act_func(pre)
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    def prediction_hist_plot(self, inputs, no_samples):
        K = no_samples
        N = tf.shape(inputs)[1]
        Din = self.size[0]
        Dout = self.size[-1]
        mw, vw, mb, vb = self.mw, self.vw, self.mb, self.vb

        act = inputs
        for i in range(self.no_layers):
            m_pre = tf.einsum('kni,io->kno', act, mw)
            m_pre = m_pre + mb
            v_pre = tf.einsum('kni,io->kno', act ** 2.0, tf.exp(vw))
            v_pre = v_pre + tf.exp(vb)
            eps = tf.random_normal([K, 1, self.size[i + 1]], 0.0, 1.0, dtype=tf.float32)
            pre = eps * tf.sqrt(1e-9 + v_pre) + m_pre
            act = self.act_func(pre)
        pre = tf.reshape(pre, [K, N, Dout])
        return pre

    def KL_term(self):

        m_kl = []
        v_kl = []
        for i in range(self.size[1]):
            m_kl = tf.concat([m_kl, self.m[i]], 0)
            v_kl = tf.concat([v_kl, self.v[i]], 0)

        m0_kl = []
        v0_kl = []
        for i in range(self.size[1]):
            m0_kl = tf.concat([m0_kl, self.m0[i]], 0)
            v0_kl = tf.concat([v0_kl, self.v0[i]], 0)

        const_term = -0.5 * self.no_weights
        log_std_diff = 0.5 * tf.reduce_sum(v0_kl - v_kl)
        ## ignore log prior for now
        # log_std_diff = 0.5 * tf.reduce_sum(- self.v)
        mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(v_kl) + (m0_kl - m_kl) ** 2) / tf.exp(v0_kl))
        kl = const_term + log_std_diff + mu_diff_term
        return kl

    # def KL_prior_term(self):
    #     v0 = 1.0
    #     m0 = 0.0
    #     const_term = -0.5 * self.no_weights
    #     log_std_diff = 0.5 * tf.reduce_sum(tf.log(v0) - self.v)
    #     ## ignore log prior for now
    #     # log_std_diff = 0.5 * tf.reduce_sum(- self.v)
    #     mu_diff_term = 0.5 * tf.reduce_sum((tf.exp(self.v) + (m0 - self.m) ** 2) / v0)
    #     kl = const_term + log_std_diff + mu_diff_term
    #     return kl