import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as mlab


def KL_term(m_target, var_target, m_prior = 0.0, var_prior = 1.0):
    const_term = -0.5 * np.size(m_target)
    log_std_diff = 0.5 * np.sum(np.log(var_prior) - np.log(var_target))
    mu_diff_term = 0.5 * np.sum((var_target + (m_prior - m_target) ** 2) / var_prior)
    kl = const_term + log_std_diff + mu_diff_term
    return kl

def KL_term_vector(m_target, var_target, m_prior = 0.0, var_prior = 1.0):
    #const_term = -0.5 * np.size(m_target)
    #log_std_diff = 0.5 * np.sum(np.log(var_prior) - np.log(var_target))
    #mu_diff_term = 0.5 * np.sum((var_target + (m_prior - m_target) ** 2) / var_prior)
    const_term = -0.5
    log_std_diff = 0.5 * (np.log(var_prior) - np.log(var_target))
    mu_diff_term = 0.5 * ((var_target + (m_prior - m_target) ** 2) / var_prior)
    kl = const_term + log_std_diff + mu_diff_term
    return kl

def hist_weights_pruning_one_layer(no_tasks=5, path = ""):


    #no_tasks=1
    for task_id in range(no_tasks):
        #task_id=1
        res = np.load(path + 'weights_%d.npz' % task_id)
        lower_post = res['lower']
        upper_post = res['upper']

        m_low = lower_post[0, :]
        var_low = np.exp(lower_post[1, :])

        m_upper = upper_post[0, :]
        var_upper = np.exp(upper_post[1, :])

        no_hiddens = [256]
        in_dim = 784
        no_params = (in_dim+1) * no_hiddens[0]
        means = m_low.reshape([in_dim+1, no_hiddens[0]]) # Include bias term
        vars = var_low.reshape([in_dim+1, no_hiddens[0]])
        KL = []
        snr = []
        for neuron_id in range(no_hiddens[0]):
            mean_neuron = means[:,neuron_id]
            var_neuron = vars[:,neuron_id]
            KL_lower = KL_term(mean_neuron, var_neuron)
            KL.append(KL_lower)

            snr_lower = np.sum(np.sqrt(mean_neuron**2/var_neuron))
            snr.append(snr_lower)

        KL = np.array(KL)
        snr = np.array(snr)

        #print 'snr', np.where(snr>200)
        #print 'KL', np.where(KL>100)
        #KL[np.where(KL<10)] = -1.0

        #plt.figure(task_id+1)
        #plt.hist(KL, bins=500, range=(20,KL.max()), histtype='stepfilled')
        #plt.suptitle('KL histogram after task %d' % (task_id+1))

        #plt.figure(task_id+6)
        #plt.hist(snr, bins=500, range=(180, snr.max()), histtype='stepfilled')
        #plt.suptitle('SNR histogram after task %d' % (task_id + 1))

        no_params_upper = (no_hiddens[0]+1) * 2
        m_upper = m_upper[:no_params_upper].reshape([no_hiddens[0]+1, 2])
        var_upper = var_upper[:no_params_upper].reshape([no_hiddens[0] + 1, 2])

        snr_upper = np.sqrt(m_upper[:no_hiddens[0]]**2/var_upper[:no_hiddens[0]])
        snr_upper_test = np.abs(np.sqrt(2)*(m_upper[:no_hiddens[0],0]-m_upper[:no_hiddens[0],1])/np.sqrt(var_upper[:no_hiddens[0],0] + var_upper[:no_hiddens[0],1]))

        plt.figure(task_id+1)
        plt.hist(snr_upper, bins=500, histtype='stepfilled')
        plt.suptitle('SNR histogram of upper weights after task %d' % (task_id + 1))

        plt.figure(task_id+6)
        plt.hist(snr_upper_test, bins=500, histtype='stepfilled')
        plt.suptitle('SNR [diff] histogram of upper weights after task %d' % (task_id + 1))

        print np.where(snr_upper>25)
        print m_upper[np.where(snr_upper>25)]
        print var_upper[np.where(snr_upper>25)]
        print 'test'
        print np.where(snr_upper_test>25)
        #means[:,np.where(KL<10)] = 0
        #vars[:,np.where(KL<10)] = 0.000001
        #m_upper[np.where(KL<10), :] = 0
        #var_upper[np.where(KL<10), :] = 0.000001


        lower_post[0, :] = means.reshape([no_params])
        lower_post[1, :] = np.log(vars.reshape([no_params]))
        upper_post[0, :] = m_upper.reshape([no_params_upper])
        upper_post[1, :] = np.log(var_upper.reshape([no_params_upper]))
        np.savez(path + 'test/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)
    plt.show()

def hist_weights_pruning(no_hiddens = [256], path = ""):

    update_test_weights = True
    no_tasks=5
    plot_figures = False
    pruned_var = 1

    pruned_units = []
    pruned_units_prev = []
    for task_id in range(no_tasks):
        #task_id=1

        res = np.load(path + 'weights_%d.npz' % task_id)
        lower_post = res['lower']
        upper_post = res['upper']
        m_upper = upper_post[0, :]
        var_upper = np.exp(upper_post[1, :])

        # Lower network
        for layer in range(len(no_hiddens)):

            if layer == 0:
                in_dim = 784
                no_params = 0
            else:
                in_dim = no_hiddens[layer - 1]

            m_low = lower_post[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
            var_low = np.exp(lower_post[1, no_params:no_params + (in_dim + 1) * no_hiddens[layer]])
            m_low = m_low.reshape([in_dim + 1, no_hiddens[layer]])
            var_low = var_low.reshape([in_dim + 1, no_hiddens[layer]])

            # For setting previous layer's pruned units' output weights later
            if layer > 0:
                pruned_units_prev = pruned_units

            # Calculate sum of KL and snr over input weights to each neuron
            KL = []
            snr = []
            for neuron_id in range(no_hiddens[layer]):
                mean_neuron = m_low[:,neuron_id]
                var_neuron = var_low[:,neuron_id]
                KL_lower = KL_term(mean_neuron, var_neuron)
                KL.append(KL_lower)

                snr_lower = np.sum(np.sqrt(mean_neuron**2/var_neuron))
                snr.append(snr_lower)

            KL = np.array(KL)
            snr = np.array(snr)

            # Criterion for pruning
            KL_cutoff = 40
            snr_cutoff = None
            snr_upper_cutoff = None
            if KL_cutoff is not None:
                pruned_units = np.where(KL < KL_cutoff)
            elif snr_cutoff is not None:
                pruned_units = np.where(snr < snr_cutoff)

            # Pruned units' input weights
            m_low[:, pruned_units] = 0
            var_low[:, pruned_units] = pruned_var

            # Pruned units' output weights
            if layer > 0:
                m_low[pruned_units_prev, :] = 0
                var_low[pruned_units_prev, :] = pruned_var

            if task_id == 4:
                print np.size(np.where(KL>=KL_cutoff))

            if plot_figures:
                plt.figure(task_id+1)
                plt.hist(KL, bins=500, range=(0,KL.max()), histtype='stepfilled', label='layer %d' % (layer))
                plt.suptitle('KL histogram after task %d' % (task_id+1))
                plt.legend()

                plt.figure(task_id+6)
                plt.hist(snr, bins=500, range=(0, snr.max()), histtype='stepfilled', label='layer %d' % (layer))
                plt.suptitle('SNR histogram after task %d' % (task_id + 1))
                plt.legend()

            if update_test_weights:
                lower_post[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]] = m_low.reshape([-1])
                lower_post[1, no_params:no_params + (in_dim + 1) * no_hiddens[layer]] = np.log(var_low.reshape([-1]))

            no_params = no_params + (in_dim + 1) * no_hiddens[layer]


        # Upper layer
        no_params_upper = (no_hiddens[-1]+1) * 2
        m_upper = m_upper.reshape([no_hiddens[-1]+1, 2])
        var_upper = var_upper.reshape([no_hiddens[-1] + 1, 2])

        snr_upper = np.sqrt(m_upper[:no_hiddens[-1]]**2/var_upper[:no_hiddens[-1]])
        snr_upper_test = np.abs(np.sqrt(2)*(m_upper[:no_hiddens[-1],0]-m_upper[:no_hiddens[-1],1])/np.sqrt(var_upper[:no_hiddens[-1],0] + var_upper[:no_hiddens[-1],1]))

        if plot_figures:
            plt.figure(task_id+11)
            plt.hist(snr_upper, bins=500, histtype='stepfilled')
            plt.suptitle('SNR histogram of upper weights after task %d' % (task_id + 1))

            plt.figure(task_id+16)
            plt.hist(snr_upper_test, bins=500, histtype='stepfilled')
            plt.suptitle('SNR [diff] histogram of upper weights after task %d' % (task_id + 1))

        #print np.where(snr_upper>25)
        #print m_upper[np.where(snr_upper>25)]
        #print var_upper[np.where(snr_upper>25)]
        #print 'test'
        #print np.where(snr_upper_test>25)

        if KL_cutoff is not None:
            m_upper[np.where(KL < KL_cutoff), :] = 0
            var_upper[np.where(KL < KL_cutoff), :] = pruned_var

        if update_test_weights:
            upper_post[0, :] = m_upper.reshape([-1])
            upper_post[1, :] = np.log(var_upper.reshape([-1]))


        if update_test_weights:
            np.savez(path + 'test/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)

    if plot_figures:
        plt.show()



if __name__ == "__main__":
    print 'plot pruning'

    no_hiddens = [256,256]
    hist_weights_pruning(no_hiddens, path="two_hidden_layers/pruned/")