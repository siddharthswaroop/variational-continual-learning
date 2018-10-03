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

    update_test_weights = False
    no_tasks = 5
    plot_figures = True
    pruned_var = 0.000001

    #if path == "two_hidden_layers/pruned_nonzeromean/":
    #    m_prior = 0.2
    #    print 'nonzero prior'
    #else:
    #    m_prior = 0.0
    m_prior = 0.0


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
                KL_lower = KL_term(mean_neuron, var_neuron, m_prior=m_prior)/np.size(mean_neuron)
                KL.append(KL_lower)

                snr_lower = np.sum(np.sqrt(mean_neuron**2/var_neuron))/np.size(mean_neuron)
                snr.append(snr_lower)

            KL = np.array(KL)
            snr = np.array(snr)

            # Criterion for pruning
            KL_cutoff = None
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

            if KL_cutoff is not None and task_id == 4:
                print np.size(np.where(KL>=KL_cutoff))


            snr_output = []
            KL_output = []
            if layer > 0:
                for neuron_id in range(no_hiddens[layer-1]):
                    mean_neuron = m_low[neuron_id,:]
                    var_neuron = var_low[neuron_id,:]
                    snr_output_lower = np.sum(np.sqrt(mean_neuron**2/var_neuron))/np.size(mean_neuron)
                    KL_output_lower = KL_term(mean_neuron, var_neuron, m_prior=m_prior)/np.size(mean_neuron)
                    snr_output.append(snr_output_lower)
                    KL_output.append(KL_output_lower)
                snr_output = np.array(snr_output)
                KL_output = np.array(KL_output)
                if snr_upper_cutoff is not None and task_id == 4:
                    print np.size(np.where(snr_output > snr_upper_cutoff))


            if plot_figures and task_id == 4:
                plt.figure()
                plt.hist(KL, bins=500, histtype='stepfilled', label='layer %d' % (layer))
                #plt.suptitle('KL histogram after task %d' % (task_id+1))
                plt.ylim((0,10))
                plt.xlabel('Value of KL to prior')
                plt.ylabel('Frequency')
                plt.legend()
                plt.savefig(path + 'hist_task%d_layer%d_KL.png' % (task_id+1, layer), bbox_inches='tight')

                if layer > 0:
                    plt.figure()
                    plt.hist(snr_output, bins=500, histtype='stepfilled', label='layer %d' % (layer-1))
                    #plt.suptitle('SNR histogram after task %d' % (task_id + 1))
                    plt.legend()
                    plt.ylim((0,10))
                    plt.xlabel('SNR')
                    plt.ylabel('Frequency')
                    plt.savefig(path + 'hist_task%d_layer%d_SNR.png' % (task_id+1, layer), bbox_inches='tight')

                #if layer > 0:
                #    plt.figure(task_id + 6)
                #    plt.hist(KL_output, bins=500, range=(0, KL_output.max()), histtype='stepfilled', label='layer %d' % (layer))
                #    plt.suptitle('KL output histogram after task %d' % (task_id + 1))
                #    plt.legend()

            if update_test_weights:
                lower_post[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]] = m_low.reshape([-1])
                lower_post[1, no_params:no_params + (in_dim + 1) * no_hiddens[layer]] = np.log(var_low.reshape([-1]))

            no_params = no_params + (in_dim + 1) * no_hiddens[layer]


        # Upper layer
        no_params_upper = (no_hiddens[-1]+1) * 2
        m_upper = m_upper.reshape([no_hiddens[-1]+1, 2])
        var_upper = var_upper.reshape([no_hiddens[-1] + 1, 2])

        snr_upper = np.sqrt(m_upper[:no_hiddens[-1]]**2/var_upper[:no_hiddens[-1]])/no_hiddens[-1]
        snr_upper_diff = np.abs(np.sqrt(2)*(m_upper[:,0]-m_upper[:,1])/np.sqrt(var_upper[:,0] + var_upper[:,1]))/(no_hiddens[-1]+1)

        if plot_figures and task_id == 4:
            plt.figure()
            plt.hist(np.resize(snr_upper,[-1]), bins=500, histtype='stepfilled')
            #plt.suptitle('SNR histogram of upper weights after task %d' % (task_id + 1))
            plt.ylim((0,10))
            plt.xlabel('SNR')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(path + 'hist_task%d_output_SNR.png' % (task_id + 1), bbox_inches='tight')

            #plt.figure(task_id+16)
            #plt.hist(snr_upper_diff, bins=500, histtype='stepfilled')
            #plt.suptitle('SNR [diff] histogram of upper weights after task %d' % (task_id + 1))
            #plt.ylim((0, 10))

        #print np.where(snr_upper>snr_upper_cutoff)
        #print m_upper[np.where(snr_upper>snr_upper_cutoff)]
        #print var_upper[np.where(snr_upper>snr_upper_cutoff)]
        #print 'test'
        if snr_upper_cutoff is not None and task_id == 4:
            print np.where(snr_upper_diff > snr_upper_cutoff)

        if snr_upper_cutoff is not None:
            m_upper[np.where(snr_upper < snr_upper_cutoff), :] = 0
            var_upper[np.where(snr_upper < snr_upper_cutoff), :] = pruned_var

        if KL_cutoff is not None:
            m_upper[np.where(KL < KL_cutoff), :] = 0
            var_upper[np.where(KL < KL_cutoff), :] = pruned_var

        if update_test_weights:
            upper_post[0, :] = m_upper.reshape([-1])
            upper_post[1, :] = np.log(var_upper.reshape([-1]))


        if update_test_weights:
            np.savez(path + 'test/weights_%d.npz' % task_id, lower=lower_post, upper=upper_post)

    #if plot_figures:
    #    plt.show()



if __name__ == "__main__":
    print 'plot pruning'

    no_hiddens = [256,256]
    hist_weights_pruning(no_hiddens, path="two_hidden_layers/pruned/")