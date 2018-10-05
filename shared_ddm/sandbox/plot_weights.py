import matplotlib

#matplotlib.use('Agg')
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import pdb


def visualise_weights_ml(no_hiddens=256, path=""):
    res_0 = np.load(path + 'weights_ml_batch.npz')
    lower_0 = res_0['lower']
    m0 = lower_0[0, :]

    upper_0 = res_0['upper']
    upper0 = upper_0[0]
    upper1 = upper_0[1]
    upper2 = upper_0[2]

    task0 = []
    task1 = []
    task2 = []

    for i in range(10):
        task0.append(np.argmax(np.abs(upper0)))
        task0.append(np.max(np.abs(upper0)))
        upper0[0,np.argmax(np.abs(upper0))] = 0
        task1.append(np.argmax(np.abs(upper1)))
        task1.append(np.max(np.abs(upper1)))
        upper1[0,np.argmax(np.abs(upper1))] = 0
        task2.append(np.argmax(np.abs(upper2)))
        task2.append(np.max(np.abs(upper2)))
        upper2[0,np.argmax(np.abs(upper2))] = 0

    print np.reshape(task0, (-1,2))
    print np.reshape(task1, (-1,2))
    print np.reshape(task2, (-1,2))


    #no_hiddens = 100
    in_dim = 784
    in_size = [28, 28]
    no_params = in_dim * no_hiddens
    m0 = m0[:no_params].reshape([in_dim, no_hiddens])
    m0min, m0max = np.min(m0), np.max(m0)

    no_cols = int(np.sqrt(no_hiddens))
    no_rows = int(np.sqrt(no_hiddens))
    print "creating figures ..."
    fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))

    fig0.suptitle("means, min = %f, max = %f" % (np.min(np.absolute(m0)), np.max(np.absolute(m0))))

    for i in range(no_rows):
        for j in range(no_cols):
            #print i, j
            k = i * no_cols + j
            ma = m0[:, k].reshape(in_size)

            axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m0min, vmax=m0max)
            axs0[i, j].set_xticks(np.array([]))
            axs0[i, j].set_yticks(np.array([]))

    plt.show()

    #fig0.savefig('/tmp/lower_mean_1.pdf')


def visualise_weights_vi_batch(no_hiddens=256, path=""):
    res_0 = np.load(path + 'weights_vi_batch.npz')
    lower_0 = res_0['lower']
    m0 = lower_0[0, :]
    v0 = np.exp(lower_0[1, :])

    upper_0 = res_0['upper']
    m1 = upper_0[0][0,:]
    m2 = upper_0[1][0,:]
    m3 = upper_0[2][0,:]
    v1 = np.exp(upper_0[0][1,:])
    v2 = np.exp(upper_0[1][1,:])
    v3 = np.exp(upper_0[2][1,:])

    #no_hiddens = 100
    in_dim = 784
    in_size = [28, 28]
    no_params = in_dim * no_hiddens
    m0 = m0[:no_params].reshape([in_dim, no_hiddens])
    v0 = v0[:no_params].reshape([in_dim, no_hiddens])
    m0min, m0max = np.min(m0), np.max(m0)
    v0min, v0max = np.min(v0), np.max(v0)

    no_params = no_hiddens * 2
    m1 = m1[:no_params].reshape([no_hiddens, 2])
    v1 = v1[:no_params].reshape([no_hiddens, 2])
    m2 = m2[:no_params].reshape([no_hiddens, 2])
    v2 = v2[:no_params].reshape([no_hiddens, 2])
    m3 = m3[:no_params].reshape([no_hiddens, 2])
    v3 = v3[:no_params].reshape([no_hiddens, 2])


    #test = m3
    #print test
    #for i in range(5):
        #print np.argmax(np.abs(test),0)
        #print test[np.argmax(np.abs(test),0)]
        #test[np.argmax(np.abs(test), 0)] = [[0, 0],[0, 0]]





    no_cols = int(np.sqrt(no_hiddens))
    no_rows = int(np.sqrt(no_hiddens))
    print "creating figures ..."
    fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(10, 10))

    fig0.suptitle("means, min = %f, max = %f" % (np.min(np.absolute(m0)), np.max(np.absolute(m0))))
    fig1.suptitle("variances, min = %f, max = %f" % (np.min(np.absolute(v0)), np.max(np.absolute(v0))))

    for i in range(no_rows):
        for j in range(no_cols):
            #print i, j
            k = i * no_cols + j
            ma = m0[:, k].reshape(in_size)
            va = v0[:, k].reshape(in_size)

            axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m0min, vmax=m0max)
            axs0[i, j].set_xticks(np.array([]))
            axs0[i, j].set_yticks(np.array([]))

            axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v0min, vmax=v0max)
            axs1[i, j].set_xticks(np.array([]))
            axs1[i, j].set_yticks(np.array([]))

    plt.show()

    #fig0.savefig('/tmp/lower_mean_1.pdf')



def visualise_layer_weights(no_hiddens=[256], path="", single_head=False):

    # Plot difference of mean and variance for lower level weights
    plot_diff = False

    # Plot lower / upper level figures
    plot_lower = True
    plot_upper = True


    res = np.load(path + 'weights_%d.npz' % 0)
    mnist_digits = res['MNISTdigits']
    classes = res['classes']
    class_index_conversion = res['class_index_conversion']
    no_tasks = len(classes)
    task_range = range(no_tasks)
    #task_range = [0, 1, 2]

    for task_id in task_range:

        res = np.load(path + 'weights_%d.npz' % task_id)
        lower = res['lower']
        upper = res['upper']
        m_upper = upper[:,0]
        var_upper = np.exp(upper[:,1])

        # Lower network
        for layer in range(len(no_hiddens)):

            if layer == 0:
                in_dim = 784
                in_size = [28, 28]
                no_rows = 28
                no_cols = 28
                no_params = 0
            else:
                in_dim = no_hiddens[layer-1]

            if plot_diff:
                if task_id == 0:
                    m_low_old = lower[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
                    m_low = m_low_old
                    var_low_old = np.exp(lower[1, no_params:no_params+(in_dim+1)*no_hiddens[layer]])
                    var_low = var_low_old
                else:
                    m_low = lower[0, no_params:no_params+(in_dim+1)*no_hiddens[layer]] - m_low_old
                    var_low = np.exp(lower[1, no_params:no_params+(in_dim+1)*no_hiddens[layer]]) - var_low_old
                    m_low_old = lower[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
                    var_low_old = np.exp(lower[1, no_params:no_params + (in_dim + 1) * no_hiddens[layer]])
            else:
                m_low = lower[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
                var_low = np.exp(lower[1, no_params:no_params+(in_dim+1)*no_hiddens[layer]])

            no_params = no_params + (in_dim + 1) * no_hiddens[layer]

            m_low = m_low.reshape([in_dim+1, no_hiddens[layer]])
            var_low = var_low.reshape([in_dim+1, no_hiddens[layer]])
            m_min, m_max = np.min(m_low), np.max(m_low)
            v_min, v_max = np.min(var_low), np.max(var_low)

            shape_dim = [no_rows, no_cols]
            no_cols = int(np.sqrt(no_hiddens[layer]))
            no_rows = int(no_hiddens[layer]/no_cols)

            no_cols = 32
            no_rows = 8

            #neurons_interest = [114, 8, 122, 1]
            #print task_id, m_low[-1,neurons_interest], var_low[-1,neurons_interest]

            if plot_lower:
                print "creating lower figures ..."

                fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(20, 5))
                fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(20, 5))

                fig0.suptitle("Task %d, Layer %d, Mean, min = %f, max = %f" % (task_id+1, layer, np.min(np.absolute(m_low)), np.max(np.absolute(m_low))))
                fig1.suptitle("Task %d, Layer %d, Variance, min = %f, max = %f" % (task_id+1, layer, np.min(np.absolute(var_low)), np.max(np.absolute(var_low))))

                fig0.suptitle("Min = %f, Max = %f" % (np.min(np.absolute(m_low)), np.max(np.absolute(m_low))))
                fig1.suptitle("Min = %f, Max = %f" % (np.min(np.absolute(var_low)), np.max(np.absolute(var_low))))

                for i in range(no_rows):
                    for j in range(no_cols):
                        k = i * no_cols + j
                        ma = m_low[:in_dim, k].reshape(shape_dim)
                        va = var_low[:in_dim, k].reshape(shape_dim)

                        axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m_min, vmax=m_max)
                        axs0[i, j].set_xticks(np.array([]))
                        axs0[i, j].set_yticks(np.array([]))

                        axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v_min, vmax=v_max)
                        axs1[i, j].set_xticks(np.array([]))
                        axs1[i, j].set_yticks(np.array([]))

                fig0.savefig(path + 'task%d_layer%d_mean.png' % (task_id+1, layer), bbox_inches='tight')
                fig1.savefig(path + 'task%d_layer%d_var.png' % (task_id+1, layer), bbox_inches='tight')


        # Upper weights
        no_digits = len(m_upper)
        #no_digits = 10
        no_params = no_hiddens[-1] * len(mnist_digits)

        # POST fix_upper_weights
        m_upper = np.transpose(m_upper)
        var_upper = np.transpose(var_upper)

        x_max = np.max(np.abs(m_upper[:no_hiddens[-1]]) + np.sqrt(var_upper[:no_hiddens[-1]]))

        #no_cols = int(np.sqrt(no_hiddens[-1]))
        #no_rows = int(no_hiddens[-1]/no_cols)

        no_cols = 32
        no_rows = 8

        x = np.linspace(-x_max, x_max, 1000)

        if plot_upper:
            print  "creating upper figures ..."

            fig, axs = plt.subplots(no_rows, no_cols, figsize=(20, 5))
            fig.suptitle("Upper level weights for task %d (after task %d), min = %f, max = %f" % (
            task_id + 1, task_id + 1, -x_max, x_max))
            fig.suptitle("Min = %f, Max = %f" % (-x_max, x_max))

            no_plot_digits = (task_id+1)*2 if single_head else 2
            no_plot_digits = no_digits
            for i in range(no_rows):
                for j in range(no_cols):
                    k = i * no_cols + j
                    for digit in range(no_plot_digits):
                        axs[i, j].plot(x, mlab.normpdf(x, m_upper[k][digit], np.sqrt(var_upper[k][digit])), label='%d' % class_index_conversion[digit])
                    axs[i, j].set_xticks(np.array([]))
                    axs[i, j].set_yticks(np.array([]))
                    axs[i, j].set_ylim([0, 2.0])
            axs[i,j].legend()
            fig.savefig(path + 'task%d_upper.png' % (task_id+1), bbox_inches='tight')

            # Plot bias of upper weights
            x_bias_max = np.max(np.abs(m_upper[-1])) + np.sqrt(np.max(var_upper[-1]))*2
            #x_bias_max = 5.0

            no_cols = int(np.sqrt(no_hiddens[-1]))
            no_rows = int(no_hiddens[-1]/no_cols)

            x_bias = np.linspace(-x_bias_max, x_bias_max, 1000)
            plt.figure()
            for digit in range(no_plot_digits):
                plt.plot(x_bias, mlab.normpdf(x_bias, m_upper[-1][digit], np.sqrt(var_upper[-1][digit])), label='%d' % class_index_conversion[digit])
                plt.ylim([0, 3.0])
            plt.legend()
            plt.suptitle("Upper level bias after task %d, min = %f, max = %f" % (task_id+1, -x_bias_max, x_bias_max))
            plt.savefig(path + 'task%d_upper_bias.png' % (task_id+1))



def visualise_neuron_weights(no_hiddens=[256], path="", single_head=False):

    # Plot difference of mean and variance for lower level weights
    plot_diff = False
    # Plot lower / upper level figures
    plot_lower = True
    plot_upper = True
    no_tasks = 1
    task_range = range(no_tasks)
    #task_range = [0, 1, 2]

    # Stores weights
    means = []
    vars = []

    for task_id in task_range:

        res = np.load(path + 'weights_%d.npz' % task_id)
        lower = res['lower']
        upper = res['upper']
        m_upper = upper[0,:]
        var_upper = np.exp(upper[1,:])


        # Lower weights
        for layer in range(len(no_hiddens)):

            if layer == 0:
                in_dim = 784
                no_params = 0
            else:
                in_dim = no_hiddens[layer-1]


            m_low = lower[0, no_params:no_params + (in_dim + 1) * no_hiddens[layer]]
            var_low = np.exp(lower[1, no_params:no_params+(in_dim+1)*no_hiddens[layer]])

            no_params = no_params + (in_dim + 1) * no_hiddens[layer]

            m_low = m_low.reshape([in_dim+1, no_hiddens[layer]])
            var_low = var_low.reshape([in_dim+1, no_hiddens[layer]])

            means.append(m_low)
            vars.append(var_low)


        # Upper weights
        no_digits = no_tasks*2 if single_head else 2
        no_digits = 10
        no_params = no_hiddens[-1] * 2
        m_upper = m_upper.reshape([no_hiddens[-1]+1, no_digits])
        var_upper = var_upper.reshape([no_hiddens[-1]+1, no_digits])

        means.append(m_upper)
        vars.append(var_upper)

        means = np.array(means)
        vars = np.array(vars)

        # Plot weights
        for layer in range(len(no_hiddens)):

            x_low_max = np.max(np.abs(means[layer]) + vars[layer])
            x_low = np.linspace(-x_low_max, x_low_max, 200)
            x_up_max = np.max(np.abs(means[layer+1]) + vars[layer+1])
            x_up = np.linspace(-x_low_max, x_low_max, 200)

            print layer, x_low_max, x_up_max

            for i in range(np.size(means[layer],1)):
                fig, axs = plt.subplots(2, 1)
                lower_i_mean = means[layer][:, i]
                lower_i_var = vars[layer][:, i]
                upper_i_mean = means[layer+1][i, :]
                upper_i_var = vars[layer+1][i, :]
                for k in range(np.size(means[layer],0)-1):
                    axs[1].plot(x_low, mlab.normpdf(x_low, lower_i_mean[k], np.sqrt(lower_i_var[k])))
                for k in range(np.size(means[layer+1],1)):
                    axs[0].plot(x_up, mlab.normpdf(x_up, upper_i_mean[k], np.sqrt(upper_i_var[k])))

                plt.savefig(path + '/neurons/layer_%d/neuron_%d.png' % (layer, i))




    """
    res_0 = np.load(path + 'weights_0.npz')
    lower_0 = res_0['lower']
    m0 = lower_0[0, :]
    v0 = np.exp(lower_0[1, :])

    res_1 = np.load(path + 'weights_1.npz')
    lower_1 = res_1['lower']
    m1 = lower_1[0, :]
    v1 = np.exp(lower_1[1, :])

    res_2 = np.load(path + 'weights_2.npz')
    lower_2 = res_2['lower']
    m2 = lower_2[0, :]
    v2 = np.exp(lower_2[1, :])

    #no_hiddens = 100
    in_dim = 784
    in_size = [28, 28]
    no_params = in_dim * no_hiddens
    m0 = m0[:no_params].reshape([in_dim, no_hiddens])
    v0 = v0[:no_params].reshape([in_dim, no_hiddens])
    m1 = m1[:no_params].reshape([in_dim, no_hiddens])
    v1 = v1[:no_params].reshape([in_dim, no_hiddens])
    m2 = m2[:no_params].reshape([in_dim, no_hiddens])
    v2 = v2[:no_params].reshape([in_dim, no_hiddens])
    m0min, m0max = np.min(m0), np.max(m0)
    m1min, m1max = np.min(m1), np.max(m1)
    v0min, v0max = np.min(v0), np.max(v0)
    v1min, v1max = np.min(v1), np.max(v1)
    m2min, m2max = np.min(m2), np.max(m2)
    v2min, v2max = np.min(v2), np.max(v2)

    no_cols = int(np.sqrt(no_hiddens))
    no_rows = int(np.sqrt(no_hiddens))
    print "creating figures ..."
    fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig2, axs2 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig3, axs3 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig4, axs4 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig5, axs5 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    # fig6, axs6 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    # fig7, axs7 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
    fig0.suptitle("mean after task 1, min = %f, max = %f" % (np.min(np.absolute(m0)), np.max(np.absolute(m0))))
    fig1.suptitle("variance after task 1, min = %f, max = %f" % (np.min(np.absolute(v0)), np.max(np.absolute(v0))))
    fig2.suptitle("mean after task 2, min = %f, max = %f" % (np.min(np.absolute(m1)), np.max(np.absolute(m1))))
    fig3.suptitle("variance after task 2, min = %f, max = %f" % (np.min(np.absolute(v1)), np.max(np.absolute(v1))))
    fig4.suptitle("mean after task 3, min = %f, max = %f" % (np.min(np.absolute(m2)), np.max(np.absolute(m2))))
    fig5.suptitle("variance after task 3, min = %f, max = %f" % (np.min(np.absolute(v2)), np.max(np.absolute(v2))))
    # fig4.suptitle("mean diff")
    # fig5.suptitle("variance diff")
    # fig6.suptitle("snr task 1")
    # fig7.suptitle("snr task 2")
    for i in range(no_rows):
        for j in range(no_cols):
            #print i, j
            k = i * no_cols + j
            ma = m0[:, k].reshape(in_size)
            va = v0[:, k].reshape(in_size)
            mb = m1[:, k].reshape(in_size)
            vb = v1[:, k].reshape(in_size)
            mc = m2[:, k].reshape(in_size)
            vc = v2[:, k].reshape(in_size)

            axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m0min, vmax=m0max)
            axs0[i, j].set_xticks(np.array([]))
            axs0[i, j].set_yticks(np.array([]))

            axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v0min, vmax=v0max)
            axs1[i, j].set_xticks(np.array([]))
            axs1[i, j].set_yticks(np.array([]))

            axs2[i, j].matshow(mb, cmap=matplotlib.cm.binary, vmin=m1min, vmax=m1max)
            axs2[i, j].set_xticks(np.array([]))
            axs2[i, j].set_yticks(np.array([]))

            axs3[i, j].matshow(vb, cmap=matplotlib.cm.binary, vmin=v1min, vmax=v1max)
            axs3[i, j].set_xticks(np.array([]))
            axs3[i, j].set_yticks(np.array([]))

            axs4[i, j].matshow(mc, cmap=matplotlib.cm.binary, vmin=m2min, vmax=m2max)
            axs4[i, j].set_xticks(np.array([]))
            axs4[i, j].set_yticks(np.array([]))

            axs5[i, j].matshow(vc, cmap=matplotlib.cm.binary, vmin=v2min, vmax=v2max)
            axs5[i, j].set_xticks(np.array([]))
            axs5[i, j].set_yticks(np.array([]))

    # axs4[i, j].matshow(ma - mb, cmap=matplotlib.cm.binary)
    # axs4[i, j].set_xticks(np.array([]))
    # axs4[i, j].set_yticks(np.array([]))

    # axs5[i, j].matshow(va - vb, cmap=matplotlib.cm.binary)
    # axs5[i, j].set_xticks(np.array([]))
    # axs5[i, j].set_yticks(np.array([]))

    # axs6[i, j].matshow(ma**2 / va, cmap=matplotlib.cm.binary)
    # axs6[i, j].set_xticks(np.array([]))
    # axs6[i, j].set_yticks(np.array([]))

    # axs7[i, j].matshow(mb**2 / vb, cmap=matplotlib.cm.binary)
    # axs7[i, j].set_xticks(np.array([]))
    # axs7[i, j].set_yticks(np.array([]))
    """

    #plt.show()
    # pdb.set_trace()

    #fig0.savefig('/tmp/lower_mean_1.pdf')
    #fig1.savefig('/tmp/lower_var_1.pdf')
    #fig2.savefig('/tmp/lower_mean_2.pdf')
    #fig3.savefig('/tmp/lower_var_2.pdf')


def visualise_weights_epoch(no_hiddens=256, epoch_pause = [20, 40, 100, 120], path=""):
    res_0 = np.load(path + 'weights_0_epoch.npz')
    lower_0 = res_0['lower']

    res_1 = np.load(path + 'weights_1_epoch.npz')
    lower_1 = res_1['lower']

    m0 = []
    v0 = []
    m1 = []
    v1 = []

    for i in range(len(lower_0)):
        # m0 = lower_0[i, 0, :]
        v0 = np.exp(lower_0[i, 1, :])

        # m1 = lower_1[i, 0, :]
        v1 = np.exp(lower_1[i, 1, :])


        in_dim = 784
        in_size = [28, 28]
        no_params = in_dim * no_hiddens
        # m0 = m0[:no_params].reshape([in_dim, no_hiddens])
        v0 = v0[:no_params].reshape([in_dim, no_hiddens])
        # m1 = m1[:no_params].reshape([in_dim, no_hiddens])
        v1 = v1[:no_params].reshape([in_dim, no_hiddens])
        # m0min, m0max = np.min(m0), np.max(m0)
        # m1min, m1max = np.min(m1), np.max(m1)
        v0min, v0max = np.min(v0), np.max(v0)
        v1min, v1max = np.min(v1), np.max(v1)

        no_cols = int(np.sqrt(no_hiddens))
        no_rows = int(np.sqrt(no_hiddens))
        print "creating figures ..."
        plt.figure(i)
        fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
        fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
        fig0.suptitle("mean after task 1 and epoch %f, min = %f, max = %f" % (epoch_pause[i], np.min(np.absolute(v0)), np.max(np.absolute(v0))))
        fig1.suptitle("mean after task 2 and epoch %f, min = %f, max = %f" % (epoch_pause[i], np.min(np.absolute(v1)), np.max(np.absolute(v1))))
        # fig4.suptitle("mean diff")
        # fig5.suptitle("variance diff")
        # fig6.suptitle("snr task 1")
        # fig7.suptitle("snr task 2")
        for i in range(no_rows):
            for j in range(no_cols):
                #print i, j
                k = i * no_cols + j
                va = v0[:, k].reshape(in_size)
                vb = v1[:, k].reshape(in_size)

                axs0[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v0min, vmax=v0max)
                axs0[i, j].set_xticks(np.array([]))
                axs0[i, j].set_yticks(np.array([]))

                axs1[i, j].matshow(vb, cmap=matplotlib.cm.binary, vmin=v1min, vmax=v1max)
                axs1[i, j].set_xticks(np.array([]))
                axs1[i, j].set_yticks(np.array([]))

    # axs4[i, j].matshow(ma - mb, cmap=matplotlib.cm.binary)
    # axs4[i, j].set_xticks(np.array([]))
    # axs4[i, j].set_yticks(np.array([]))

    # axs5[i, j].matshow(va - vb, cmap=matplotlib.cm.binary)
    # axs5[i, j].set_xticks(np.array([]))
    # axs5[i, j].set_yticks(np.array([]))

    # axs6[i, j].matshow(ma**2 / va, cmap=matplotlib.cm.binary)
    # axs6[i, j].set_xticks(np.array([]))
    # axs6[i, j].set_yticks(np.array([]))

    # axs7[i, j].matshow(mb**2 / vb, cmap=matplotlib.cm.binary)
    # axs7[i, j].set_xticks(np.array([]))
    # axs7[i, j].set_yticks(np.array([]))

    plt.show()
    pdb.set_trace()

    #fig0.savefig('/tmp/lower_mean_1.pdf')
    #fig1.savefig('/tmp/lower_var_1.pdf')
    #fig2.savefig('/tmp/lower_mean_2.pdf')
    #fig3.savefig('/tmp/lower_var_2.pdf')


def check_weight_pruning(no_hiddens=256, path=""):
    res_0 = np.load(path + 'weights_0.npz')
    lower_0 = res_0['lower']
    upper_0 = res_0['upper']

    res_1 = np.load(path + 'weights_1.npz')
    lower_1 = res_1['lower']
    upper_1 = res_1['upper']

    m0 = lower_0[0, :]
    v0 = np.exp(lower_0[1, :])

    m1 = lower_1[0, :]
    v1 = np.exp(lower_1[1, :])

    m2 = upper_0[0, :]
    v2 = np.exp(upper_0[1, :])

    m3 = upper_1[0, :]
    v3 = np.exp(upper_1[1, :])

    in_dim = 784
    in_size = [28, 28]
    no_params = in_dim * no_hiddens
    m0 = m0[:no_params].reshape([in_dim, no_hiddens])
    v0 = v0[:no_params].reshape([in_dim, no_hiddens])
    m1 = m1[:no_params].reshape([in_dim, no_hiddens])
    v1 = v1[:no_params].reshape([in_dim, no_hiddens])

    no_params = no_hiddens * 2
    m2 = m2[:no_params].reshape([no_hiddens, 2])
    v2 = v2[:no_params].reshape([no_hiddens, 2])
    m3 = m3[:no_params].reshape([no_hiddens, 2])
    v3 = v3[:no_params].reshape([no_hiddens, 2])


    print np.argmax(np.abs(m3),0)
    print m3[np.argmax(np.abs(m3),0)]
    print m2[np.argmax(np.abs(m3),0)]

    """
    print np.max(m2)
    print np.min(m2)
    print np.max(m3)
    print np.min(m3)
    print np.max(v2)
    print np.min(v2)
    print np.max(v3)
    print np.min(v3)
    print v2[54:57]
    print v2[62:65]


    # task 1
    x = np.linspace(-2, 2, 100)
    lims = 10
    for i in range(lims):
        print i
        fig, axs = plt.subplots(2, 1)
        lower_i_mean = m0[:, i]
        lower_i_var = v0[:, i]
        upper_i_mean = m2[i, :]
        upper_i_var = v2[i, :]
        for k in range(in_dim):
            axs[1].plot(x, mlab.normpdf(x, lower_i_mean[k], np.sqrt(lower_i_var[k])))
        for k in range(2):
            axs[0].plot(x, mlab.normpdf(x, upper_i_mean[k], np.sqrt(upper_i_var[k])))

        plt.savefig('/tmp/task_1_unit_%d.pdf' % i)

    x = np.linspace(-2, 2, 100)
    for i in range(lims):
        print i
        fig, axs = plt.subplots(2, 1)
        lower_i_mean = m1[:, i]
        lower_i_var = v1[:, i]
        upper_i_mean = m3[i, :]
        upper_i_var = v3[i, :]
        for k in range(in_dim):
            axs[1].plot(x, mlab.normpdf(x, lower_i_mean[k], np.sqrt(lower_i_var[k])))
        for k in range(2):
            axs[0].plot(x, mlab.normpdf(x, upper_i_mean[k], np.sqrt(upper_i_var[k])))

        plt.savefig('/tmp/task_2_unit_%d.pdf' % i)
    """


def plot_pred_values(path=""):

    no_tasks = 5
    task_range = range(no_tasks)
    #task_range = [0, 1, 2]

    for task_id in task_range:
        res = np.load(path + 'pred_%d.npz' % task_id)
        pred = res['pred']
        pred_true = res['pred_true']
        pred_total = res['pred_total']

        #print task_id

        plt.figure(task_id + 1)
        for task_id2 in range(task_id+1):
            #print task_id2, np.average(lower[task_id2])
            hist_plot = np.array(pred[task_id2])
            plt.hist(hist_plot, bins=50, histtype='stepfilled', label='%d' % task_id2)
            plt.suptitle('Predicted classes\' pred values for task %d' % task_id)

        plt.figure(task_id + no_tasks + 1)
        for task_id2 in range(task_id+1):
            #print task_id2, np.average(lower[task_id2])
            hist_plot = np.array(pred_true[task_id2])
            plt.hist(hist_plot, bins=50, histtype='stepfilled', label='%d' % task_id2)
            plt.suptitle('True classes\' pred values for task %d' % task_id)

        plt.figure(task_id + 2*no_tasks + 1)
        for task_id2 in range(task_id+1):
            #print task_id2, np.average(lower[task_id2])
            hist_plot = []

            for task_id3 in task_range:
                hist_plot = np.concatenate((hist_plot, pred_total[task_id2][task_id3]))
                #print pred_total[task_id3]

            plt.hist(hist_plot, bins=50, histtype='stepfilled', label='%d' % task_id2)
            plt.suptitle('Total classes\' pred values after task %d' % task_id)

    for task_id in task_range:
        plt.figure(task_id + 1)
        plt.legend()
        plt.savefig(path + 'pred_hist_%d.png' % task_id)

        plt.figure(task_id + no_tasks + 1)
        plt.legend()
        plt.savefig(path + 'pred_true_hist_%d.png' % task_id)

        plt.figure(task_id + 2*no_tasks + 1)
        plt.legend()
        plt.savefig(path + 'pred_total_hist_%d.png' % task_id)



if __name__ == "__main__":
    epoch_pause = []
    no_hiddens = [256]

    path = 'singlehead/one_hidden_layer/300epochs/observed_classes/'

    print 'Local reparameterisation, plot weights'

    #visualise_layer_weights(no_hiddens, path="singlehead/test_long/", single_head=single_head)
    visualise_layer_weights(no_hiddens, path=path)
    #plot_pred_values(path=path)