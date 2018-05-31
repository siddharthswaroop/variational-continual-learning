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



def visualise_weights(no_hiddens=256, path=""):
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

    plt.show()
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


def check_weight_pruning(no_hiddens=[256], no_tasks = 3, no_classifiers=10, path=""):
    res_0 = np.load(path + 'weights_vi_batch_permuted_1000.npz')
    lower_0 = res_0['lower']
    upper_0 = res_0['upper']

    m0 = lower_0[0, :]
    v0 = np.exp(lower_0[1, :])

    # mu1 = means, upper layer, after task 1
    mu1 = upper_0[0][0]
    vu1 = np.exp(upper_0[0][1])

    mu2 = upper_0[0][0]
    vu2 = np.exp(upper_0[0][1])

    mu3 = upper_0[0][0]
    vu3 = np.exp(upper_0[0][1])

    # ml1 = means, lower layers, 1st (lowest) layer
    in_dim = 784
    in_size = [28, 28]
    no_params = in_dim * no_hiddens[0]
    ml1 = m0[:no_params].reshape([in_dim, no_hiddens[0]])
    vl1 = v0[:no_params].reshape([in_dim, no_hiddens[0]])
    param_index = no_params + no_hiddens[0]
    no_params = no_hiddens[0] * no_hiddens[1]
    ml2 = m0[param_index:param_index+no_params].reshape([no_hiddens[0], no_hiddens[1]])
    vl2 = v0[param_index:param_index+no_params].reshape([no_hiddens[0], no_hiddens[1]])

    no_params = no_hiddens[-1] * no_classifiers
    mu1 = mu1[:no_params].reshape([no_hiddens[-1], no_classifiers])
    vu1 = vu1[:no_params].reshape([no_hiddens[-1], no_classifiers])
    mu2 = mu2[:no_params].reshape([no_hiddens[-1], no_classifiers])
    vu2 = vu2[:no_params].reshape([no_hiddens[-1], no_classifiers])
    mu3 = mu3[:no_params].reshape([no_hiddens[-1], no_classifiers])
    vu3 = vu3[:no_params].reshape([no_hiddens[-1], no_classifiers])


    neuron = 82
    print np.max(mu3[neuron,:])
    print np.min(mu3[neuron,:])

    print np.max(vu3[neuron,:])
    print np.min(vu3[neuron,:])

    print mu3[neuron,:]
    print vu3[neuron,:]

    x = np.linspace(-2, 2, 100)

    """
    lims = no_hiddens[0]
    for i in range(lims):
        plt.figure(i+1)
        print i
        lower1_i_mean = ml1[:, i]
        lower1_i_var = vl1[:, i]
        lower2_i_mean = ml2[i,:]
        lower2_i_var = vl2[i,:]
        for k in range(lower1_i_mean.size):
            plt.subplot(212)
            plt.plot(x, mlab.normpdf(x, lower1_i_mean[k], np.sqrt(lower1_i_var[k])))
        for k in range(lower2_i_mean.size):
            plt.subplot(211)
            plt.plot(x, mlab.normpdf(x, lower2_i_mean[k], np.sqrt(lower2_i_var[k])))

        plt.savefig('tmp2/lower/task_3_lower_unit_%d.pdf' % i)
    
    
    lims = no_hiddens[-1]
    for i in range(lims):
        plt.figure(no_hiddens[0]+i+1)
        print i
        lower_i_mean = ml2[:, i]
        lower_i_var = vl2[:, i]
        upper_i_mean = mu3[i, :]
        upper_i_var = vu3[i, :]
        for k in range(lower_i_mean.size):
            plt.subplot(212)
            plt.plot(x, mlab.normpdf(x, lower_i_mean[k], np.sqrt(lower_i_var[k])))
        for k in range(upper_i_mean.size):
            plt.subplot(211)
            plt.plot(x, mlab.normpdf(x, upper_i_mean[k], np.sqrt(upper_i_var[k])))
            plt.ylim(0, 2.0)

        plt.savefig('tmp2/upper/task_3_upper_unit_%d.pdf' % i)
    """



if __name__ == "__main__":
    epoch_pause = [20, 80, 140, 142, 144, 146, 148, 150]
    no_hiddens = [100, 100]
    no_tasks = 3
    no_classifiers = 10
    # check_weight_pruning(path='small_init/')
    # visualise_weights(no_hiddens)
    check_weight_pruning(no_hiddens, no_tasks, no_classifiers)
    # visualise_weights_epoch(no_hiddens,epoch_pause)
    # visualise_weights_vi_batch(no_hiddens)