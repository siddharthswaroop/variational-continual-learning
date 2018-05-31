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



def visualise_weights(no_hiddens=256, path=""):


    no_tasks = 5

    for task_id in range(no_tasks):
        res = np.load(path + 'weights_%d.npz' % task_id)
        lower = res['lower']
        m_low = lower[0,:]
        var_low = np.exp(lower[1,:])
        upper = res['upper']
        m_upper = upper[0,:]
        var_upper = np.exp(upper[1,:])


        # Lower network
        in_dim = 784
        in_size = [28, 28]
        no_params = in_dim * no_hiddens
        m_low = m_low[:no_params].reshape([in_dim, no_hiddens])
        var_low = var_low[:no_params].reshape([in_dim, no_hiddens])
        m_min, m_max = np.min(m_low), np.max(m_low)
        v_min, v_max = np.min(var_low), np.max(var_low)

        no_cols = int(np.sqrt(no_hiddens))
        no_rows = int(no_hiddens/no_cols)

        print "creating figures ..."
        fig0, axs0 = plt.subplots(no_rows, no_cols, figsize=(10, 10))
        fig1, axs1 = plt.subplots(no_rows, no_cols, figsize=(10, 10))

        fig0.suptitle("Mean after task %d, min = %f, max = %f" % (task_id+1, np.min(np.absolute(m_low)), np.max(np.absolute(m_low))))
        fig1.suptitle("Variance after task %d, min = %f, max = %f" % (task_id+1, np.min(np.absolute(var_low)), np.max(np.absolute(var_low))))

        for i in range(no_rows):
            for j in range(no_cols):
                k = i * no_cols + j
                ma = m_low[:, k].reshape(in_size)
                va = var_low[:, k].reshape(in_size)

                axs0[i, j].matshow(ma, cmap=matplotlib.cm.binary, vmin=m_min, vmax=m_max)
                axs0[i, j].set_xticks(np.array([]))
                axs0[i, j].set_yticks(np.array([]))

                axs1[i, j].matshow(va, cmap=matplotlib.cm.binary, vmin=v_min, vmax=v_max)
                axs1[i, j].set_xticks(np.array([]))
                axs1[i, j].set_yticks(np.array([]))


        # Upper weights
        no_params = no_hiddens * 2
        m_upper = m_upper[:no_params].reshape([no_hiddens, 2])
        var_upper = var_upper[:no_params].reshape([no_hiddens, 2])
        x_max = np.max(m_upper) + np.sqrt(np.max(var_upper))

        no_cols = int(np.sqrt(no_hiddens))
        no_rows = int(no_hiddens/no_cols)

        x = np.linspace(-x_max, x_max, 100)
        fig, axs = plt.subplots(no_rows, no_cols, figsize=(10, 10))
        fig.suptitle("Upper level weights for task %d (after task %d), min = %f, max = %f" % (
        task_id + 1, task_id + 1, -x_max, x_max))

        for i in range(no_rows):
            for j in range(no_cols):
                k = i * no_cols + j
                axs[i,j].plot(x, mlab.normpdf(x, m_upper[k][0], np.sqrt(var_upper[k][0])))
                axs[i, j].plot(x, mlab.normpdf(x, m_upper[k][1], np.sqrt(var_upper[k][1])))
                axs[i, j].set_xticks(np.array([]))
                axs[i, j].set_yticks(np.array([]))


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

if __name__ == "__main__":
    epoch_pause = [20, 80, 140, 142, 144, 146, 148, 150]
    no_hiddens = 256
    print 'Local reparameterisation, plot weights'
    # check_weight_pruning(path='small_init/')
    visualise_weights(no_hiddens, path="")
    # check_weight_pruning(no_hiddens)
    # visualise_weights_epoch(no_hiddens,epoch_pause)
    # visualise_weights_vi_batch(no_hiddens)