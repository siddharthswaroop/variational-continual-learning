import matplotlib

#matplotlib.use('Agg')
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import pdb


def accuracy_epoch(no_tasks = 5, path=""):

    plt.figure(1, figsize=(20, 5))

    res = np.load(path + 'accuracy.npz')
    accuracies = res['acc']
    index = res['ind']

    ymin = 1.0
    end_ind = 23

    for task_id in reversed(range(no_tasks)):

        if np.min(accuracies[:,task_id,:task_id+1]) < ymin:
            ymin = np.min(accuracies[:,task_id,:task_id+1])

        for task_id_plot in range(task_id+1):
            plt.figure(1)
            plt.subplot(1, 5, task_id_plot+1)
            plt.plot(index[:end_ind], accuracies[:end_ind,task_id,task_id_plot], label='After task %d' % (task_id+1))

        plt.figure(2)
        acc_avg = np.mean(accuracies[:,task_id,:task_id+1],1)
        plt.plot(index[:end_ind], acc_avg[:end_ind], label='After task %d' % (task_id+1))

    plt.figure(1)
    plt.legend()
    for plot_id in range(no_tasks):
        plt.subplot(1, 5, plot_id + 1)
        plt.ylim((ymin, 1.0))
    if path == "smallinitalways/":
        plt.suptitle('Accuracy on each task, init small means, small variances')
    elif path == 'prevposterior/':
        plt.suptitle('Accuracy on each task, init at previous posterior')
    plt.figure(2)
    plt.ylim((ymin, 1.0))
    plt.legend()
    if path == "smallinitalways/":
        plt.suptitle('Average accuracy over all seen tasks, init small means and variances')
    elif path == 'prevposterior/':
        plt.suptitle('Average accuracy over all seen tasks, init at previous posterior')

    plt.show()


if __name__ == "__main__":
    no_tasks = 5
    print 'Local reparameterisation, plot accuracy'

    #fix_acc()
    accuracy_epoch(no_tasks, path="smallinitalways/")
