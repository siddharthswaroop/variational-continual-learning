import matplotlib

#matplotlib.use('Agg')
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import pdb


def accuracy_epoch(no_tasks = 5, path=""):

    plt.figure(1, figsize=(25, 3))

    res = np.load(path + 'accuracy.npz')
    accuracies = res['acc']
    index = res['ind']

    for task_id in reversed(range(no_tasks)):

        acc = accuracies[task_id*len(index):(task_id+1)*len(index)]
        for task_id_plot in range(task_id+1):
            plt.figure(1)
            plt.subplot(1, 5, task_id_plot+1)
            plt.plot(index, acc[:,task_id_plot], label='After task %d' % (task_id+1))
            plt.ylim((0.9,1.0))

        plt.figure(2)
        acc_avg = np.mean(acc[:,:task_id+1],1)
        plt.plot(index, acc_avg, label='After task %d' % (task_id+1))
        plt.ylim((0.95,1.0))


    plt.figure(1)
    plt.legend()
    plt.suptitle('Accuracy on each task')
    plt.figure(2)
    plt.legend()
    plt.suptitle('Average accuracy over all seen tasks')

    plt.show()



if __name__ == "__main__":
    no_tasks = 5
    print 'Local reparameterisation, plot accuracy'

    accuracy_epoch(no_tasks, path="smallinitalways/")
