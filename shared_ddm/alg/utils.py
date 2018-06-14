import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb

def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y

# For plotting histogram of activations
def get_scores_hist_plot(model, x_testsets, y_testsets, no_repeats=1, single_head=False, path=""):
    acc = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        pred = model.prediction_hist_plot(x_test, head)
        for j in range(no_repeats-1):
            pred = pred + model.prediction_hist_plot(x_test, head)

        y = np.argmax(y_test, axis=1)
        hist_plot = []
        for i in range(np.size(y)):
            pred_test = pred[:,i,y[i]]
            hist_plot.append(pred_test)
        hist_plot = np.array(hist_plot)
        hist_plot = hist_plot.reshape([-1])
        plt.figure(1)
        plt.hist(hist_plot, bins=500, histtype='stepfilled')
        plt.suptitle('Histogram of activations (upper layer)')
        plt.savefig(path + 'hist_test.png')

    return acc

def get_scores(model, x_testsets, y_testsets, no_repeats=1, single_head=False):
    acc = []
    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        pred = model.prediction(x_test, head)
        for j in range(no_repeats-1):
            pred = pred + model.prediction_prob(x_test, head)
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        y = np.argmax(y_test, axis=1)

        sum_task_prob = 0.0
        for j in range(np.size(y)):
            sum_task_prob = sum_task_prob + pred_mean[j,y[j]]
        print 2*i, sum_task_prob/np.size(y)

        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)
    return acc

def concatenate_results(score, all_score):
    if all_score.size == 0:
        all_score = np.reshape(score, (1,-1))
    else:
        new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
        new_arr[:] = np.nan
        new_arr[:,:-1] = all_score
        all_score = np.vstack((new_arr, score))
    return all_score

def plot(filename, vcl, rand_vcl, kcen_vcl):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig = plt.figure(figsize=(7,3))
    ax = plt.gca()
    plt.plot(np.arange(len(vcl))+1, vcl, label='VCL', marker='o')
    plt.plot(np.arange(len(rand_vcl))+1, rand_vcl, label='VCL + Random Coreset', marker='o')
    plt.plot(np.arange(len(kcen_vcl))+1, kcen_vcl, label='VCL + K-center Coreset', marker='o')
    ax.set_xticks(range(1, len(vcl)+1))
    ax.set_ylabel('Average accuracy')
    ax.set_xlabel('\# tasks')
    ax.legend()

    fig.savefig(filename, bbox_inches='tight')
    plt.close()
