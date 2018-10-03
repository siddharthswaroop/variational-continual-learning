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

def get_scores_output_pred(model, x_testsets, y_testsets, no_repeats=1, single_head=False, task_id=0):

    acc = []
    pred_vec = []
    pred_vec_true = []
    pred_vec_total = []

    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]
        pred = model.prediction(x_test, head)
        for j in range(no_repeats-1):
            pred = pred + model.prediction_prob(x_test, head)
        pred_mean = np.mean(pred, axis=0)
        pred_y = np.argmax(pred_mean, axis=1)
        y = np.argmax(y_test, axis=1)

        #print 'pred mean', i, model.prediction_prob(x_test, head)

        sum_task_prob = 0.0
        for j in range(np.size(y)):
            sum_task_prob = sum_task_prob + pred_mean[j,y[j]]

            ## Print if we are predicting something other than digits from the most recently seen task
            #if not (pred_y[j] == 2*task_id or pred_y[j] == 2*task_id+1):
            #    print y[j], pred_y[j]

        print 2*i, sum_task_prob/np.size(y)

        ## Compare pred values for certain tasks
        #if i == 1 and task_id == 2:
        #    for j in range(20):
        #        print y[j], pred_y[j]
        #        print pred_vec_total_task1_interm[j], pred_vec_total_task2_interm[j]

        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

    return acc, pred_vec, pred_vec_true, pred_vec_total

def get_scores_splitMNIST_output_pred(model, x_testsets, y_testsets, no_repeats=1, single_head=False, task_id=0):
    acc = []
    pred_vec_true = []
    pred_vec = []
    pred_vec_total = []
    pred_vec_total_task0 = []
    pred_vec_total_task1 = []
    pred_vec_total_task2 = []
    pred_vec_total_task3 = []
    pred_vec_total_task4 = []

    pred_fix = np.zeros(5)
    pred_mean_list = []

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
        pred_vec_true_interm = np.zeros(np.size(y))
        pred_vec_interm = np.zeros(np.size(y))
        pred_vec_total_task0_interm = np.zeros(np.size(y))
        pred_vec_total_task1_interm = np.zeros(np.size(y))
        pred_vec_total_task2_interm = np.zeros(np.size(y))
        pred_vec_total_task3_interm = np.zeros(np.size(y))
        pred_vec_total_task4_interm = np.zeros(np.size(y))
        for j in range(np.size(y)):
            sum_task_prob = sum_task_prob + pred_mean[j,y[j]]
            pred_vec_true_interm[j] = pred_mean[j,y[j]]
            pred_vec_interm[j] = pred_mean[j,pred_y[j]]
            pred_vec_total_task0_interm[j] = np.amax([pred_mean[j, 0], pred_mean[j, 1]]) # Classes (digits) 0 or 1
            pred_vec_total_task1_interm[j] = np.amax([pred_mean[j, 2], pred_mean[j, 3]])
            pred_vec_total_task2_interm[j] = np.amax([pred_mean[j, 4], pred_mean[j, 5]])
            pred_vec_total_task3_interm[j] = np.amax([pred_mean[j, 6], pred_mean[j, 7]])
            pred_vec_total_task4_interm[j] = np.amax([pred_mean[j, 8], pred_mean[j, 9]])

            ## Print if we are predicting something other than digits from the most recently seen task
            #if not (pred_y[j] == 2*task_id or pred_y[j] == 2*task_id+1):
            #    print y[j], pred_y[j]

        print 2*i, sum_task_prob/np.size(y)

        pred_fix[i] = sum_task_prob/np.size(y)
        pred_mean_list.append(pred_mean)

        ## Compare pred values for certain tasks
        #if i == 1 and task_id == 2:
        #    for j in range(20):
        #        print y[j], pred_y[j]
        #        print pred_vec_total_task1_interm[j], pred_vec_total_task2_interm[j]

        pred_vec_true.append(pred_vec_true_interm)
        pred_vec.append(pred_vec_interm)

        pred_vec_total_task0.append(pred_vec_total_task0_interm)
        pred_vec_total_task1.append(pred_vec_total_task1_interm)
        pred_vec_total_task2.append(pred_vec_total_task2_interm)
        pred_vec_total_task3.append(pred_vec_total_task3_interm)
        pred_vec_total_task4.append(pred_vec_total_task4_interm)

        #cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        #acc.append(cur_acc)

    pred_vec_total.append(pred_vec_total_task0)
    pred_vec_total.append(pred_vec_total_task1)
    pred_vec_total.append(pred_vec_total_task2)
    pred_vec_total.append(pred_vec_total_task3)
    pred_vec_total.append(pred_vec_total_task4)

    for i in range(len(x_testsets)):
        head = 0 if single_head else i
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred_mean = pred_mean_list[i]


        y = np.argmax(y_test, axis=1)

        sum_task_prob = 0.0
        for j in range(np.size(y)):
            for task_id2 in range(task_id+1):
                pred_mean[j, 2*task_id2] += (pred_fix[task_id] - pred_fix[task_id2])
                pred_mean[j, 2*task_id2+1] += (pred_fix[task_id] - pred_fix[task_id2])
            sum_task_prob = sum_task_prob + pred_mean[j, y[j]]

        pred_y = np.argmax(pred_mean, axis=1)

        print 't', 2*i, sum_task_prob / np.size(y)

        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

    return acc, pred_vec, pred_vec_true, pred_vec_total

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
