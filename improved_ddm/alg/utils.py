import numpy as np


def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    return merged_x, merged_y


# Get test accuracies from model
def get_scores_output_pred(model, x_testsets, y_testsets, test_classes, task_idx=[0], multi_head=False):
    acc = []

    # Go over each task's testset
    for i in task_idx:
        x_test, y_test = x_testsets[i], y_testsets[i]

        # Output from model
        pred = model.prediction_prob(x_test)

        # Mean over the different Monte Carlo models
        pred_mean_total = np.mean(pred, axis=1)

        heads = i if multi_head else 0  # Different for multi-head and single-head

        # test_classes[heads] holds which classes we are predicting between
        # We are only interested in finding which of these classes has maximum prediction output from model
        # We therefore set all the other classes to have a large negative value: this is pred_mean
        pred_mean = -10000000000*np.ones(np.shape(pred_mean_total))
        pred_mean[test_classes[heads], :, :] = pred_mean_total[test_classes[heads], :, :]

        # Predicted class
        pred_y = np.argmax(pred_mean, axis=0)
        pred_y = pred_y[:, 0]

        # True class
        y = np.argmax(y_test, axis=1)

        # Calculate test accuracy
        cur_acc = len(np.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)

    return acc
