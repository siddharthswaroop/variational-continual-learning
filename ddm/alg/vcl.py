import numpy as np
import tensorflow as tf
import utils
from cla_models_multihead import Vanilla_NN, MFVI_NN

def run_vcl(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, ml_init=False):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    #ml_init = False

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        if task_id == 0:
            mf_weights = None
            mf_variances = None
            ml_weights = None
            if (ml_init):
                # Train network with maximum likelihood to initialize model
                ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_weights=mf_weights)
                ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
                ml_weights = ml_model.get_weights()
                # mf_variances = None
                ml_model.close_session()

        if task_id > 0:
            ml_weights = None

        #if ml_init and task_id > 0:
        #    # Train network with maximum likelihood to initialize model
        #    ml_model = Vanilla_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_weights=mf_weights)
        #    ml_model.train(x_train, y_train, task_id, no_epochs, bsize)
        #    ml_weights = ml_model.get_weights()
        #    #mf_variances = None
        #    ml_model.close_session()

        # Select coreset if needed
        if coreset_size > 0:
            x_coresets, y_coresets, x_train, y_train = coreset_method(x_coresets, y_coresets, x_train, y_train, coreset_size)

        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
        mf_model.train(x_train, y_train, head, no_epochs, bsize)
        mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size, no_repeats=1)
        all_acc = utils.concatenate_results(acc, all_acc)
        print all_acc

        np.savez('sandbox/weights_%d.npz' % task_id, means=mf_weights, variances=mf_variances)

        mf_model.close_session()


    # Print in a suitable format
    for task_id in range(data_gen.max_iter):
        for i in range(task_id+1):
            print all_acc[task_id][i],
        print ''


    return all_acc



def vcl_predictions(hidden_size, no_epochs, data_gen, coreset_method, coreset_size=0, batch_size=None, single_head=True, ml_init = False, no_pred_repeats=10):
    in_dim, out_dim = data_gen.get_dims()
    x_coresets, y_coresets = [], []
    x_testsets, y_testsets = [], []

    all_acc = np.array([])

    #ml_init = False

    for task_id in range(data_gen.max_iter):
        x_train, y_train, x_test, y_test = data_gen.next_task()
        x_testsets.append(x_test)
        y_testsets.append(y_test)

        # Set the readout head to train
        head = 0 if single_head else task_id
        bsize = x_train.shape[0] if (batch_size is None) else batch_size

        load_weights = np.load('sandbox/weights_%d.npz' % task_id)
        mf_weights = load_weights['means']
        mf_variances = load_weights['variances']

        # Train on non-coreset data
        mf_model = MFVI_NN(in_dim, hidden_size, out_dim, x_train.shape[0], prev_means=mf_weights, prev_log_variances=mf_variances)
        #mf_weights, mf_variances = mf_model.get_weights()

        # Incorporate coreset data and make prediction
        acc = utils.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, no_epochs, single_head, batch_size, no_repeats=no_pred_repeats)
        all_acc = utils.concatenate_results(acc, all_acc)
        print all_acc

        #np.savez('sandbox/weights_%d.npz' % task_id, means=mf_weights, variances=mf_variances)

        mf_model.close_session()

    # Print in a suitable format
    for task_id in range(data_gen.max_iter):
        for i in range(task_id+1):
            print all_acc[task_id][i],
        print ''


    return all_acc