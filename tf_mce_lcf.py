# coding: utf-8

import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_data(dataset_path, delimiter = ' ', sort_by_label=True, print_result=True):
    data = np.genfromtxt(dataset_path, delimiter=delimiter)
    x_data = data[:, :-1]
    class_data = data[:, -1]

    raw_labels = sorted(set(class_data))
    mapping = {}

    for raw_label, i in zip(raw_labels, range(len(raw_labels))):
        mapping[raw_label] = i

    for i in range(len(class_data)):
        y_old = class_data[i]
        class_data[i] = mapping[y_old]

    class_data = class_data.astype(int)
    if sort_by_label:
        x_data = x_data[class_data.argsort()]
        class_data = np.sort(class_data)

    if print_result:
        print (" -- Load data results --")
        print ("  * Dataset path: %s" % dataset_path)
        print ("  * X data: %s {%s}" % (x_data, x_data.shape))
        print ("  * Class label: %s {%s}" % (class_data, class_data.shape))
        print ("  * Sorted by class: %s" % sort_by_label)

    return (x_data, class_data)

def make_directories(output_dir):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

def save_plt_fig(plt_obj, save_path, clear_plt_obj=True):
    base_dir = os.path.dirname(save_path)
    make_directories(base_dir)
    plt_obj.savefig(save_path, bbox_inches='tight', pad_inches=0.5, format="png", dpi=200)
    if clear_plt_obj:
        plt_obj.clf()


# Main
if __name__ == "__main__":

    ## Set variables
    ## About IO processing
    script_name = str(os.path.basename(__file__)).split(".")[0]
    dataset_path = "./datasets/GMM.dat"
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    output_dir = "./out"
    execute_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")

    ## Read data
    X, y = load_data(dataset_path)

    ## About data correcting
    all_sample_size = X.shape[0]
    #test_sample_size = 0
    test_sample_size = int(all_sample_size * 0.5)

    ## About model construction
    epoch = 100
    loss_smoothness = 1.0
    learning_rate = 0.5

    # About plotting results
    plt.rcParams['font.size'] = 18
    fig = plt.figure(figsize=(20, 16))
    observation_lapse = 10

    ## Correct data
    categories, N_categories = np.unique(y, return_counts=True)
    if len(categories) < 2 and np.amin(N_categories) < 3:  # classes >= 2, samples for a class >= 3
        print ("ERROR: running this script allows only classes >= 2, and samples >= 3")
        print (" * Found: classes: %s; samples: %s" % (categories, N_categories))
        quit(1)

    if test_sample_size > all_sample_size:
        print ("ERROR: cannot create test set. test_sample_size: %s; all_sample_size: %s" % (test_sample_size, all_sample_size))
        quit(1)

    ## Fix random seed for experiments
    sess = tf.Session()
    np.random.seed(0)
    tf.set_random_seed(0)

    corrected_x = np.array(X)
    one_hot_y = np.identity(max(y) + 1)[y]  # One-hot vector (max_value + 1 (for "0" class))
    corrected_y = np.array(one_hot_y)  # One-hot vector

    train_index = np.random.choice(all_sample_size, size=(all_sample_size - test_sample_size), replace=False)
    test_index = np.ones(all_sample_size, dtype=np.bool)
    test_index[train_index] = 0
    train_x = corrected_x[train_index]
    train_y = np.array([y_ for y_ in corrected_y[train_index]])

    if not test_sample_size == 0:
        test_x = corrected_x[test_index]
        test_y = np.array([y_ for y_ in corrected_y[test_index]])
    else:  # If testing sample is none, set training samples as testing samples
        print ("WARNING: There is no testing samples, set training samples insted of it.")
        test_x = corrected_x[train_index]
        test_y = np.array([y_ for y_ in corrected_y[train_index]])

    # Set data parameters
    dim_num = corrected_x.shape[1]
    class_num = corrected_y.shape[1]

    ## MCE Training: Prototype setting
    ## Set Tensorflow statement
    y_target = tf.placeholder(shape=[None, class_num], dtype=tf.float32)
    x_data = tf.placeholder(shape=[None, dim_num], dtype=tf.float32)
    alpha = tf.Variable(loss_smoothness, dtype=tf.float32)
    rho = tf.Variable(learning_rate, dtype=tf.float32)

    # g = A' X'
    x_data_dash = tf.concat([tf.reshape(tf.ones(tf.shape(x_data)[0]), [-1, 1]), x_data], axis=1)
    A_dash = tf.Variable(tf.random_uniform(shape=[dim_num + 1, class_num]))
    g = tf.matmul(x_data_dash, A_dash)

    # Extract g value of class i
    class_i_ind = y_target
    g_i_value = tf.multiply(class_i_ind, g)
    g_not_i_value = tf.multiply(tf.ones_like(class_i_ind), -np.inf)
    g_i = tf.where(tf.equal(g_i_value, 0), g_not_i_value, g_i_value)
    g_i_max = tf.reduce_max(g_i, axis=1, keepdims=True)

    # Extract g value of class j (j != i)
    class_j_ind = tf.subtract(tf.ones_like(y_target), y_target)
    g_j_value = tf.multiply(class_j_ind, g)
    g_not_j_value = tf.multiply(tf.ones_like(class_j_ind), -np.inf)
    g_j = tf.where(tf.equal(g_j_value, 0), g_not_j_value, g_j_value)
    g_j_max = tf.reduce_max(g_j, axis=1, keepdims=True)
    class_j_max_ind = tf.one_hot(tf.argmax(g_j, axis=1), class_num)

    # Calculate miss classification measure and losses
    d = tf.subtract(g_j_max, g_i_max)  # n * 1
    l = tf.divide(1.0, tf.add(1.0, tf.exp(tf.multiply(-alpha, d))))  # sigmoid -> n * 1
    L = tf.reduce_mean(l, axis=0)  # 1

    g_sign_ind = tf.add(tf.multiply(-1.0, class_i_ind), class_j_max_ind)
    diff_l = tf.multiply(alpha, tf.multiply(l, tf.subtract(1.0, l)))  # l' = alpha * l * (1-l)
    diff_d = tf.matmul(tf.transpose(tf.multiply(diff_l, x_data_dash)), g_sign_ind)
    diff_A = tf.divide(tf.multiply(rho, diff_d), tf.cast(tf.shape(x_data)[0], tf.float32))
    A_next = tf.assign(A_dash, tf.subtract(A_dash, diff_A))

    y_pred = tf.argmax(g, axis=1)  # n
    y_true = tf.argmax(y_target, axis=1)  # n
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))  # 1
    error = tf.subtract(1.0, accuracy)  # 1

    # Initialize all of tensor
    init = tf.global_variables_initializer()
    sess.run(init)

    results = []
    res_out_dir = "%s/%s_%s_%s" % (output_dir, script_name, dataset_name, execute_time)

    print ("--------------------------------------------------")
    print (" Let us play experiment on %s" % script_name)
    print ("  * Input data path: %s" % dataset_path)
    print ("  * Dataset name: %s" % dataset_name)
    print ("  * Output directory: %s" % res_out_dir)
    print ("  * Plotting lapse: %s" % observation_lapse)
    print ("  * Training X shape: %s" % str(train_x.shape))
    print ("  * Training y shape: %s" % str(train_y.shape))
    print ("  * Testing X shape: %s" % str(test_x.shape))
    print ("  * Testing y shape: %s" % str(test_y.shape))
    print ("  * Epoch: %s" % epoch)
    print ("  * Training step size (Learning rate): %s" % learning_rate)
    print ("  * Loss smoothness: %s" % loss_smoothness)
    print ("--------------------------------------------------")

    for n in range(epoch+1):
        print("step #", str(n))

        # evaluating from 0 to epoch
        tr_err = float(sess.run(error, feed_dict={x_data: train_x, y_target: train_y}))
        ts_err = float(sess.run(error, feed_dict={x_data: test_x, y_target: test_y}))
        l_r = float(sess.run(rho))
        l_s = float(sess.run(alpha))
        loss = float(sess.run(L[0], feed_dict={x_data: train_x, y_target: train_y}))
        print("loss: %s, training_error: %s, testing_error: %s, learning_rate: %s, loss_smoothness: %s" % (loss, tr_err, ts_err, l_r, l_s))

        results.append([n, tr_err, ts_err, loss, l_s, l_r])

        # print results
        if n % observation_lapse == 0:

            # 2D plotting
            X_class_i = X[np.where(y == 0)]
            X_class_j = X[np.where(y == 1)]
            A_values = sess.run(A_dash)

            ax0 = fig.add_subplot(111)
            ax0.scatter(X_class_i[:, 0], X_class_i[:, 1], marker="x", c="skyblue", label="class blue true")
            ax0.scatter(X_class_j[:, 0], X_class_j[:, 1], marker="x", c="lightgreen", label="class green true")
            ax0.scatter(A_values[1, 0], A_values[1, 1], marker="s", s=50, c="blue", label="class blue proto")
            ax0.scatter(A_values[2, 0], A_values[2, 1], marker="s", s=50, c="green", label="class green proto")
            ax0.set_title("Epoch=%s\nTR_Error=%.4f, TS_Error=%.4f, Loss=%.4f, Smoothness=%.2f, LR=%.2f" % (n, tr_err, ts_err, loss, l_s, l_r))
            ax0.set_xlabel("X[0]")
            ax0.set_ylabel("X[1]")
            save_plt_fig(fig, "%s/%s_plots_%06d.png" % (res_out_dir, dataset_name, n))

        if n % observation_lapse == 0 and n != 0:
            ax1 = fig.add_subplot(211)
            res_np = np.array(results)
            res_n = res_np[:, 0]
            res_loss = res_np[:, 2]
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss", color="purple")
            ax1.plot(res_n, res_loss, c="purple")
            ax1.set_title("Loss")

            ax2 = fig.add_subplot(212)
            res_tr_err = res_np[:, 1]
            res_ts_err = res_np[:, 2]
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Training error", color="orange")
            ax2.plot(res_n, res_tr_err, c="orange")
            ax2.set_ylim((0, 0.5))
            ax2_1 = ax2.twinx()
            ax2_1.set_ylabel("Testing error", color="red")
            ax2_1.plot(res_n, res_ts_err, c="red")
            ax2_1.set_ylim((0, 0.5))
            ax2.set_title("Errors")
            save_plt_fig(fig, "%s/%s_values_%06d.png" % (res_out_dir, dataset_name, n))

        # Training
        sess.run(A_next, feed_dict={x_data: train_x, y_target: train_y})

    # Output result
    final_results = np.array(results)
    np.savetxt("%s/%s_ep%s.csv" % (res_out_dir, dataset_name, epoch), final_results, delimiter=",")

    print("-- final results --")
    np.set_printoptions(suppress=True)
    print (final_results)
