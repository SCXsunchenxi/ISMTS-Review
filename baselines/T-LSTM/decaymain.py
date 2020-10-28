import sys
import pickle
import numpy as np
import math
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from LSTMtimedecay import LSTM

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj

def convert_one_hot(label_list):
    for i in range(len(label_list)):
        sec_col = np.ones([label_list[i].shape[0],label_list[i].shape[1],1])
        label_list[i] = np.reshape(label_list[i],[label_list[i].shape[0],label_list[i].shape[1],1])
        sec_col -= label_list[i]
        label_list[i] = np.concatenate([label_list[i],sec_col],2)
    return label_list


def training(path,learning_rate,training_epochs,train_dropout_prob,hidden_dim,fc_dim,key,model_path):
    path_string = path + '/TrainData.seqs'
    data_train_batches = load_pkl(path_string)

    path_string = path + '/TrainInterval.seqs'
    elapsed_train_batches = load_pkl(path_string)

    path_string = path + '/TrainLabel.seqs'
    labels_train_batches = load_pkl(path_string)

    number_train_batches = len(data_train_batches)
    print("Train data is loaded!")

    input_dim = np.array(data_train_batches[0]).shape[2]
    output_dim = np.array(labels_train_batches[0]).shape[1]

    lstm = LSTM(input_dim, output_dim, hidden_dim, fc_dim,key)

    cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):  #
            # Loop over all batches
            total_cost = 0
            for i in range(number_train_batches):  #
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                                                         elapsed_train_batches[i]
                #batch_ts = np.reshape(batch_ts, [np.array(batch_ts).shape[0], np.array(batch_ts).shape[2]])
                sess.run(optimizer,feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,\
                                              lstm.keep_prob:train_dropout_prob, lstm.time:batch_ts})


        print("Training is over!")
        saver.save(sess,model_path)

        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []
        for i in range(number_train_batches):  #
            batch_xs, batch_ys, batch_ts = data_train_batches[i], labels_train_batches[i], \
                                                     elapsed_train_batches[i]
            #batch_ts = np.reshape(batch_ts, [np.array(batch_ts).shape[0], np.array(batch_ts).shape[2]])
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), feed_dict={
                lstm.input:batch_xs, lstm.labels: batch_ys,lstm.keep_prob: train_dropout_prob, lstm.time: batch_ts})

            if i > 0:
                Y_true = np.concatenate([Y_true, y_train], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_train], 0)
                Labels = np.concatenate([Labels, labels_train], 0)
                Logits = np.concatenate([Logits, logits_train], 0)
            else:
                Y_true = y_train
                Y_pred = y_pred_train
                Labels = labels_train
                Logits = logits_train

        total_acc = accuracy_score(Y_true, Y_pred)
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_auc_macro = roc_auc_score(Labels, Logits, average='macro')

        print("Train Accuracy = {:.3f}".format(total_acc))
        print("Train AUC = {:.3f}".format(total_auc))
        print("Train AUC Macro = {:.3f}".format(total_auc_macro))



def testing(path,hidden_dim,fc_dim,key,model_path):
    path_string = path + '/TestData.seqs'
    data_test_batches = load_pkl(path_string)

    path_string = path + '/TestInterval.seqs'
    elapsed_test_batches = load_pkl(path_string)

    path_string = path + '/TestLabel.seqs'
    labels_test_batches = load_pkl(path_string)

    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    input_dim = np.array(data_test_batches[0]).shape[2]
    output_dim = np.array(labels_test_batches[0]).shape[1]

    test_dropout_prob = 1.0
    lstm_load = LSTM(input_dim, output_dim, hidden_dim, fc_dim, key)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        Y_true = []
        Y_pred = []
        Logits = []
        Labels = []
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], labels_test_batches[i], \
                                                     elapsed_test_batches[i]
            c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
                                                                             feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.labels: batch_ys,\
                                                                                        lstm_load.time: batch_ts,\
                                                                                        lstm_load.keep_prob: test_dropout_prob})
            if i > 0:
                Y_true = np.concatenate([Y_true, y_test], 0)
                Y_pred = np.concatenate([Y_pred, y_pred_test], 0)
                Labels = np.concatenate([Labels, labels_test], 0)
                Logits = np.concatenate([Logits, logits_test], 0)
            else:
                Y_true = y_test
                Y_pred = y_pred_test
                Labels = labels_test
                Logits = logits_test
        total_auc = roc_auc_score(Labels, Logits, average='micro')
        total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
        total_acc = accuracy_score(Y_true, Y_pred)
        print("Test Accuracy = {:.3f}".format(total_acc))
        print("Test AUC Micro = {:.3f}".format(total_auc))
        print("Test AUC Macro = {:.3f}".format(total_auc_macro))

        fpr, tpr, threshold = metrics.roc_curve(Y_true, Y_pred)

        plt.title('Test AUC-ROC')
        plt.plot(fpr, tpr, c='r',label='Val AUC = %0.3f' % total_auc)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('testROC.jpg', dpi=300)

        states=[]
        for i in range(number_test_batches):
            batch_xs, batch_ys, batch_ts = data_test_batches[i], labels_test_batches[i], \
                                                     elapsed_test_batches[i]
            state = sess.run(lstm_load.get_states(),feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.labels: batch_ys,\
                                                                                        lstm_load.time: batch_ts,\
                                                                                        lstm_load.keep_prob: test_dropout_prob})
            states.append(state)
        pickle.dump(states, open('../emb/states.seqs', 'wb'), -1)
        print("[*] States saved at emb/states.seqs")



def main(training_mode,data_path, learning_rate, training_epochs,dropout_prob,hidden_dim,fc_dim,model_path):

    training_mode = int(training_mode)
    path = str(data_path)

    # train
    if training_mode == 1:
        learning_rate = learning_rate
        training_epochs = int(training_epochs)
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        training(path,learning_rate,training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path)

    # test
    elif training_mode==0:
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        testing(path, hidden_dim, fc_dim, training_mode, model_path)




if __name__ == "__main__":

   main(training_mode=0,data_path='../BatchData', learning_rate= 2e-3, training_epochs=15,dropout_prob=0.25,hidden_dim=64,fc_dim=32,model_path='../model/')


