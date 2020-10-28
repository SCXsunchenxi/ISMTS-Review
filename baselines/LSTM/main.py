import sys
import pickle
import numpy as np
import math
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from LSTMmodel import LSTM

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj


def training(path,fold,training_epochs,train_dropout_prob,hidden_dim,fc_dim,key,model_path,learning_rate=[1e-5, 2e-2],lr_decay=2000,):
    # train data
    path_string = path + '/TrainData.seqs'
    data_train_batches = load_pkl(path_string)

    path_string = path + '/TrainLabel.seqs'
    labels_train_batches = load_pkl(path_string)

    number_train_batches = len(data_train_batches)

    input_dim = np.array(data_train_batches[0]).shape[2]
    output_dim = np.array(labels_train_batches[0]).shape[1]

    print("Train data is loaded!")

    path_string = path + '/TestData.seqs'
    data_test_batches = load_pkl(path_string)

    path_string = path + '/TestLabel.seqs'
    labels_test_batches = load_pkl(path_string)

    number_test_batches = len(data_test_batches)

    print("Test data is loaded!")

    # model built
    lstm = LSTM(input_dim, output_dim, hidden_dim, fc_dim,key)
    cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
    lr = learning_rate[0]+ tf.train.exponential_decay(learning_rate[1],
                                                    lstm.step,
                                                    lr_decay,
                                                    1 / np.e)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    best_valid_loss = 1e10

    # train
    with tf.Session() as sess:
        sess.run(init)
        valid_acc = []
        valid_auc = []
        valid_loss=[]
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(number_train_batches):
                # batch_xs is [number of patients x sequence length x input dimensionality]
                batch_xs, batch_ys = data_train_batches[i], labels_train_batches[i]
                step = epoch * number_train_batches+ i
                sess.run(optimizer,feed_dict={lstm.input: batch_xs, lstm.labels: batch_ys,lstm.keep_prob:train_dropout_prob,lstm.step:step})
                print('Training epoch ' + str(epoch) + ' batch ' + str(i) + ' done')

            # valid
            loss = []
            Y_pred = []
            Y_true = []
            Labels = []
            Logits = []
            for k in range(number_train_batches):  #
                if (len(data_train_batches[k])<fold): continue
                batch_xs, batch_ys = [data_train_batches[k][-fold]], [labels_train_batches[k][-fold]]
                c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), \
                                                                                      feed_dict={
                                                                                          lstm.input: batch_xs,
                                                                                          lstm.labels: batch_ys,
                                                                                          lstm.keep_prob: train_dropout_prob})
                loss.append(c_train)
                if k > 0:
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
            mean_loss=np.mean(loss)
            print("validation Accuracy = {:.3f}".format(total_acc))
            print("validation AUC = {:.3f}".format(total_auc))
            print("validation AUC Macro = {:.3f}".format(total_auc_macro))
            print("validation Loss = {:.3f}".format(mean_loss))
            print('epoch ' + str(epoch) + ' done........................')

            valid_acc.append(total_acc)
            valid_auc.append(total_auc)
            valid_loss.append(mean_loss)

            if(mean_loss<=best_valid_loss):
                print ("[*] Best loss so far! ")
                saver.save(sess, model_path+'model'+str(fold)+'/')
                print ("[*] Model saved at", model_path+'model'+str(fold)+'/', flush=True)

        pickle.dump(valid_acc, open('../results/valid_acc'+str(fold)+'.seqs', 'wb'), -1)
        pickle.dump(valid_auc, open('../results/valid_auc'+str(fold)+'.seqs', 'wb'), -1)
        pickle.dump(valid_loss, open('../results/valid_loss' + str(fold) + '.seqs', 'wb'), -1)

        print("Fold "+str(fold)+" training is over!")
        saver.save(sess, model_path+'model'+str(fold)+'/')
        print("[******] Model saved at", model_path+'model'+str(fold)+'/', flush=True)

        # test
        loss = []
        Y_pred = []
        Y_true = []
        Labels = []
        Logits = []
        for i in range(number_test_batches):  #
            batch_xs, batch_ys = data_test_batches[i], labels_test_batches[i]
            c_train, y_pred_train, y_train, logits_train, labels_train = sess.run(lstm.get_cost_acc(), \
                                                                                  feed_dict={lstm.input: batch_xs,
                                                                                             lstm.labels: batch_ys,
                                                                                             lstm.keep_prob: train_dropout_prob})
            loss.append(c_train)
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
        print("Test Accuracy = {:.3f}".format(total_acc))
        print("Test AUC = {:.3f}".format(total_auc))
        print("Test AUC Macro = {:.3f}".format(total_auc_macro))



def testing(path, hidden_dim, fc_dim, key, model_path):
    path_string = path + '/TestData.seqs'
    data_test_batches = load_pkl(path_string)

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
            batch_xs, batch_ys = data_test_batches[i], labels_test_batches[i]
            c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
                                                                             feed_dict={lstm_load.input: batch_xs,
                                                                                        lstm_load.labels: batch_ys, \
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


def testing_Uncertainty(path,test_dropout_prob,hidden_dim,fc_dim,key,model_path,model_num):

    path_string = path + '/batches_data_test.seqs'
    data_test_batches = load_pkl(path_string)

    path_string = path + '/batches_label_test.seqs'
    labels_test_batches = load_pkl(path_string)

    print("Test data is loaded!")

    input_dim = np.array(data_test_batches[0]).shape[2]
    output_dim = np.array(labels_test_batches[0]).shape[1]

    test_dropout_prob = test_dropout_prob

    lstm_load = LSTM(input_dim, output_dim, hidden_dim, fc_dim, key)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        acc_in_time_length=[]
        auc_in_time_length=[]
        uncertainty_in_time_length=[]
        batch_xs, batch_ys = data_test_batches[0], labels_test_batches[0]
        time_length = len(batch_xs[0])

        for length in range(time_length-12 , time_length):
            # 时间截断
            batch_xs_sub =  np.array(batch_xs)[:, :length].tolist()

            ACCs = []
            AUCs = []
            Pcs = []
            for j in range(model_num):

                c_test, y_pred_test, y_test, logits_test, labels_test = sess.run(lstm_load.get_cost_acc(),
                                                                                 feed_dict={lstm_load.input: batch_xs_sub,
                                                                                            lstm_load.labels: batch_ys,\
                                                                                           lstm_load.keep_prob: test_dropout_prob})
                Y_true = y_test
                Y_pred = y_pred_test
                Labels = labels_test
                Logits = logits_test

                total_auc_macro = roc_auc_score(Labels, Logits, average='macro')
                total_acc = accuracy_score(Y_true, Y_pred)
                print("Test Accuracy = {:.3f}".format(total_acc))
                print("Test AUC Micro = {:.3f}".format(total_auc_macro))
                print("Test AUC Macro = {:.3f}".format(total_auc_macro))
                ACCs.append(total_acc)
                AUCs.append(total_auc_macro)

                C=np.bincount(Y_pred)
                Pc=[x/np.sum(C) for x in C]
                Pcs.append(Pc)

            meanACC=np.mean(ACCs)
            meanAUC=np.mean(AUCs)

            # total uncertainty
            p_avg=np.array(Pcs).mean(axis=0)
            total_uncertainty=sum((-x)*math.log(x,2) for x in p_avg)
            # expected data uncertainty
            entropy = [sum((-x) * math.log(x, 2) for x in i) for i in Pcs]
            expected_data_uncertainty=np.array(entropy).mean(axis=0)
            # model uncertainty
            model_uncertainty=total_uncertainty-expected_data_uncertainty
            print('mean ACC: '+ str(meanACC)+' mean AUC: '+ str(meanAUC)+' uncertainty: '+ str(model_uncertainty))

            acc_in_time_length.append(meanACC)
            auc_in_time_length.append(meanAUC)
            uncertainty_in_time_length.append(model_uncertainty)

    return acc_in_time_length,auc_in_time_length,uncertainty_in_time_length

#
# def main(argv):
#     training_mode = int(sys.argv[1])
#     path = str(sys.argv[2])
#
#     if training_mode == 1:
#         learning_rate = float(sys.argv[3])
#         training_epochs = int(sys.argv[4])
#         dropout_prob = float(sys.argv[5])
#         hidden_dim = int(sys.argv[6])
#         fc_dim = int(sys.argv[7])
#         model_path = str(sys.argv[8])
#         training(path, learning_rate, training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path)
#     else:
#         hidden_dim = int(sys.argv[3])
#         fc_dim = int(sys.argv[4])
#         model_path = str(sys.argv[5])
#         testing(path, hidden_dim, fc_dim, training_mode, model_path)


def main(training_mode,fold,data_path, learning_rate,lr_decay, training_epochs,dropout_prob,hidden_dim,fc_dim,model_path,model_num=0):
    """

    :param training_mode:  1train，0test，2uncertainty
    :param fold: 5-fold
    :param data_path:
    :param learning_rate:
    :param training_epochs:
    :param dropout_prob: dropout，1 keep all
    :param hidden_dim:
    :param fc_dim:
    :param model_path: save/load model path
    :param model_num: number of models when uncertainty
    """
    training_mode = int(training_mode)
    path = str(data_path)

    # train
    if training_mode == 1:
        learning_rate = learning_rate
        lr_decay=lr_decay
        training_epochs = int(training_epochs)
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        training(path, fold,training_epochs, dropout_prob, hidden_dim, fc_dim, training_mode, model_path,learning_rate, lr_decay)

    # test
    elif training_mode==0:
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        testing(path, hidden_dim, fc_dim, training_mode, model_path)

    #test with mc_dropout
    elif training_mode==2:
        dropout_prob = float(dropout_prob)
        hidden_dim = int(hidden_dim)
        fc_dim = int(fc_dim)
        model_path = str(model_path)
        model_num=model_num
        acc_in_time_length,auc_in_time_length,uncertainty_in_time_length=testing_Uncertainty(path, dropout_prob, hidden_dim, fc_dim, training_mode, model_path,model_num)
        print(acc_in_time_length)
        print(auc_in_time_length)
        print(uncertainty_in_time_length)

if __name__ == "__main__":

   main(training_mode=1,fold=1,data_path='../BatchData', learning_rate=[1e-5, 2e-2],lr_decay=2000, training_epochs=15,dropout_prob=0.25,hidden_dim=64,fc_dim=32,model_path='../model/',model_num=5)


