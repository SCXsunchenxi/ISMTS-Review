import tensorflow as tf
import numpy as np

def load_pkl(path):
    with open(path,'rb') as f:
        obj = pickle.load(f)
        return obj


num_epochs = 1000
path='../BatchData'
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


#define variables used in the graph
X = tf.placeholder('float', shape=[None, None, input_dim]
y = tf.placeholder(tf.float32, [None, 2])


w_h = tf.Variable(tf.random_normal([input_dim,64], stddev=0.01))
b1 = tf.Variable(tf.zeros([64]))
w_h2 = tf.Variable(tf.random_normal([64,64], stddev=0.01))
b2 = tf.Variable(tf.zeros([64]))
w_out = tf.Variable(tf.random_normal([64,2], stddev=0.01))
b3 = tf.Variable(tf.zeros([2]))

#build the graph
def model(X,y,w_h,w_h2,w_out, b1, b2, b3):
	h = tf.nn.relu(tf.matmul(X,w_h) + b1)
	h = tf.nn.dropout(h, .2)
	h2 = tf.nn.relu(tf.matmul(h,w_h2) + b2)
	h2 = tf.nn.dropout(h2, .2)
	output = tf.matmul(h2, w_out) + b3
	return output

#feed forward through the model
yhat = model(X, y, w_h, w_h2, w_out,b1, b2, b3)

#loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat, y))

#optimizer function for backprop + weight updates
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(yhat, 1)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(num_epochs):
        for i in range(number_train_batches):
            # batch_xs is [number of patients x sequence length x input dimensionality]
            batch_xs, batch_ys = data_train_batches[i], labels_train_batches[i]
		    train = sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
		    if epoch % 100 == 0:
			    acc = (np.mean(np.argmax(batch_ys, axis=1) == sess.run(predict_op, feed_dict={X: batch_xs, y: batch_ys})))
			    print ("training accuracy at epoch " + str(epoch) + ": " + str(acc))
	#print test set results
    for i in range(number_train_batches):
        # batch_xs is [number of patients x sequence length x input dimensionality]
        batch_xs, batch_ys = data_test_batches[i], labels_test_batches[i]
        acc = (np.mean(np.argmax(batch_ys, axis=1) == sess.run(predict_op, feed_dict={X: batch_xs, y: batch_ys})))
        print ("Test set accuracy: " + str(acc))
