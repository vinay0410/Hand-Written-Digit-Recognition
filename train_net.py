import scipy.io as so
import numpy as np
import cv2
import tensorflow as tf
import cPickle
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("mnist/", one_hot=True)

batch_size = 100

def accuracy(predictions, labels):
  arr = np.argmax(predictions, axis=1)
  lbl = np.argmax(labels, axis=1)
  return 100*np.mean(arr == lbl)



mygraph = tf.Graph()

with mygraph.as_default():
    train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 784))
    train_labels = tf.placeholder(dtype = tf.float32, shape=(batch_size, 10))
    #train_labels_hot = tf.one_hot(train_labels, depth=10, dtype = tf.float32)
    valid_data = tf.placeholder(tf.float32, shape = (10000, 784))
    #mylambda = tf.constant(0.0001, dtype=tf.float32)
    print train_labels.shape
    layer1 = tf.Variable(tf.truncated_normal([784, 800] , dtype=tf.float32), name="layer1")
    layer1_b = tf.Variable(tf.zeros([800]), name="layer1_b")
    layer2 = tf.Variable(tf.truncated_normal([800, 10] , dtype=tf.float32), name="layer2")
    layer2_b = tf.Variable(tf.zeros([10]), name="layer2_b")

    def model(data):
        a2 = tf.sigmoid((tf.matmul(data, layer1) + layer1_b))
        #a2_dropout = tf.nn.dropout(a2, 0.8)
        #a3_dropout = tf.matmul(a2_dropout, layer2) + layer2_b
        a3 = tf.matmul(a2, layer2) + layer2_b
        #return a3_dropout, a3
        return a3
    logits = model(train_dataset)
    #loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels, logits=logits_d))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_labels, logits=logits))

    #loss_reg1 = tf.nn.l2_loss(layer1)
    #loss_reg2 = tf.nn.l2_loss(layer2)

    #loss_reg = (loss_reg1 + loss_reg2)

    #loss = tf.reduce_sum(loss1 + mylambda*loss_reg)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train_prediction = tf.sigmoid(logits)

    valid_prediction = model(valid_data)
    valid_prediction = tf.sigmoid(valid_prediction)
    saver = tf.train.Saver()


num_steps = 10001

with tf.Session(graph=mygraph, config=tf.ConfigProto(log_device_placement=True)) as s:
    tf.global_variables_initializer().run()

    #saver.restore(s, "model/conv_net.ckpt")
    #print("Model Restored")


    plt.ion()

    for step in range(num_steps):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        #offset = (step * batch_size) % (train[1].shape[0] - batch_size)
        # Generate a minibatch.
        #batch_data = train[0][offset:(offset + batch_size), :]
        #batch_labels = train[1][offset:(offset + batch_size)]

        o, l, p, v = s.run([optimizer, loss, train_prediction, valid_prediction], feed_dict = {train_dataset : batch_xs, train_labels : batch_ys , valid_data : mnist.test.images})
        if step%40 == 0:
            acc_v = accuracy(v, mnist.test.labels)
            print (o, l, step, "Mini: " + str(accuracy(p, batch_ys)) , "Valid: " + str(acc_v))
            plt.figure(1)
            plt.scatter(step, l)
            plt.figure(2)
            plt.scatter(step, 100.0 - acc_v)
            plt.pause(0.05)
    save_path = saver.save(s, "model/neural_net.ckpt")
    print("Model saved in file %s" % save_path)
