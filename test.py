import scipy.io as so
import numpy as np
import tensorflow as tf
import cv2
import cPickle
import extract

f = open("../../Downloads/mnist.pkl")

_, _, test = cPickle.load(f)

imgs = extract.getImage('sample_images/numbers1.jpg')

def accuracy(predictions, labels):
  arr = np.argmax(predictions, axis=1)
  return 100*np.mean(arr == labels)

digit = tf.placeholder(shape=[1, 784], dtype=tf.float32)

layer1 = tf.Variable(tf.truncated_normal([784, 800] , dtype=tf.float32), name="layer1")
layer1_b = tf.Variable(tf.zeros([800]), name="layer1_b")
layer2 = tf.Variable(tf.truncated_normal([800, 10] , dtype=tf.float32), name="layer2")
layer2_b = tf.Variable(tf.zeros([10]), name="layer2_b")


def model(data):
    a2 = tf.sigmoid((tf.matmul(data, layer1) + layer1_b))
    a3 = tf.sigmoid(tf.matmul(a2, layer2) + layer2_b)


    return a3

test_prediction = model(test[0])
digit_prediction = model(digit)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "model/neural_net.ckpt")
    print("Model Restored")
    print layer1.eval().shape
    print layer1_b.eval().shape
    print layer2.eval().shape
    print layer2_b.eval().shape
    for i in imgs:
        i = i/255.0
        cv2.imshow('im', i)
        
        i = np.reshape(i.ravel(), [1, 784])
        print i.shape
        p = sess.run(digit_prediction, feed_dict= {digit : i})
        print np.argmax(p)
        cv2.waitKey(0)


    p = sess.run(test_prediction)
    print(accuracy(p, test[1]))
