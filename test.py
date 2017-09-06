import scipy.io as so
import numpy as np
import tensorflow as tf
import cv2
import cPickle
import extract

f = open("../../Downloads/mnist.pkl")

_, _, test = cPickle.load(f)
img = cv2.imread('sample_images/numbers.jpg')

imgs, coords, align = extract.getImage('sample_images/numbers.jpg')

def accuracy(predictions, labels):
  arr = np.argmax(predictions, axis=1)
  return 100*np.mean(arr == labels)

def print_on_image(img, number, (x,y,w,h)):
    if align == 'x':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(number), (x , y + 155), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    elif align == 'y':
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(number), (x + 155, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("final detections", img)
    cv2.waitKey(0)

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
arr = []
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "model/neural_net.ckpt")
    print("Model Restored")
    print layer1.eval().shape
    print layer1_b.eval().shape
    print layer2.eval().shape
    print layer2_b.eval().shape
    cv2.imshow("final detections", img)
    cv2.waitKey(0)
    for i, coord in zip(imgs, coords):
        i = i/255.0
        cv2.imshow('im', i)

        i = np.reshape(i.ravel(), [1, 784])
        #print i.shape
        p = sess.run(digit_prediction, feed_dict= {digit : i})
        #print np.argmax(p)

        print_on_image(img, np.argmax(p),coord)

    p = sess.run(test_prediction)
    print(accuracy(p, test[1]))
