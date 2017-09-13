import numpy as np
import tensorflow as tf
import cv2
import extract
import argparse

parser = argparse.ArgumentParser(description='To read some arguments')

parser.add_argument('-i', "--image", help="Path to the test image", required=True)


args = vars(parser.parse_args())

imagepath = args["image"]



img = cv2.imread(imagepath)

imgs, coords, align = extract.getImage(imagepath)

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



mygraph = tf.Graph()

with mygraph.as_default():

    def conv2d(x, W):
      """conv2d returns a 2d convolution layer with full stride."""
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


    def max_pool_2x2(x):
      """max_pool_2x2 downsamples a feature map by 2X."""
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


    def weight_variable(shape):
      """weight_variable generates a weight variable of a given shape."""
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)


    def bias_variable(shape):
      """bias_variable generates a bias variable of a given shape."""
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)


    digit = tf.placeholder(shape=[1, 784], dtype=tf.float32)

    def deepnn(x):

        x_image = tf.reshape(x, [-1, 28, 28, 1])


        W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)


        h_pool2 = max_pool_2x2(h_conv2)


        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

        return y_conv


    digit_prediction = deepnn(digit)

    saver = tf.train.Saver()


with tf.Session(graph=mygraph) as sess:

    saver.restore(sess, "model/conv_net.ckpt")
    print("Model Restored")

    cv2.imshow("final detections", img)
    cv2.waitKey(0)
    for i, coord in zip(imgs, coords):
        i = i/255.0
        cv2.imshow('im', i)

        i = np.reshape(i.ravel(), [1, 784])

        p = sess.run(digit_prediction, feed_dict= {digit : i})

        print_on_image(img, np.argmax(p),coord)
