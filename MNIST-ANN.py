import tensorflow as tf
from tensorflow.examples.tutorials import mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist =input_data.read_data_sets("MNIST/data",one_hot=True)
mnist.train.images
mnist.train.num_examples
mnist.test.num_examples
mnist.validation.num_examples

import matplotlib.pyplot as  plt
mnist.train.images[1].shape
plt.imshow(mnist.train.images[1].reshape(28,28))
plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')
plt.imshow(mnist.train.images[1].reshape(784,1),cmap='gist_gray',aspect=0.02)

#creating Model
x = tf.placeholder(tf.float32,shape=[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W)+b

y_true = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

#creating session
init=  tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        batch_x,batch_y =mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
    #tf.argmax(y,1) #returns index postion of a label with highest probability . second argument is axis=1 .Gives predicted label
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1)) #for all predictions we get True/False values
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #reduce_mean gives average
    #PREDICTED [3,4] TRUE [3,9]
    #[True,False]
    #[1,0]  (Casting)
    #0.5 (Average)
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))















