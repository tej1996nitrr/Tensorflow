import tensorflow as tf
import numpy as np
n_features=10
n_dense_neurons=3
x = tf.placeholder(tf.float32,(None,n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))
xW = tf.matmul(x,W)
z = tf.add(xW,b)
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a,feed_dict={x:np.random.random([1,n_features])})

print(layer_out)

X_data = np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
X_data
y_label = np.linspace(0,10,10)+ np.random.uniform(-1.5,1.5,10)
import  matplotlib.pyplot as plt
plt.plot(X_data,y_label,'*')
np.random.rand(2)
m = tf.Variable(0.02)
b = tf.Variable(0.05)
error =  0
for x,y in zip(X_data,y_label):
    y_hat = m*x+b
    error+=(y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init =tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    epochs =  10
    for i in range(epochs):
        sess.run(train)
    final_slope,final_intercept=sess.run([m,b])
final_intercept
final_slope
x_test = np.linspace(-1,11,10)
y_pred_plot = final_slope*x_test + final_intercept
plt.plot(x_test,y_pred_plot,'r')

plt.plot(X_data,y_label,'*')