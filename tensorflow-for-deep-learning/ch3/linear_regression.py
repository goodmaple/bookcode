import tensorflow as tf
import numpy as np

N = 100
w_true = 5
b_true = 2
noise_scale = .1
x_np = np.random.rand(N, 1)
noise = np.random.normal(scale=noise_scale, size = (N, 1))
y_np = np.reshape(w_true * x_np + b_true + noise, (-1))
learning_rate = .001

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N, 1))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal((1, 1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
    y_pred = tf.matmul(x, W) + b
with tf.name_scope("loss"):
    l = tf.reduce_mean((y - y_pred) ** 2)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("./lr-train", tf.get_default_graph())

n_steps = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: x_np, y: y_np}
    for i in range(n_steps):
        _, summary, loss = sess.run([train_op, merged, l], feed_dict = feed_dict)
        print("step %d, loss: %f" % (i, loss))
        train_writer.add_summary(summary, i)