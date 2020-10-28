import tensorflow as tf
import numpy as np

N = 100
x_zeros = np.random.multivariate_normal(
    mean = np.array((-1, -1)), cov = .1 * np.eye(2), size = (N//2,))
y_zeros = np.zeros((N//2,))
x_ones = np.random.multivariate_normal(
    mean = np.array((1, 1)), cov = .1 * np.eye(2), size = (N//2,))
y_ones = np.ones((N//2,))
x_np = np.vstack([x_zeros, x_ones])
y_np = np.concatenate([y_zeros, y_ones])
learning_rate = .01

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (N, 2))
    y = tf.placeholder(tf.float32, (N,))
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal((2, 1)))
    b = tf.Variable(tf.random_normal((1,)))
with tf.name_scope("prediction"):
    y_logit = tf.squeeze(tf.matmul(x, W) + b)
    y_one_prob = tf.sigmoid(y_logit)
    y_pred = tf.round(y_one_prob)
with tf.name_scope("loss"):
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit, labels = y)
    l = tf.reduce_sum(entropy)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()
    
train_writer = tf.summary.FileWriter("./logistic-train", tf.get_default_graph())

n_steps = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {x: x_np, y: y_np}
    for i in range(n_steps):
        _, summary, loss = sess.run([train_op, merged, l], feed_dict = feed_dict)
        print("step %d, loss: %f" % (i, loss))
        train_writer.add_summary(summary, i)
        pass
