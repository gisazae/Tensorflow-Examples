## With KDD 2015 Intrusion Detection 
#!/usr/bin/env python
import tensorflow as tf
msg = tf.constant("Test Dataset TF")
sess = tf.Session()
print(sess.run(msg))
x = tf.constant(9)
y = tf.constant(2)
print(sess.run(x + y))
# Based on https://github.com/albahnsen/ML_SecurityInformatics
import pandas as pn
pn.set_option('display.max_columns', 500)
import zipfile
# DATASET NSL-KDD from http://www.unb.ca/cic/research/datasets/nsl.html
with zipfile.ZipFile('../datasets/UNB_ISCX_NSL_KDD.csv.zip', 'r') as z:
    f = z.open('UNB_ISCX_NSL_KDD.csv')
    data = pn.io.parsers.read_table(f, sep=',')
data.head()
print(data)
y.value_counts()
train_X = data[['same_srv_rate','dst_host_srv_count']]
print(train_X)
train_Y = (data['class'] == 'anomaly').astype(int)
print(train_Y)
## Aplying tensores con linear_regression
## .................
# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 56

# Training Data
n_samples = train_X.shape[0]

# tf Graph Input (Tensorflow Graph)
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights (W and Bias)
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Calculate the MSE Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
