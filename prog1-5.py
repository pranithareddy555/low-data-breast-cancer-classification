import tensorflow as tf
import numpy as np

#set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
BATCH_SIZE=32
EPOCHS=500
#specify path to training data and testing data
train_x_location = "x_train_5.csv"
train_y_location = "y_train_5.csv"
test_x_location = "x_test.csv"
test_y_location = "y_test.csv"


print("Reading training data")
x_train = np.loadtxt(train_x_location, dtype="uint8", delimiter=",")
y_train = np.loadtxt(train_y_location, dtype="uint8", delimiter=",")

m, n = x_train.shape # m training examples, each with n features
m_labels, = y_train.shape # m2 examples, each with k labels
l_min = y_train.min()
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
0.007,
decay_steps=np.ceil(m / BATCH_SIZE)*EPOCHS,
decay_rate=0.5,
staircase=False)
assert m_labels == m, "x_train and y_train should have same length."
assert l_min == 0, "each label should be in the range 0 - k -1."
k = y_train.max()+1

print(m, "examples,", n, "features,", k, "categories.")

#define the training model
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(20, activation=tf.keras.activations.relu, kernel_initializer='normal', input_shape=(n,)),
tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',moving_mean_initializer='zeros',moving_variance_initializer='ones'),

tf.keras.layers.Dense(20, activation=tf.keras.activations.relu),
tf.keras.layers.Dropout(0.3),
#tf.keras.layers.Dense(5, activation=tf.keras.activations.elu),
tf.keras.layers.Dense(k, activation=tf.keras.activations.linear),
tf.keras.layers.Dense(k, activation=tf.keras.activations.softmax)
])

##loss = 'categorical_entropy' expects input to be one-hot encoded
#loss = 'sparse_categorical_entropy' expects input to be the category as a number
#options for optimizer: 'sgd' and 'adam'. sgd is stochastic gradient descent
model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

print("Train")
model.fit(x_train, y_train, epochs=500, batch_size=32)

#default batch size is 32
print("Reading testing data")
x_test = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

m_test, n_test = x_test.shape
m_test_labels, = y_test.shape
l_min = y_train.min()

assert m_test_labels == m_test, "x_test and y_test should have same length."
assert n_test == n, "train and x_test should have same number of features."

print(m_test, "test examples.")

print("Evaluate")
model.evaluate(x_test, y_test)




