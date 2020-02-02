import tensorflow as tf 
import numpy as np 
from sklearn.datasets import fetch_mldata
import datapreprocess
import time





def batch_generator(X,y, batch_size=100, shuffle=False, random_seed=None):
	idx = np.arange(y.shape[0])
	if shuffle:
		rng = np.random.RandomState(random_seed)
		rng.shuffle(idx)
		X = X[idx]
		y = y[idx]

	for i in range(0, X.shape[0], batch_size):
		yield(X[i:i+batch_size,:], y[i:i+batch_size])



class convNN(object):
 	def __init__ (self, batchsize=100, epochs=20, learning_rate=1e-4, dropout_rate=0.5, shuffle=True, random_seed=None):
 		np.random.seed(random_seed)
 		self.batchsize = batchsize
 		self.epochs = epochs
 		self.learning_rate = learning_rate
 		self.dropout_rate = dropout_rate
 		self.shuffle = shuffle

 		g = tf.Graph()
 		with g.as_default():
 			tf.set_random_seed(random_seed)
 			self.build()
 			self.init_op = tf.global_variables_initializer()
 			self.saver = tf.train.Saver()

 		self.sess = tf.Session(graph=g)



 	def build (self):
 		tf_x = tf.placeholder(tf.float32, shape=[None, 784], name = 'tf_x')
 		tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')
 		is_train = tf.placeholder(tf.bool, shape=(), name = 'is_train')

 		tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1], name ='input_x_2dimages')
 		tf_y_onehot = tf.one_hot(indices=tf_y, depth=10, dtype=tf.float32, name='input_y_onehot')


 		input_shape = tf_x_image.get_shape().as_list()
 		print("Shape1: ", input_shape)



 		h1 = tf.layers.conv2d(tf_x_image, kernel_size = (3,3),strides = (1,1),filters=4, activation=tf.nn.relu)
 		input_shape = h1.get_shape().as_list()
 		print("Shape1: ", input_shape)

 		h1_pool = tf.layers.max_pooling2d(h1, pool_size =(2,2), strides=(2,2))
 		input_shape = h1_pool.get_shape().as_list()
 		print("Shape1: ", input_shape)


 		h2 = tf.layers.conv2d(tf_x_image, kernel_size=(3,3), filters=2, strides = (3,3),activation = tf.nn.relu)
 		input_shape = h2.get_shape().as_list()
 		print("Shape1: " ,input_shape)

 		h2_pool = tf.layers.max_pooling2d(h2, pool_size=(4,4), strides=(4,4))
 		input_shape = h2_pool.get_shape().as_list()
 		print("Shape1: " ,input_shape)

 		n_input_units =  np.prod(input_shape[1:])
 		h2_pool_flat = tf.reshape(h2_pool, shape=[-1,n_input_units])
 		h3 = tf.layers.dense(h2_pool_flat, 10, activation = None)
 		input_shape = h3.get_shape().as_list()
 		print("Shape1: " ,input_shape)

 		#h3_drop = tf.layers.dropout(h3, rate=self.dropout_rate, training = is_train)

 		#h4 = tf.layers.dense(h3_drop, 10, activation=None)
 		#print(h4)

 		predictions = { 'probabilities': tf.nn.softmax(h3, name='probabilities'),
 		                 'labels': tf.cast(tf.argmax(h3, axis=1), tf.int32, name='labels')}

 		cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels = tf_y_onehot), name = 'cross_entropy_loss')


 		optimizer = tf.train.AdamOptimizer(self.learning_rate)
 		optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')

 		correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
 		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')



 	def train(self, training_set, validation_set=None, initialize=True):

 		if initialize:
 			self.sess.run(self.init_op)

 		self.train_cost_ =[]
 		X_data = np.array(training_set[0])
 		y_data = np.array(training_set[1])

 		for epoch in range(1,self.epochs+1):
 			batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
 			avg_loss =0.0
 			for i, (batch_x, batch_y) in enumerate (batch_gen):
 				feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'is_train:0': True}
 				loss, _ =self.sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict = feed)
 				avg_loss += loss

 			print('Epoch %02d: Training Avg. Loss: ''%7.3f' % (epoch, avg_loss), end='')
 			if validation_set is not None:
 				feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'is_train:0': False}
 				valid_acc = self.sess.run('accuracy:0', feed_dict = feed)
 				print('Validation Acc: %7.3f' % valid_acc)

 			else:
 				print()



 	def predict(self, X_test, return_proba=False):
 		feed = {'tf_x:0': X_test, 'is_train:0': False}
 		if return_proba:
 			return self.sess.run('probabilities:0', feed_dict=feed)
 		else:
 			return self.sess.run('labels:0', feed_dict=feed)


 	def save(self, epoch, path ='./tflayers-model/'):
 		if not os.path.isdir (path):
 			os.makedirs(path)

 		print('Saving model in %s' % path)
 		self.saver.save(self.sess, os.path.join(path, 'model.ckpt'), global_step=epoch)


 	def load(self, epoch, path):
 		print('Loading model from %s' % path)
 		self.saver.restore(self.sess, os.path.join(path, 'model.ckpt-%d' % epoch))





mnist = fetch_mldata('MNIST original')
X_mnist, y_mnist = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = datapreprocess.split_data(X_mnist, y_mnist)
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)
X_train_centered = (X_train - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/ std_val




start_time = time.time()
cnn  = convNN(random_seed=123)
cnn.train(training_set=(X_train_centered, y_train))
total_time = time.time() - start_time
print(cnn.predict(X_test_centered[:10,:]))
preds = cnn.predict(X_test_centered)
print('Test accuracy: %.2f%%' % (100 * np.sum(y_test==preds) / len(y_test)))
print('Running time: ', total_time)














