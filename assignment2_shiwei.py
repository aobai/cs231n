# example for cs231n

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import functools 
import sys
sys.path.append('/Users/aobai/AI/course/cs231n/assignment2/cs231nassignment2virtural/lib/python3.6/site-packages')
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')

from tensorflow.examples.tutorials.mnist import input_data
from cs231n.data_utils  import load_CIFAR10


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:
	def __init__(self,X,y,is_training,global_step = 0,step = None):

		self.X = X
		self.y = y
		self.is_training = is_training
		self.step = step
		# self.global_step = global_step  is not None  else tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
		if global_step:
			self.global_step = global_step
		else:
			self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

		#
		self.prediction 
		self.optimizer
		self.error
		self.accuracy
		self.correct
		# self.network
		self.mean_loss

		#notes: 要把自变量放在前面初始化，否则会提示错误
		#具体原因待查

		# self.X = X
		# self.y = y
		# self.is_training = is_training
		# self.step = step

		# print(tf.get_default_graph().as_graph_def())

	@define_scope(initializer = tf.contrib.slim.xavier_initializer())
	def prediction(self):
	    wconv = tf.get_variable("wconv",shape = [7,7,3,32])
	    bconv = tf.get_variable("bconv",shape = [32])

	    #al.shape =  [none, 26,26,32] 
	    # (32 -7 + 1) / 1   = 26
	    a1 = tf.nn.conv2d(self.X,wconv,strides = [1,1,1,1],padding= "VALID") + bconv
	    h1 = tf.nn.relu(a1)
	    n1 = tf.layers.batch_normalization(h1, training=self.is_training)  # Using the tf.layers batch norm as its easier

	    #p1.shape = [none , ,16,32]
	    #out = ceil(float(26 - 2 + 1)/ 2)) = 13  13*13*32 = 5408
	    p1 = tf.nn.max_pool(n1,ksize = [1,2,2,1],strides = [1,2,2,1],padding='VALID')
	    flatten1 = tf.reshape(p1,[-1,5408])
	    waffine = tf.get_variable("waffine",shape = [5408,1024])
	    baffine = tf.get_variable('baffine',shape = [1024])

	    affine = tf.matmul(flatten1,waffine) + baffine
	    h2 = tf.nn.relu(affine)

	    waffine2 = tf.get_variable('affine2',[1024,10])
	    baffine2 = tf.get_variable('baffine2',[10])

	    output  = tf.matmul(h2,waffine2) + baffine2
	    print("output.shape:",output.shape)
	    return output

	# @define_scope(initializer = tf.contrib.slim.xavier_initializer())
	# def prediction(self):
	# 	return  self.network

	@define_scope
	def optimizer(self):
		opt = tf.train.RMSPropOptimizer(learning_rate = 0.001)
		# mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
		# 		(labels=tf.one_hot(self.y,10), logits=self.prediction))

		extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		# print("extra_update_ops:",extra_update_ops)
		with tf.control_dependencies(extra_update_ops):
		    return opt.minimize(self.mean_loss,global_step = self.global_step)

	@define_scope
	def error(self):
		pass
		
	@define_scope
	def mean_loss(self):
		return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
				(labels=tf.one_hot(self.y,10), logits=self.prediction))

	@define_scope
	def correct(self):
		return tf.equal(tf.argmax(self.prediction,1),self.y)

	@define_scope
	def accuracy(self):
		return tf.reduce_mean(tf.cast(self.correct,tf.float32))



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    print("mask:",mask)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_summary(loss,accuracy):
	with tf.name_scope('summaries_tests'):
		tf.summary.scalar('loss',loss)
		tf.summary.scalar('accuracy',accuracy)
		tf.summary.histogram('histogram loss:',loss)
		tf.summary.histogram('histogram accuracy:',accuracy)

		return tf.summary.merge_all()


def run_model(X,y,is_training,model,Xd,yd,epochs,batch_size,print_every,training_now,plot_loss = False):
	'''
	test
	'''

	#这里不能够用default， 否则saver 报错
	# tf.reset_default_graph()

	with tf.Session() as sess:

		#save 之前必须要用变量，否则报错
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())

		writer = tf.summary.FileWriter('graphs',sess.graph)
		ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
		# if ckpt and ckpt.model_checkpoint_path:
		# 	saver.restore(sess,ckpt.model_checkpoint_path)

		#don't use initial_step right now 
		# initial_step = model.global_step.eval()
		initial_step = 0
		print("initial_step:",initial_step)

		x_size = Xd.shape[0]

		x_indicies = np.arange(x_size)
		np.random.shuffle(x_indicies)

		summary_opt = create_summary(model.mean_loss,model.accuracy)

		if training_now:
			variables = [summary_opt,model.mean_loss,model.correct,model.optimizer]
		else:
			variables = [summary_opt,model.mean_loss,model.correct,model.accuracy]

		iter_cnt = 0	

		#这里应该把epochs * size 转换为一次
		for e in range(epochs):
			correct = 0
			losses  = []
			start_time = time.time()

			for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
				start_idx = (i*batch_size)%x_size
				idx = x_indicies[start_idx:start_idx+batch_size]

				feed_dict = { X: Xd[idx,:],
							  y: yd[idx],
							  is_training: training_now}

				actual_batch_size = yd[idx].shape[0]
				
				summary,loss, corr, _ = sess.run(variables,feed_dict = feed_dict)
				writer.add_summary(summary,global_step = iter_cnt)

				losses.append(loss * actual_batch_size)
				correct += np.sum(corr)

				#record  variables every  print_every times(100) 
				if training_now and (iter_cnt % print_every) == 0:
					print("Iteration{0}: with minibatch training loss = {1:.3g} and accuracy of:{2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
					saver.save(sess,'checkpoints/tensorflow_test',iter_cnt)


				iter_cnt += 1

				# saver.save(sess,'checkpoints/tensorflow_test',)

			total_correct = correct / x_size
			total_loss    = np.sum(losses) / x_size
			print("Epoch:{2},Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,e+1))
			print('Time consuming every epoch: {}'.format(time.time() - start_time))

			# if plot_loss:
			# 	plt.plot(losses)
			# 	plt.grid(True)
			# 	plt.title('Epoch {} Loss'.format(e+1))
			# 	plt.xlabel('minibatch number')
			# 	plt.ylabel('minibatch loss')
			# 	plt.show()


def main():
	X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
	print('Train data shape: ', X_train.shape)
	print('Train labels shape: ', y_train.shape)
	print('Validation data shape: ', X_val.shape)
	print('Validation labels shape: ', y_val.shape)
	print('Test data shape: ', X_test.shape)
	print('Test labels shape: ', y_test.shape)

	with  tf.variable_scope('input_test') as scope:
		#可有可无的
		# tf.reset_default_graph()
		X = tf.placeholder(tf.float32,[None,32,32,3],name ="placeholder_x")
		y = tf.placeholder(tf.int64,[None],name = 'placeholder_y')
		is_training = tf.placeholder(tf.bool,name = 'placeholder_is_training')

	make_dir('checkpoints')
	make_dir('outputs')
	model = Model(X,y,is_training)

	print("Training.......")
	#在assignment2 中， 用train_step当作是否正在训练的标示，其实train_step 的返回值时一个optimizer
	#所以这里直接用true替代了train_step
	# (session,model,Xd,yd,,batch_size,print_every,training_now,plot_loss):

	#这里必须显示传进去，也是有问题的，正常情况不应该传进去呀
	run_model(X,y,is_training,model,X_train,y_train,1,64,100,True,True)


	print("Validation......")
	# run_model(X,y,is_training,model,X_val,y_val,1,64,100,False,False)







if __name__ == '__main__':
  main()


























