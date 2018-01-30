from __future__ import print_function, division
import numpy as np
import h5py
import scipy.io
import random
import sys,os
import itertools
import numbers
from collections import Counter
from warnings import warn
from abc import ABCMeta, abstractmethod

np.random.seed(1337)  # for reproducibility

from keras.optimizers import RMSprop, SGD
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.layers.core as core
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, multiply, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec

class Attention(Layer):

	def __init__(self,hidden,init='glorot_uniform',activation='linear',W_regularizer=None,b_regularizer=None,W_constraint=None,**kwargs):
	    self.init = initializers.get(init)
	    self.activation = activations.get(activation)
	    self.W_regularizer = regularizers.get(W_regularizer)
	    self.b_regularizer = regularizers.get(b_regularizer)
	    self.W_constraint = constraints.get(W_constraint)
	    self.hidden=hidden
	    super(Attention, self).__init__(**kwargs)
	    
	def build(self, input_shape):
	    input_dim = input_shape[-1]
	    self.input_length = input_shape[1]
	    self.W0 = self.add_weight(name ='{}_W1'.format(self.name), shape = (input_dim, self.hidden), initializer = 'glorot_uniform', trainable=True) # Keras 2 API
	    self.W  = self.add_weight( name ='{}_W'.format(self.name),  shape = (self.hidden, 1), initializer = 'glorot_uniform', trainable=True)
	    self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
	    self.b  = K.zeros((1,), name='{}_b'.format(self.name))
	    self.trainable_weights = [self.W0,self.W,self.b,self.b0]
	    
	    self.regularizers = []
	    if self.W_regularizer:
	        self.W_regularizer.set_param(self.W)
	        self.regularizers.append(self.W_regularizer)
	    
	    if self.b_regularizer:
	        self.b_regularizer.set_param(self.b)
	        self.regularizers.append(self.b_regularizer)
	    
	    self.constraints = {}
	    if self.W_constraint:
	        self.constraints[self.W0] = self.W_constraint
	        self.constraints[self.W] = self.W_constraint
	        
	    super(Attention, self).build(input_shape)
	    
	def call(self,x,mask=None):
	        attmap = self.activation(K.dot(x, self.W0)+self.b0)
	        attmap = K.dot(attmap, self.W) + self.b
	        attmap = K.reshape(attmap, (-1, self.input_length)) # Softmax needs one dimension
	        attmap = K.softmax(attmap)
	        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
	        out = K.concatenate([dense_representation, attmap]) # Output the attention maps but do not pass it to the next layer by DIY flatten layer
	        return out


	def compute_output_shape(self, input_shape):
	    return (input_shape[0], input_shape[-1] + input_shape[1])

	def get_config(self):
	    config = {'init': 'glorot_uniform',
	              'activation': self.activation.__name__,
	              'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
	              'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
	              'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
	              'hidden': self.hidden if self.hidden else None}
	    base_config = super(Attention, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer): # Based on the source code of Keras flatten
	def __init__(self, keep_dim, **kwargs):
	    self.keep_dim = keep_dim
	    super(attention_flatten, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
	    if not all(input_shape[1:]):
	        raise Exception('The shape of the input to "Flatten" '
	                        'is not fully defined '
	                        '(got ' + str(input_shape[1:]) + '. '
	                        'Make sure to pass a complete "input_shape" '
	                        'or "batch_input_shape" argument to the first '
	                        'layer in your model.')
	    return (input_shape[0], self.keep_dim)   # Remove the attention map

	def call(self, x, mask=None):
	    x=x[:,:self.keep_dim]
	    return K.batch_flatten(x)


def _bootstrap(max_samples, X, y):
	"""Private function used to build a batch of estimators within a job."""
	# Retrieve settings
	#print X.shape
	n_samples, n_features1, n_features2 = X.shape

	if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
	        (0.0 < max_samples <= 1.0)):
	    max_samples = int(max_samples * n_samples)

	indices = np.random.randint(0, n_samples, max_samples)
	#sample_counts = bincount(indices, minlength=n_samples)

	# TEF: Main change in this next call to _downsample
	(Xbal, ybal) = _downsample(X[indices], y[indices])
	#samples = sample_counts > 0.

	return Xbal, ybal


# TEF defined this simple downsampling routine.
def _downsample(X, y):
	#X = np.array(X)
	#y = np.array(y)
	#y = y.T[0]
	y = y.T
	ccounts = Counter(y)                                # Get class counts
	if len(ccounts) != 2:
	    raise ValueError("BlaggingClassifier only implemented for 2 classes")
	# Identify minority and majority classes
	classes = [cls for cls in ccounts.keys()]
	if ccounts[classes[1]] < ccounts[classes[0]]:
	    minority_class = classes[1]
	else:
	    minority_class = classes[0]
	# Get indices of each class
	n_minority = ccounts[minority_class]
	indices = np.array(range(len(X)))
	minority_egs_idx = indices[y==minority_class]
	majority_egs_idx = indices[y!=minority_class]
	# Trim majority class so they're balanced
	majority_egs_idx = majority_egs_idx[:n_minority]
	total_egs_idx = np.append(minority_egs_idx, majority_egs_idx)
	# Extract examples of each class and combine
	X = X[total_egs_idx]
	y = y[total_egs_idx]
	return (X, y)

def load_and_data():
	print('Loading data...')

	X_pos_train = np.load('data/GD_MSE_train_pos.npy')
	y_pos_train = np.ones(len(X_pos_train))

	X_neg_train = np.load('data/GD_MSE_train_neg.npy')
	y_neg_train = np.zeros(len(X_neg_train))

	X_train = np.concatenate([X_pos_train,X_neg_train])
	y_train = np.concatenate([y_pos_train,y_neg_train])

	train_index = range(len(X_train))
	np.random.shuffle(train_index)
	X_train = X_train[train_index]
	y_train = y_train[train_index]
	X_train, y_train = _bootstrap(max_samples=1.0, X=X_train, y=y_train)

	X_vali = np.load('data/X_valid.npy')
	y_vali = np.load('data/y_valid.npy')

	return X_train, y_train, X_vali, y_vali

def set_up_model_up():
	print('building model')

	seq_input_shape = (2000,4)
	nb_filter = 64
	filter_length = 6
	input_shape = (2000,4)
	attentionhidden = 256

	seq_input = Input(shape = seq_input_shape, name = 'seq_input')
	convul1   = Convolution1D(filters = nb_filter,
                        	  kernel_size = filter_length,
                        	  padding = 'valid',
                        	  activation = 'relu',
                        	  kernel_constraint = maxnorm(3),
                        	  subsample_length = 1)

	pool_ma1 = MaxPooling1D(pool_size = 3)
	dropout1 = Dropout(0.5977908689086315)
	dropout2 = Dropout(0.30131233477637737)
	decoder  = Attention(hidden = attentionhidden, activation = 'linear')
	dense1   = Dense(1)
	dense2   = Dense(1)

	output_1 = pool_ma1(convul1(seq_input))
	output_2 = dropout1(output_1)
	att_decoder  = decoder(output_2)
	output_3 = attention_flatten(output_2._keras_shape[2])(att_decoder)

	output_4 =  dense1(dropout2(Flatten()(output_2)))
	all_outp =  merge([output_3, output_4], mode = 'concat')
	output_5 =  dense2(all_outp)
	output_f =  Activation('sigmoid')(output_5)

	model = Model(inputs = seq_input, outputs = output_f)
	model.compile(loss = 'binary_crossentropy', optimizer = 'nadam', metrics = ['accuracy'])

	print (model.summary())
	return model

def train_model(n_estimators = 16):
	for i in range(n_estimators):
		X_train, y_train, X_vali, y_vali = load_and_data()

		print ('X_train.shape', X_train.shape)
		print ('X_vali.shape', X_vali.shape)
		
		batch_size = 64
		nb_epoch = 20

		model = set_up_model_up()

		checkpointer = ModelCheckpoint(filepath = 'model/bestmodel_split_chr_GD_'+ str(i) + '.hdf5', verbose = 1, save_best_only = True)
		earlystopper = EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1)  #####################################################

		print('Train...')
		model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epoch, 
			validation_data = (X_vali, y_vali), callbacks=[checkpointer,earlystopper])


def test(n_estimators = 16):

		model = set_up_model_up()

		X_test = np.load('data/X_test.npy')
		y_test = np.load('data/y_test.npy')

		ensemble = np.zeros(len(X_test))

		for i in range(n_estimators):
			print ('testing', i, 'model')

			model.load_weights('model/bestmodel_split_chr_GD_'+ str(i) + '.hdf5')

			print ('Predicting...')
			y_score = model.predict(X_test, verbose = 1, batch_size = 512)
			y_pred = []
			for item in y_score:
			        y_pred.append(item[0])
			y_pred =  np.array(y_pred)
			ensemble += y_pred

			auroc = roc_auc_score(y_test, y_pred)
			aupr = average_precision_score(y_test, y_pred)

			print ('auroc', auroc)
			print ('aupr' , aupr)

		ensemble /= n_estimators

		np.save('test_result/y_test', y_test)
		np.save('test_result/y_pred', ensemble)

		auroc = roc_auc_score(y_test, ensemble)
		aupr  = average_precision_score(y_test, ensemble)

		print ('auroc', auroc)
		print ('aupr' , aupr)

	

if __name__ == '__main__':
	set_up_model_up()
 	train_model(n_estimators = 16)
	test(n_estimators = 16)
