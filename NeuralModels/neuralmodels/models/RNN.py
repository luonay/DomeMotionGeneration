from headers import *

class RNN(object):
	def __init__(self,layers,cost,Y,learning_rate,update_type=RMSprop(),clipnorm=0.0):
		self.settings = locals()
		del self.settings['self']
		self.layers = layers
		self.L2_sqr = theano.shared(value=np.float32(0.0))
		for i in range(1, len(layers)):
			layers[i].connect(layers[i-1])
			self.L2_sqr += layers[i].L2_sqr  

		self.X = layers[0].input
		self.Y_pr = layers[-1].output()
		self.Y = Y

		self.cost = 0.0000*self.L2_sqr + cost(self.Y_pr,self.Y)
	        self.params = []
		for l in self.layers:
	                if hasattr(l,'params'):
				self.params.extend(l.params)

		self.num_params = 0
		for par in self.params:
			val = par.get_value()
			temp = 1
			for i in range(val.ndim):
				temp *= val.shape[i]		
			self.num_params += temp
			
		#rmsprop = RMSprop()
		[self.updates,self.grads] = update_type.get_updates(self.params,self.cost)

		self.train = theano.function([self.X,self.Y],self.cost,updates=self.updates)
		self.objective = theano.function([self.X,self.Y],self.cost)
		self.predict = theano.function([self.X],self.layers[-1].output())
	
	def fitModel(self,trX,trY,valX, valY, snapshot_rate=1,path=None,epochs=30,batch_size=50,learning_rate_decay=0.97,decay_after=10):
		from neuralmodels.loadcheckpoint import save
		X_minibatch=[]
		Y_minibatch=[]
		num_examples = trX.shape[1]
		#print('num_examples:{0}'.format(num_examples))
		#print('batch_size:{0}'.format(batch_size))
		batches_in_one_epoch = int(num_examples / batch_size)
		#print('batches in one epoch:{0}'.format(batches_in_one_epoch))	
		loss_values = []
		for epoch in range(epochs):
			t0 = time.time()
			permutation = permute(num_examples)
			if self.X.ndim > 2:
				trX = trX[:,permutation,:]
			else:
				trX = trX[:,permutation]
			
			if self.Y.ndim > 2:
				trY = trY[:,permutation,:]
			elif self.Y.ndim > 1:
				trY = trY[:,permutation]
			else:
				trY = trY[permutation]
			#print('size trX:{0}'.format(trX.shape))
			#print('size trY:{0}'.format(trY.shape))
			for j in range(batches_in_one_epoch):
				if self.X.ndim > 2:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size,:]
				else:
					X_minibatch = trX[:,j*batch_size:(j+1)*batch_size]
				if self.Y.ndim > 2:
					Y_minibatch = trY[:,j*batch_size:(j+1)*batch_size,:]
				elif self.Y.ndim > 1:
					Y_minibatch = trY[:,j*batch_size:(j+1)*batch_size]
				else:
					Y_minibatch = trY[j*batch_size:(j+1)*batch_size]
				#print('size X_minibatch:{0}'.format(X_minibatch.shape))
				#print('size Y_minibatch:{0}'.format(Y_minibatch.shape))
				loss = self.train(X_minibatch,Y_minibatch)
				#print('loss:{0}'.format(loss))
				loss_values.append(loss)
				#print "epoch={0} loss={1}".format(epoch,loss)
			#validation loss
			print "epoch={0} loss={1}".format(epoch, loss_values[-1])
			val_loss = self.objective(valX, valY)
			print("----------------------------validation loss={0}".format(val_loss))	
				
			if path and epoch % snapshot_rate == 0:
				print 'saving snapshot checkpoint.{0}'.format(epoch)
				save(self,"{0}checkpoint.{1}".format(path,epoch))
				f = open('{0}logfile'.format(path),'w')
				for v in loss_values:
					f.write('{0}\n'.format(v))
				f.close()
			t1 = time.time()
			print 'Epoch took {0} seconds'.format(t1-t0)

			#if epoch > decay_after:
			#	self.learning_rate *= learning_rate_decay 


	def predict_output(self,teX,predictfn):
		prediction = self.predict(teX)
		shape = prediction.shape
		if prediction.ndim > 2:
			# prediction dim = T x N x D
			# Sequence prediction
			prediction = prediction.reshape(shape[0]*shape[1],shape[2])
			prediction = predictfn(prediction)
			prediction = prediction.reshape(shape[0],shape[1])
			# Output dim = T x N
		else:
			# prediction dim = N x D
			# Single prediction at the end of sequence
			prediction = predictfn(prediction)
			# Output dim = N
		return prediction

	def predict_motion_sequence(self, teX, seq_length):
		future_sequence = []
		for i in range(seq_length):
			prediction = self.predict(teX)
			prediction = prediction[-1]
			teX = np.append(teX, [prediction],axis=0)
			
			future_sequence.append(prediction)
		return np.array(future_sequence)

	def predict_language_model(self,teX,seq_length,predictfn,Temperature=0.7):
		# Currently only supported with language modeling
		predict = theano.function([self.X],self.layers[-1].output(Temperature))
		future_sequence = []
		for i in range(seq_length):
			prediction = predict(teX)
			prediction = prediction[-1]
			prediction = predictfn(prediction)
			teX = np.append(teX,[prediction],axis=0)
			future_sequence.append(prediction)
		return np.array(future_sequence)
		# dim = seq_length x N

