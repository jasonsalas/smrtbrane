import numpy as np

''' SIGMOID FUNCTION '''
def nonlin(x, derivative=False):
	if(derivative == True):
		return x*(1-x)
	return 1/(1+np.exp(-x))

''' INPUT DATASET '''
X = np.array([ [0,0,1], [0,1,1], [1,0,1], [1,1,1] ])

''' OUTPUT DATASET '''
Y = np.array([ [0,0,1,1] ]).T  # transposed matrix of the output data (the expected values)

# seed random numbers to make
# calculation deterministic (good practice)
np.random.seed(1)

# initialize weights randomly with mean '0'
syn0 = 2 * np.random.random((3,1)) - 1

for iter in xrange(1000):

	# forward propagation
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))	# prediction function

	# how badly did we miss the mark?
	l1_error = Y - l1

	# multiply the amount missed by the slope (derivative) of the sigmoid for the values of l1
	l1_delta = l1_error * nonlin(l1,True)

	# update the weights
	syn0 += np.dot(l0.T, l1_delta)

print 'Output after training the model: '
print l1