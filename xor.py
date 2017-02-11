# Quick and dirty XOR 2-3-1 example network
# for an example on /r/MLQuestions
# Vanilla gradient descent with no momentum
# Adam Smith 1.26.2017
import numpy as np

# X are the inputs, Y are the outputs (the XOR truth table)
X = np.array([[[1],[1]], [[0],[1]], [[1],[0]], [[0],[0]]])
Y = np.array([0, 1, 1, 0])

#Learning rate divided by batch size
learn = 0.5
batchSize = Y.shape[0] #4

# Weights from the inputs to the hidden layer
W0 = 0.2 * ( 2.0 * np.random.random((3,2)) - 1.0 )

# Hidden layer biases
B0 = np.zeros((3,1))

# Weights from the hidden layer to the output node
W1 = 0.2 * ( 2.0 * np.random.random((3,1)) - 1.0 )

# Output node bias
B1 = np.zeros((1,1))

# Training loop
for epoch in range(10000):
	dW0 = np.zeros((3,2))
	dW1 = np.zeros((3,1))
	dB0 = np.zeros((3,1))
	dB1 = np.zeros((1,1))
	
	# Batch loop
	for example in range(4):
		# Hidden Layer forward prop
		S0 = W0.dot(X[example]) + B0
		A0 = np.tanh(S0)
		
		# Output Layer forward prop
		S1 = W1.T.dot(A0) + B1
		
		# This is an unnecessary assignment I'm including for illustration
		# purposes only. I'm using a linear output node which makes the
		# activation function a no-op
		A1 = S1
		
		# Hidden to Output weights backprop
		# The variables called 'delta' are the error signals, which are
		# usually represented in the literature by Greek lowercase delta
		delta1 = A1 - Y[example]
		dW1 += -delta1 * A0
		dB1 += -delta1
		
		# Input to Hidden weights backprop
		# Because the hidden layer has tanh activation, the error term
		# has a derivative component ( 1 - A0 * A0 ) is the derivative
		# of A0 w.r.t. S0, i.e. ( 1 - tanh^2(S0) ) and tanh(S0) is just A0
		delta0 = -delta1 * ( 1 - A0 * A0 ) * W1
		dW0 += delta0.dot(X[example].T)
		dB0 += delta0
	
	# Weight updates after each batch
	W1 += learn / batchSize * dW1
	B1 += learn / batchSize * dB1
	W0 += learn / batchSize * dW0
	B0 += learn / batchSize * dB0
	
# Print the results for each example case
for example in range(4):
	S0 = W0.dot(X[example]) + B0
	A0 = np.tanh(S0)
	S1 = W1.T.dot(A0) + B1
	A1 = S1
	print (X[example].T[0],A1[0][0])
		
		


