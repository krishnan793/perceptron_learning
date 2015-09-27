import numpy as np

def compute_epoch(V,T,W,alpha):
	for i in range(V.shape[0]):
		# V_i current input vector
		# T_i actual output
		# y predicted output
		V_i = V[i]
		T_i = T[i]
		
		if (np.dot(W,V_i) >= 0):
			y = 1
		else:
			y = 0
		a = alpha * (T_i-y)
		#print "a = ",a
		W_delta = a * V_i
		#print "W_delta = ",W_delta
		W = W + W_delta
		#print "W = ",W
	#print "="*21
	return W

def compute_weight(V,T,W,alpha):
	Max_iteration = 100
	epoch_difference = 1
	epoch_1 = compute_epoch(V,T,W,alpha)
	
	while(epoch_difference != 0):	
		epoch_2 = compute_epoch(V,T,epoch_1,alpha)
		tmp = np.absolute(epoch_1-epoch_2)
		epoch_difference = np.dot(tmp,tmp)
		epoch_1 = epoch_2
		Max_iteration = Max_iteration - 1
		
		if (Max_iteration == 0):
			print "Max Iteration Reached. ANN not convergerd!"
			exit()
	
	return epoch_2

def validate_weight(V,T,W):
	space = 4
	print "     X      Y  Predicted"
	for i in range(V.shape[0]):
		print V[i,1:],T[i],
		if (np.dot(V[i],W) >= 0):
			print "[ 1 ]"
		else:
			print "[ 0 ]"
		
data = np.loadtxt("OR.txt")

V = data[:,:-1]
V = np.insert(V,0,-1,axis=1)	# Padd 1 to account for theta
T = data[:,-1:]

# Initialise Weights to zero
W = np.array([0 for i in range(V.shape[1])])	# V.shape[1] = no of columns
# learning rate
alpha = 0.5

W = compute_weight(V,T,W,alpha)
print "Calculated weight: ",W
print "Validating the weights calculated"
validate_weight(V,T,W)
