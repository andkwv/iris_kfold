import csv
import random as rd
import math as mt
import numpy as np
# libraries for the graph only
import matplotlib.pyplot as plt

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    DATA    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Here a class is used to load the file once instead of reading it over and over again, for effieciency
class IrisData():
	def __init__(self, filename):
		with open(filename, "r") as f_input:
			csv_input = csv.reader(f_input)
			self.details = list(csv_input)

			#create 5 lists inside the block list to separate into 5 blocks
			self.blocks = [[] for _ in range(5)]
			#we proceed to restructure the data
			self.restructure()

	#Pre-process the data
	#we want to prepare the data before process it 
	#because the data is originally ordered, we need to arrange it in a way that is more balanced

	#To restructure the data:
	# We have a list of 2d originally in self details 
	# First we divide into 5 blocks, it is determined through modulo. say row 11 will be in 1st block
	# This is done by using a nested list, we create 5 list to store them
	# Then we proceed to move the data


	def restructure(self):	
		detail_copy = self.details

		for row in range(150):
			block_num = row % 5
			self.blocks[block_num].append(detail_copy[row])

		#we proceed to randomize every block
		for block in range(5):
			self.randomize(block)

	# function to randomize one of the blocks
	def randomize(self, num_block):
		#randomize the list
		rd.shuffle(self.blocks[num_block])

	def get_col_row(self, block, row, col):
		return self.blocks[block][row][col]
		# Python index starts from 0 so we have to substract by 1

	#This function prints the list, to visualize the data
	def get_blocks(self):
		for block in range(5):
		 	print("\n\nTHIS IS BLOCK %s \n" % (block+1))

		 	for row in range(30):
		 		print(self.blocks[block][row])

	# def training_data(self,k):
	# 	self.training = self.blocks
	# 	del self.training[k]


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   CALCULATOR   XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX #

class Calculation():
	def __init__(self, data_object):
		self.data = data_object

	# the function to retrieve values from the csv file
	def getSepalPetal(self, block_num, row):

		sepal_length = float(self.data.get_col_row(block_num,row,0))
		sepal_width = float(self.data.get_col_row(block_num,row,1))
		petal_length = float(self.data.get_col_row(block_num,row,2))
		petal_width = float(self.data.get_col_row(block_num,row,3))

		y1 = float(self.data.get_col_row(block_num,row,5))
		y2 = float(self.data.get_col_row(block_num,row,6))

		# assigned to a list to simplify process 
		SepalPetal_list = [sepal_length, sepal_width, petal_length, petal_width, y1, y2]
		return SepalPetal_list
		# we simply return the list

	# the values will be pre calculated before calling the target calculation function
	def target(self, t1, t2, t3, t4, b):
		result = t1 + t2 + t4 + b
		return result

	# we calculate the sigmoid simply using the formula
	def sigmoid(self, targetx):
		exp = mt.exp(-targetx)
		sgmd = 1 / (1 + exp)
		return sgmd

	# we normalize, if it is below 0.5 then we will predict it to be 1
	def prediction(self, sigmoid_var):
		if(sigmoid_var < 0.5):
			return 0
		else:
			return 1

	#To calculate error
	def error(self, sigmoid_var, predc):
		err = (abs(sigmoid_var - predc)) ** 2
		return err

	#To calculate the dtheta
	def dtheta(self, sigmoid_var, y_target, x_theta):
		dtheta_res = 2*(sigmoid_var - y_target)*(1 - sigmoid_var) * sigmoid_var * x_theta
		return dtheta_res


# #########################################  INITIAL  ###########################################################


# ////////////////////////////////   TEST    //////////////////////////////

#Showing the data to validate that we divide and randomize correctly
# data.get_blocks()
# print("\n this is the values (sepal, petal) for block 5 (4) row 1 \n %s" % (calc.getSepalPetal(4,0)))

# Determine the LEARNING RATE
# answer = input("What do you want the learning rate to be ? \na. 0.1 	b. 0.8\n")
# if(answer == 'a'):
# 	l_rate = 0.1
# else:
# 	l_rate = 0.8

l_rate = 0.1


# ------------------------------------------ TRAINING ----------------------------------------------------------

# After dividing the data into K blocks, we keep 1 block for validation and use the rest for training
# In this case we start with using the 1st block as the validator. Basically creating a predictor for block 1
# Then we start to use the second block as validator, and so on.
# Finally after all block has been predicted (Validated), we calculate the average and show the graph

# Each phase of block validator, we will train the data 100 times (epoch)

class Trainer():
	def __init__(self, calculator):
		self.calc = calculator
		self.thetas = []
		self.bias = []
		self.ex = 0

	def k_iteration(self):
		avg_err_t = [[],[],[],[],[]]
		avg_err_v = [[],[],[],[],[]]
		avg_acc_t = [[],[],[],[],[]]
		avg_acc_v = [[],[],[],[],[]]

		for k in range(0,5):
			validate_block = k
			# //////////////////////////////////////////////////////////////////////////
			#Before we begin we will initiate all values starting from random number
			#for the initial values of the theta we will use a random number
			# ORIGINALLY LIKE THIS 
			# theta_1 = rd.random()
			# theta_2 = rd.random()
			# theta_3 = rd.random()
			# theta_4 = rd.random()
			# bias_a = rd.random()
			# theta_5 = rd.random()
			# theta_6 = rd.random()
			# theta_7 = rd.random()
			# theta_8 = rd.random()
			# bias_b = rd.random()

			#store theta initial number inside list.
			# 0 - 3 WILL BE THETHA_1 - THETA_4
			# 4 - 7 WILL BE THETA_5 - THETHA_8
			#NOW
			#FOR THETAS
			thetas = [rd.random()] * 8
			#FOR BIAS
			bias = [rd.random()] * 2

			epoch_graph = [] 
			error_train_graph = []
			accuracy_train_graph = []
			error_val_graph = []
			accuracy_val_graph = []

			print("\n\nTHIS IS FOR BLOCK %s\nERROR:\n" % (validate_block))

			for epoc in range(0,100):
				#initalize

				error = 0
				epoch_err_1 = 0		#value to store the epoch
				epoch_err_2 = 0 	
				
				accuracy = 0		#value t
				accuracy_1 = 0
				accuracy_2 = 0


				#calls the function to calculate every block

				new_data = self.training_data(validate_block, thetas, bias, epoc)
				
				thetas = new_data[0]
				bias = new_data[1]

				epoch_graph.append(epoc)
				error_train_graph.append(new_data[2])
				accuracy_train_graph.append(new_data[3])
				
				error_val_graph.append(new_data[4])
				accuracy_val_graph.append(new_data[5])

				avg_err_t[validate_block].append(new_data[2])
				avg_err_v[validate_block].append(new_data[4])

				avg_acc_t[validate_block].append(new_data[3])
				avg_acc_v[validate_block].append(new_data[5])

			error_train_graph = np.array(error_train_graph)
			error_val_graph = np.array(error_val_graph)
			accuracy_train_graph = np.array(accuracy_train_graph)
			accuracy_val_graph = np.array(accuracy_val_graph)
			# to plot the graph
			plt.figure(1)	#this is meant for first and second error graph, both will be combined in same grpah
			plt.plot(epoch_graph, error_train_graph, label='Training')
			plt.plot(epoch_graph, error_val_graph,  label='Validation')
			plt.title('Error Graph Block %s' % (k+1))
			plt.legend(loc = "best")

			plt.figure(2)	#this is meant for first and second error graph, both will be combined in same grpah
			plt.plot(epoch_graph, accuracy_train_graph, label='Training')
			plt.plot(epoch_graph, accuracy_val_graph,  label='Validation')
			plt.title('Accuracy Graph Block %s' % (k+1))
			plt.legend(loc = "best")

			plt.show()

		avg_err_t_graph = []
		avg_err_v_graph = [] 

		avg_acc_t_graph = []
		avg_acc_v_graph = []

		avg_epoch_graph = []

		avg_err_t = np.array(avg_err_t)
		avg_err_v = np.array(avg_err_v)

		avg_acc_t = np.array(avg_acc_t)
		avg_acc_v = np.array(avg_acc_v)

		# print(avg_acc_t)

		for epoc in range(0,100):
			avg_err_t_avg = 0.0
			avg_err_v_avg = 0.0
			avg_acc_t_avg = 0.0
			avg_acc_v_avg = 0.0
			
			for data in range(0,5):
				avg_err_t_avg += avg_err_t[data][epoc]
				avg_err_v_avg += avg_err_v[data][epoc]

				avg_acc_t_avg += avg_acc_t[data][epoc]
				avg_acc_v_avg += avg_acc_v[data][epoc]

			avg_err_t_graph.append((avg_err_t_avg/5))
			avg_err_v_graph.append(avg_err_v_avg/5)

			avg_acc_t_graph.append(avg_acc_t_avg/5)
			avg_acc_v_graph.append(avg_acc_v_avg/5)

			avg_epoch_graph.append(epoc+1)

		# print(avg_acc_t_graph)

		# valke = 0
		# for j in range(5):
		# 	valke += avg_err_t[j][99]
		# print(valke/5)

		plt.figure(3)
		plt.plot(avg_epoch_graph, avg_err_t_graph, label = 'Training Error average')
		plt.plot(avg_epoch_graph, avg_err_v_graph, label = 'Validation Error average')
		plt.title('Average Error Graph Block')
		plt.legend(loc = "best")

		plt.figure(4)
		plt.plot(avg_epoch_graph, avg_acc_t_graph, label = 'Training Accuracy average')
		plt.plot(avg_epoch_graph, avg_acc_v_graph, label = 'Validation Accuracy average')
		plt.title('Average Accuracy Graph Block')
		plt.legend(loc = "best")

		plt.show()

		


	# def block_iteration(self, validate_block, thetas, bias):

	# 	# #we will process each block EXCEPT the validation block
	# 	# for k in range(5):
	# 	# 	if(k != validate_block): #execute training besides validation block

	# 	self.training_data(validate_block, thetas, bias)

	# -------------------------------------------------------------------------------

	def training_data(self, block_num, thetas, bias, epoch):
		#put it in variable to shorten it
		formula = self.calc
		#the list will keep the values of thetas and bias for the future
		epoch_err_1 = 0
		epoch_err_2 = 0
		epoch_err = 0
		accuracy = 0
		
		epoch_num = []
		epoch_train_err = []

		#we will process each block EXCEPT the validation block
		for k in range(5):
			if(k != block_num): #execute training besides validation block
				for row in range(30):
					SP = formula.getSepalPetal(k, row) # SP WILL SIGNIFY THE LIST FOR SEPAL/PETAL VALUES

					# we pre-calculate to simplify calculating target and others
					
					r1 = thetas[0] * SP[0]		
					r2 = thetas[1] * SP[1]
					r3 = thetas[2] * SP[2]
					r4 = thetas[3] * SP[3]
					y1 = SP[4]

					#to calculate each of the values we call the function
					target_1 = formula.target(r1,r2,r3,r4,bias[0])
					sigmoid_1 = formula.sigmoid(target_1)
					prediction_1 = formula.prediction(sigmoid_1)
					err_1 = formula.error(sigmoid_1, y1)

					# Now we generate the dtheta to improve each theta
					dt_list_1 = []
					for j in range(4):									 
						dt_list_1.append(formula.dtheta(sigmoid_1, y1, SP[j]))
					# We begin by calculating dtheta for each dtheta, then we calculate for the bias	
					dt_list_1.append(formula.dtheta(sigmoid_1, y1, 1))		

			# *******************************************************
					
					#prepare for second prediction

					r5 = thetas[4] * SP[0]		#same with the previous calculation, but with differnt thetas
					r6 = thetas[5] * SP[1]
					r7 = thetas[6] * SP[2]
					r8 = thetas[7] * SP[3]
					y2 = SP[5]

					target_2 = formula.target(r5,r6,r7,r8, bias[1]) #we all the target function, formula in function
					sigmoid_2 = formula.sigmoid(target_2)		#we call sigmoid function, formula is also in function
					prediction_2 = formula.prediction(sigmoid_2)	#call the prediction function
					err_2 = formula.error(sigmoid_2, y2)

					dt_list_2 = []
					for k in range(4):		
						dt_list_2.append(formula.dtheta(sigmoid_2, y2, SP[k]))

					# We begin by calculating dtheta for each dtheta, then we calculate for the bias	
					dt_list_2.append(formula.dtheta(sigmoid_2, y2, 1))


			# **********************************************************************************************************

					#NOW WE CALCULATE THE VALUES FOR THE NEXT ITERATION THETA (IMPROVED FROM DTHETA)

					thetas[0] =  thetas[0] - (l_rate * dt_list_1[0])
					thetas[1] =  thetas[1] - (l_rate * dt_list_1[1])
					thetas[2] =  thetas[2] - (l_rate * dt_list_1[2])
					thetas[3] =  thetas[3] - (l_rate * dt_list_1[3])
					bias[0] =  bias[0] - (l_rate * dt_list_1[4])

					thetas[4] =  thetas[4] - (l_rate * dt_list_2[0])
					thetas[5] =  thetas[5] - (l_rate * dt_list_2[1])
					thetas[6] =  thetas[6] - (l_rate * dt_list_2[2])
					thetas[7] =  thetas[7] - (l_rate * dt_list_2[3])
					bias[1] =  bias[1] - (l_rate * dt_list_2[4])  

					#calculate the error
					epoch_err +=  (err_1 + err_2)

					if(prediction_1 == y1 and prediction_2 == y2):
						accuracy += 1

		val_err_1 = 0
		val_err_2 = 0
		val_err = 0
		val_acc = 0

		#THIS IF FOR VALIDATION
		for v_row in range(30):
			SP = formula.getSepalPetal(block_num, v_row) # SP WILL SIGNIFY THE LIST FOR SEPAL/PETAL VALUES

			# we pre-calculate to simplify calculating target and others
					
			r1 = thetas[0] * SP[0]		
			r2 = thetas[1] * SP[1]
			r3 = thetas[2] * SP[2]
			r4 = thetas[3] * SP[3]
			y1 = SP[4]

			#to calculate each of the values we call the function
			target_1 = formula.target(r1,r2,r3,r4,bias[0])
			sigmoid_1 = formula.sigmoid(target_1)
			prediction_1 = formula.prediction(sigmoid_1)
			err_1 = formula.error(sigmoid_1, y1)


			# *******************************************************
					
			#prepare for second prediction

			r5 = thetas[4] * SP[0]		#same with the previous calculation, but with differnt thetas
			r6 = thetas[5] * SP[1]
			r7 = thetas[6] * SP[2]
			r8 = thetas[7] * SP[3]
			y2 = SP[5]

			target_2 = formula.target(r5,r6,r7,r8, bias[1]) #we all the target function, formula in function
			sigmoid_2 = formula.sigmoid(target_2)		#we call sigmoid function, formula is also in function
			prediction_2 = formula.prediction(sigmoid_2)	#call the prediction function
			err_2 = formula.error(sigmoid_2, y2)


			# **********************************************************************************************************

			#calculate the error
			val_err +=  (err_1 + err_2)

			if(prediction_1 == y1 and prediction_2 == y2):
				val_acc += 1

		val_err = val_err/30
		val_acc = val_acc/30

		val_data = [val_err, val_acc]


		epoch_err = epoch_err/120
		accuracy = accuracy/120
		self.ex = self.ex + 1
		print("%s. error: %s + accuracy: %s" % (self.ex, epoch_err, accuracy))
		print("%s. v_error: %s + v_accuracy: %s" % (self.ex, val_err, val_acc))

		#Now Validate for the BLOCK. for 30 rows
		# val_data = self.validating_data(block_num, trained_thetas, trained_bias)

		update = [thetas, bias, epoch_err, accuracy, val_data[0], val_data[1]]

		return update

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VALIDATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# def validating_data(self, val_blockNum, thetas, bias):
	# 	thetas_comp = thetas
	# 	bias_comp = bias
		
	# 	for v_row in range(30):
	# 		formula = self.calc

	# 		SP = formula.getSepalPetal(val_blockNum, v_row) # SP WILL SIGNIFY THE LIST FOR SEPAL/PETAL VALUES

	# 		val_err_1 = 0
	# 		val_err_2 = 0
	# 		val_err = 0
	# 		val_acc = 0

	# 		# we pre-calculate to simplify calculating target and others
						
	# 		r1 = thetas_comp[0] * SP[0]		
	# 		r2 = thetas_comp[1] * SP[1]
	# 		r3 = thetas_comp[2] * SP[2]
	# 		r4 = thetas_comp[3] * SP[3]
	# 		y1 = SP[4]

	# 		#to calculate each of the values we call the function
	# 		target_1 = formula.target(r1,r2,r3,r4,bias[0])
	# 		sigmoid_1 = formula.sigmoid(target_1)
	# 		prediction_1 = formula.prediction(sigmoid_1)
	# 		err_1 = formula.error(sigmoid_1, y1)

	# 		# *******************************************************
						
	# 			#prepare for second prediction

	# 		r5 = thetas_comp[4] * SP[0]		#same with the previous calculation, but with differnt thetas
	# 		r6 = thetas_comp[5] * SP[1]
	# 		r7 = thetas_comp[6] * SP[2]
	# 		r8 = thetas_comp[7] * SP[3]
	# 		y2 = SP[5]

	# 		target_2 = formula.target(r5,r6,r7,r8, bias[1]) #we all the target function, formula in function
	# 		sigmoid_2 = formula.sigmoid(target_2)		#we call sigmoid function, formula is also in function
	# 		prediction_2 = formula.prediction(sigmoid_2)	#call the prediction function
	# 		err_2 = formula.error(sigmoid_2, y2)

	# 		#calculate the error
	# 		val_err +=  (err_1 + err_2)

	# 		if(prediction_2 == y2):
	# 			val_acc += 1

	# 	val_err = val_err/30
	# 	val_acc = val_acc/30

	# 	val_data = [val_err, val_acc]

	# 	print("%s. v_error: %s + v_accuracy: %s" % (self.ex,val_err, val_acc))

	# 	return val_data

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# *********************************************   BEGIN   ***********************************************

data = IrisData("D:\KULIAH\PELAJARAN_SEMESTER_6\Machine Learning\TUGAS_3\iris.csv")
calc = Calculation(data)
train = Trainer(calc)

#we start with training the data
#This loop is to DETERMINE WHICH BLOCK IS THE VALIDATION BLOCK

train.k_iteration()
		
