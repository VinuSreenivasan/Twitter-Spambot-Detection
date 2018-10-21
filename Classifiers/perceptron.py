import sys
import random

TRAIN = 11
TEST = 22
RANGE = 16

def sgn(scalar):
	return 1 if scalar >= 0 else -1


def perceptron(data, attr, bias, lf, flag):
	cum = 0.0
	prediction = []
	
	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]
		
		pred =  cum + bias
		prediction.append(sgn(pred))
		
		if flag == TRAIN:
			if (sgn(pred) != label):
				for j in range(len(data[i][1])):
					attr[j] = attr[j] + (lf * data[i][1][j] * label)
				bias = bias + (lf * label) 

	return attr, bias, prediction 


def accuracy(data, predict):
	pr_ct = 0
	pr_wg = 0
	
	for i in range(len(data)):
		label = data[i][0]
		
		if (predict[i] == label):
			pr_ct += 1
		else:
			pr_wg += 1
	
	if (pr_ct + pr_wg) == len(data):
		accuracy = round(((float(pr_ct) / len(data)) * 100),4)
	
	return accuracy


def data_join(filename):
	data_set = []
	with open(filename) as f:
		for line in f:
			value = line.split()
			label = value[0]
			value.pop(0)
			
			temp = []	
			a_list = []
			b_list = []
			w = 1
			for i in value:
				new = i.split(":")
				while (w != int(new[0])): 
					a_list.insert(w, w)
					b_list.insert(w, float(0))
					w = w + 1
				a_list.append(int(new[0]))
				b_list.append(float(new[1]))
				w = int(new[0])+1

			l1 = len(a_list)
			if (a_list[l1-1] != RANGE):
				for i in range(l1+1, RANGE+1, 1):
					a_list.insert(i, i)
					b_list.insert(i, float(0))

			temp.append(float(label))
			temp.append(b_list)
			data_set.append(temp)

	return data_set


def cross_validation_simple(lf):
	cv_result = []
	for i in range(0,5,1):
		training_set = []
		test_set = []
		weight, bias = reset()
		weight, bias = random_wb(weight, bias)

		if i != 0:
			training_set+=data_join("Dataset/CVSplits/training00.data")
		else:
			test_set = data_join("Dataset/CVSplits/training00.data")

		if i != 1:
			training_set+=data_join("Dataset/CVSplits/training01.data")
		else:
			test_set = data_join("Dataset/CVSplits/training01.data")

		if i != 2:
			training_set+=data_join("Dataset/CVSplits/training02.data")
		else:
			test_set = data_join("Dataset/CVSplits/training02.data")
		
		if i != 3:
			training_set+=data_join("Dataset/CVSplits/training03.data")
		else:
			test_set = data_join("Dataset/CVSplits/training03.data")
		
		if i != 4:
			training_set+=data_join("Dataset/CVSplits/training04.data")
		else:
			test_set = data_join("Dataset/CVSplits/training04.data")
		

		for epoch in range(10):
			weight, temp_bias, prediction = perceptron(training_set, weight, bias, lf, TRAIN)
			
		weight, temp_bias, prediction = perceptron(test_set, weight, temp_bias, lf, TEST)
		accu = accuracy(test_set, prediction)

		cv_result.append(accu)

	return cv_result
		
def perceptron_dynamic(data, attr, bias, lf, flag, t):
	cum = 0.0
	prediction = []
	
	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]
		
		pred =  cum + bias
		prediction.append(sgn(pred))
		
		if flag == TRAIN:
			if (sgn(pred) != label):

				new_lf = float(lf) / (1 + t)

				for j in range(len(data[i][1])):
					attr[j] = attr[j] + (new_lf * data[i][1][j] * label)
				bias = bias + (new_lf * label)
				t = t + 1

	return attr, bias, prediction, t 


def cross_validation_dynamic(lf):
	cv_result = []
	for i in range(0,5,1):
		training_set = []
		test_set = []
		weight, bias = reset()
		weight, bias = random_wb(weight, bias)

		time = 0

		if i != 0:
			training_set+=data_join("Dataset/CVSplits/training00.data")
		else:
			test_set = data_join("Dataset/CVSplits/training00.data")

		if i != 1:
			training_set+=data_join("Dataset/CVSplits/training01.data")
		else:
			test_set = data_join("Dataset/CVSplits/training01.data")

		if i != 2:
			training_set+=data_join("Dataset/CVSplits/training02.data")
		else:
			test_set = data_join("Dataset/CVSplits/training02.data")
		
		if i != 3:
			training_set+=data_join("Dataset/CVSplits/training03.data")
		else:
			test_set = data_join("Dataset/CVSplits/training03.data")
		
		if i != 4:
			training_set+=data_join("Dataset/CVSplits/training04.data")
		else:
			test_set = data_join("Dataset/CVSplits/training04.data")
		
			
		for epoch in range(10):
			weight, temp_bias, prediction, t1 = perceptron_dynamic(training_set, weight, bias, lf, TRAIN, time)
			time = t1
		
		weight, temp_bias, prediction, time = perceptron_dynamic(test_set, weight, temp_bias, lf, TEST, time)
		accu = accuracy(test_set, prediction)
		cv_result.append(accu)

	return cv_result


def perceptron_margin(data, attr, bias, lf, flag, t, margin):
	cum = 0.0
	prediction = []

	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]

		pred1 = cum + bias	
		pred =  label * (cum + bias)
		prediction.append(sgn(pred1))
		
		if flag == TRAIN:
			if (pred < margin):

				new_lf = float(lf) / (1 + t)

				for j in range(len(data[i][1])):
					attr[j] = attr[j] + (new_lf * data[i][1][j] * label)
				bias = bias + (new_lf * label)
				t = t + 1


	return attr, bias, prediction, t 


def cross_validation_margin(lf, margin):
	cv_result = []
	for i in range(0,5,1):
		training_set = []
		test_set = []
		weight, bias = reset()
		weight, bias = random_wb(weight, bias)

		time = 0

		if i != 0:
			training_set+=data_join("Dataset/CVSplits/training00.data")
		else:
			test_set = data_join("Dataset/CVSplits/training00.data")

		if i != 1:
			training_set+=data_join("Dataset/CVSplits/training01.data")
		else:
			test_set = data_join("Dataset/CVSplits/training01.data")

		if i != 2:
			training_set+=data_join("Dataset/CVSplits/training02.data")
		else:
			test_set = data_join("Dataset/CVSplits/training02.data")
		
		if i != 3:
			training_set+=data_join("Dataset/CVSplits/training03.data")
		else:
			test_set = data_join("Dataset/CVSplits/training03.data")
		
		if i != 4:
			training_set+=data_join("Dataset/CVSplits/training04.data")
		else:
			test_set = data_join("Dataset/CVSplits/training04.data")
		
			
		for epoch in range(10):
			weight, temp_bias, prediction, time = perceptron_margin(training_set, weight, bias, lf, TRAIN, time, margin)
		
		weight, temp_bias, prediction, time = perceptron_margin(test_set, weight, temp_bias, lf, TEST, time, margin)
		accu = accuracy(test_set, prediction)
		cv_result.append(accu)

	return cv_result

def reset():
	weight = [0.0 for i in range(RANGE)]
	bias = 0.0

	return weight, bias

def random_wb(weight, bias):
	for i in range(len(weight)):
		weight[i] = random.uniform(-0.01,0.01)
	bias = random.uniform(-0.01,0.01)

	return weight, bias

learning_factor = [10, 1, 0.1, 0.01, 0.001, 0.0001]


def hyper_function(new_list):
	maximum = max(new_list)
	for i in range(len(new_list)):
		if (maximum == new_list[i]):
			final = new_list[i]
			index = i
	print ("Hyper parameter: 'Learning rate = %s' and its 'accuracy = %s'" %(learning_factor[index], maximum))
	
	return (learning_factor[index])


def wrapper_simple_perceptron():
	print 'CROSS-VALIDATION SIMPLE PERCEPTRON'
	lf_acr = []
	for i in learning_factor:
		cv_out = cross_validation_simple(i)
		average = round((sum(cv_out) / len(cv_out)),4)
		lf_acr.append(average)

	#print lf_acr
	hy_pm = hyper_function(lf_acr)


	#print '\n'
	print 'SIMPLE PERCEPTRON'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)
	most = 0
	graph_1 = []

	for epoch in range(20):
		weight, temp_bias, prediction = perceptron(train_set, weight, bias, hy_pm, TRAIN)

		w, tb, pr = perceptron(dev_set, weight, temp_bias, hy_pm, TEST)
		accu = accuracy(dev_set, pr)
		graph_1.append(accu)

		if (accu > most):
			most = accu
			b_weight = w
			b_bias = tb

	w1, b1, p1 = perceptron(test_set, b_weight, b_bias, hy_pm, TEST)
	a = accuracy(test_set, p1)

	#for i in p1:
	#	print i

	print graph_1
	print ("Learning rate %s and its test_set accuracy is %.4f" %(hy_pm ,a))



def wrapper_dynamic_perceptron():
	#perceptron with dynamic learning rate
	#print '\n'
	print 'CROSS-VALIDATION PERCEPTRON WITH DYNAMIC LEARNING RATE'
	lf_acr_1 = []
	for i in learning_factor:
		cv_out = cross_validation_dynamic(i)
		average = round((sum(cv_out) / len(cv_out)),4)
		lf_acr_1.append(average)

	#print lf_acr_1
	hy_pm2 = hyper_function(lf_acr_1)


	#perceptron with dynamic learning rate
	#print '\n'
	print 'PERCEPTRON WITH DYNAMIC LEARNING RATE'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)
	time = 0
	most = 0
	graph_2 = []

	for epoch in range(20):
		weight, temp_bias, prediction, time = perceptron_dynamic(train_set, weight, bias, hy_pm2, TRAIN, time)

		w, tb, pr, time = perceptron_dynamic(dev_set, weight, temp_bias, hy_pm2, TEST, time)
		accu = accuracy(dev_set, pr)
		graph_2.append(accu)

		if (accu > most):
			most = accu
			b_weight = w
			b_bias = tb

	w1, b1, p1, t1 = perceptron_dynamic(test_set, b_weight, b_bias, hy_pm2, TEST, time)
	a = accuracy(test_set, p1)

	#for i in p1:
	#	print i

	print graph_2
	print ("Learning rate %s and its test_set accuracy is %.4f" %(hy_pm2 ,a))


def wrapper_margin_perceptron():
	#margin perceptron
	#print '\n'
	print 'CROSS-VALIDATION MARGIN PERCEPTRON WITH DYNAMIC LEARNING RATE'
	maximum = 0
	margin = [1, 0.1, 0.01]
	for i in learning_factor:
		for j in margin:
			cv_out = cross_validation_margin(i, j)
			average = round((sum(cv_out) / len(cv_out)),4)
			
			if (average > maximum):
				maximum = average
				best_margin = j
				best_lf = i
			
	#print maximum
	print ("Hyper parameters: Learning rate %s , Margin %s and its Accuracy %s" % (best_lf, best_margin, maximum))
	
	#margin perceptron
	#print '\n'
	print 'MARGIN PERCEPTRON'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)

	time = 0
	most = 0
	graph_3 = []

	for epoch in range(20):
		weight, temp_bias, prediction, time = perceptron_margin(train_set, weight, bias, best_lf, TRAIN, time, best_margin)

		w, tb, pr, time = perceptron_margin(dev_set, weight, temp_bias, best_lf, TEST, time, best_margin)
		accu = accuracy(dev_set, pr)
		graph_3.append(accu)

		if (accu > most):
			most = accu
			b_weight = w
			b_bias = tb

	w2, b2, p2, t2 = perceptron_margin(test_set, b_weight, b_bias, best_lf, TEST, time, best_margin)
	a = accuracy(test_set, p2)

	#for i in p2:
	#	print i

	print graph_3

	print ("Learning rate %s, margin %s and its test_set accuracy is %.4f" %(best_lf, best_margin, a))


def perceptron_average(data, attr, bias, lf, flag, avg_weight, avg_bias):
	cum = 0.0
	prediction = []
	
	#print avg_weight, avg_bias	

	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]
		
		pred =  cum + bias
		prediction.append(sgn(pred))
		
		if flag == TRAIN:
			for j in range(len(data[i][1])):
				avg_weight[j] += attr[j]
			avg_bias = avg_bias + bias

			if (sgn(pred) != label):

				for j in range(len(data[i][1])):
					attr[j] = attr[j] + (lf * data[i][1][j] * label)
				bias = bias + (lf * label) 

	if flag == TEST:
		return prediction
	else:
		return avg_weight, avg_bias, prediction, attr, bias 



def cross_validation_average(lf):
	cv_result = []
	for i in range(0,5,1):
		training_set = []
		test_set = []
		weight, bias = reset()
		weight, bias = random_wb(weight, bias)

		if i != 0:
			training_set+=data_join("Dataset/CVSplits/training00.data")
		else:
			test_set = data_join("Dataset/CVSplits/training00.data")

		if i != 1:
			training_set+=data_join("Dataset/CVSplits/training01.data")
		else:
			test_set = data_join("Dataset/CVSplits/training01.data")

		if i != 2:
			training_set+=data_join("Dataset/CVSplits/training02.data")
		else:
			test_set = data_join("Dataset/CVSplits/training02.data")
		
		if i != 3:
			training_set+=data_join("Dataset/CVSplits/training03.data")
		else:
			test_set = data_join("Dataset/CVSplits/training03.data")
		
		if i != 4:
			training_set+=data_join("Dataset/CVSplits/training04.data")
		else:
			test_set = data_join("Dataset/CVSplits/training04.data")
		
		
		a_w = [0.0 for i in range(RANGE)]
		a_bias = 0
		
		for epoch in range(10):
			a_w1, a_bias1, prediction, weight1, bias1 = perceptron_average(training_set, weight, bias, lf, TRAIN, a_w, a_bias)

			a_w = a_w1
			a_bias = a_bias1
			weight = weight1
			bias = bias1			

		prediction = perceptron_average(test_set, a_w1, a_bias1, lf, TEST, a_w, a_bias)
		accu = accuracy(test_set, prediction)
		cv_result.append(accu)

	return cv_result


def wrapper_average_perceptron():
	#cross_validation_simple
	#print '\n'
	print 'CROSS-VALIDATION AVERAGE PERCEPTRON'
	lf_acr = []
	for i in learning_factor:
		cv_out = cross_validation_average(i)
		average = round((sum(cv_out) / len(cv_out)),4)
		lf_acr.append(average)

	#print lf_acr
	hy_pm = hyper_function(lf_acr)


	#simple perceptron without cross-valiation
	#print '\n'
	print 'AVERAGE PERCEPTRON'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)
	most = 0
	a_w = [0.0 for i in range(RANGE)]
	a_b = 0.0
	graph_4 = []

	
	for epoch in range(20):
		a_w, a_b, prediction, w3, b3 = perceptron_average(train_set, weight, bias, hy_pm, TRAIN, a_w, a_b)
		
		pr = perceptron_average(dev_set, a_w, a_b, hy_pm, TEST, a_w, a_b)
		accu = accuracy(dev_set, pr)
		graph_4.append(accu)

		if (accu > most):
			most = accu
			b_weight = a_w
			b_bias = a_b

	a_w = [0.0 for i in range(RANGE)]
	a_b = 0.0

	p1 = perceptron_average(test_set, b_weight, b_bias, hy_pm, TEST, a_w, a_b)
	a = accuracy(test_set, p1)

	#for i in p1:
	#	print i

	print graph_4

	print ("Learning rate %s and its test_set accuracy is %.4f" %(hy_pm ,a))



def perceptron_aggressive(data, attr, bias, mr, flag):
	prediction = []

	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]

		pred1 = cum + bias
		pred =  label * (cum + bias)
		prediction.append(sgn(pred1))
		
		if flag == TRAIN:
			if (pred <= mr):

				cum_x = 0
				for j in range(len(data[i][1])):
					cum_x = cum_x + ((data[i][1][j]) * (data[i][1][j]))

				n = float(mr - pred) / (cum_x + 1)

				for j in range(len(data[i][1])):
					attr[j] = attr[j] + (n * data[i][1][j] * label)
				bias = bias + (n * label) 

	return attr, bias, prediction 

def cross_validation_aggressive(mr):
	cv_result = []
	for i in range(0,5,1):
		training_set = []
		test_set = []
		weight, bias = reset()
		weight, bias = random_wb(weight, bias)

		if i != 0:
			training_set+=data_join("Dataset/CVSplits/training00.data")
		else:
			test_set = data_join("Dataset/CVSplits/training00.data")

		if i != 1:
			training_set+=data_join("Dataset/CVSplits/training01.data")
		else:
			test_set = data_join("Dataset/CVSplits/training01.data")

		if i != 2:
			training_set+=data_join("Dataset/CVSplits/training02.data")
		else:
			test_set = data_join("Dataset/CVSplits/training02.data")
		
		if i != 3:
			training_set+=data_join("Dataset/CVSplits/training03.data")
		else:
			test_set = data_join("Dataset/CVSplits/training03.data")
		
		if i != 4:
			training_set+=data_join("Dataset/CVSplits/training04.data")
		else:
			test_set = data_join("Dataset/CVSplits/training04.data")
		
		
		for epoch in range(10):
			w, bias, prediction = perceptron_aggressive(training_set, weight, bias, mr, TRAIN)
			
		weight, temp_bias, prediction = perceptron_aggressive(test_set, w, bias, mr, TEST)
		accu = accuracy(test_set, prediction)
		cv_result.append(accu)

	return cv_result


def wrapper_aggressive_perceptron():
	#cross_validation_simple
	print '\n'
	print 'CROSS-VALIDATION AGGRESSIVE PERCEPTRON'
	lf_acr = []
	margin = [1, 0.1, 0.01]
	for i in margin:
		cv_out = cross_validation_aggressive(i)
		average = round((sum(cv_out) / len(cv_out)),4)
		lf_acr.append(average)

	#print lf_acr
	hy_pm = hyper_function(lf_acr)


	#simple perceptron without cross-valiation
	#print '\n'
	print 'AGGRESSIVE PERCEPTRON'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)
	most = 0
	graph_5 = []

	
	for epoch in range(20):
		w, b, prediction = perceptron_aggressive(train_set, weight, bias, hy_pm, TRAIN)

		w, tb, pr = perceptron_aggressive(dev_set, w, b, hy_pm, TEST)
		accu = accuracy(dev_set, pr)
		graph_5.append(accu)

		if (accu > most):
			most = accu
			b_weight = w
			b_bias = tb

	w1, b1, p1 = perceptron_aggressive(test_set, b_weight, b_bias, hy_pm, TEST)
	a = accuracy(test_set, p1)

	#for i in p1:
	#	print i

	print graph_5

	print ("Learning rate %s and its test_set accuracy is %.4f" %(hy_pm ,a))

	
#wrapper_simple_perceptron()
wrapper_dynamic_perceptron()
#wrapper_margin_perceptron()
#wrapper_average_perceptron()
wrapper_aggressive_perceptron()
