import sys
import math
import random
import fileinput

TRAIN = 11
TEST = 22
RANGE = 16

CV_EPOCH = 10
TEST_EPOCH = 20

def sgn(scalar):
	return 1 if scalar >= 0 else -1


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


def logistic_algo(data, attr, bias, lf, flag, t, margin):
	cum = 0.0
	prediction = []
	

	for i in range(len(data)):
		label = data[i][0]

		cum = 0
		for j in range(len(data[i][1])):
			cum += attr[j] * data[i][1][j]

		#clarify
		pred1 = cum + bias	
		prediction.append(sgn(pred1))

		pred =  label * (cum + bias)
		
		if flag == TRAIN:
			if (pred <= 1):
				for j in range(len(data[i][1])):

					numer = label * data[i][1][j]
					denom = 1 + math.exp(pred)

					first_val = numer / denom
					second_val = (2 * attr[j]) / margin

					attr[j] = attr[j] - lf * (-(first_val) + second_val)
					#attr[j] = ((1 - lf) * attr[j]) + (lf * margin * label * data[i][1][j])

				#bias = ((1 - lf) * bias) + (lf * margin * label * 1)
				bias = bias + (lf / denom)
			else:
				for j in range(len(data[i][1])):
					attr[j] = ((1 - lf) * attr[j])

				bias = ((1 - lf) * bias)


	return attr, bias, prediction, t 


def cross_validation_logistic(lf, margin):
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
		
			
		for epoch in range(CV_EPOCH):
			weight, temp_bias, prediction, time = logistic_algo(training_set, weight, bias, lf, TRAIN, time, margin)
		
		#clarify	
		weight, temp_bias, prediction, time = logistic_algo(test_set, weight, temp_bias, lf, TEST, time, margin)
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


learning_factor = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
margin = [0.1, 1, 10, 100, 1000, 10000]

def wrapper_logistic():
	print 'CROSS-VALIDATION LOGISTIC'
	maximum = 0
	for i in learning_factor:
		for j in margin:
			cv_out = cross_validation_logistic(i, j)
			average = round((sum(cv_out) / len(cv_out)),4)
			
			if (average > maximum):
				maximum = average
				best_margin = j
				best_lf = i
			
	print ("Hyper parameters: Learning rate %s , Tradeoff %s and its Accuracy %s" % (best_lf, best_margin, maximum))
	
	print 'Logistic Regression'
	train_set = data_join(sys.argv[1])
	dev_set = data_join(sys.argv[2])
	test_set = data_join(sys.argv[3])

	weight, bias = reset()
	weight, bias = random_wb(weight, bias)

	time = 0
	most = 0
	graph_3 = []

	for epoch in range(TEST_EPOCH):
		weight, temp_bias, prediction, time = logistic_algo(train_set, weight, bias, best_lf, TRAIN, time, best_margin)

		w, tb, pr, time = logistic_algo(dev_set, weight, temp_bias, best_lf, TEST, time, best_margin)
		accu = accuracy(dev_set, pr)
		graph_3.append(accu)

		if (accu > most):
			most = accu
			b_weight = w
			b_bias = tb

	w2, b2, p2, t2 = logistic_algo(test_set, b_weight, b_bias, best_lf, TEST, time, best_margin)
	a = accuracy(test_set, p2)

	#for i in p2:
	#	print i

	print graph_3

	print ("Learning rate %s, Tradeoff %s and its test_set accuracy is %.4f" %(best_lf, best_margin, a))


wrapper_logistic()
