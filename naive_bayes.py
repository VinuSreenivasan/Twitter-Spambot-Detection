import sys
import math
import random
import numpy as np
from numpy import ma
from collections import OrderedDict
import scipy.sparse as sp

CV_EPOCH = 10

def create_matrix(data, label):
	mat = sp.dok_matrix((len(data), 16), dtype=np.float64)
	lab = np.empty([len(data)], dtype=int)

	for k, v in data.items():
		for k1, v1 in v.items():
			mat[int(k),int(k1)-1] = float(v1)

		lab[int(k)]=label[int(k)]

	final_matrix = mat.todense()

	return final_matrix, lab


def label_count(train_label):
	unique = list(set(train_label))
	j = unique[0]
	count1 = 0

	for i in train_label:
		if i==j:
			count1 = count1 + 1
	count2 = len(train_label) - count1

	return count1, count2



def naive_train(train_data, train_label, pos_matrix, neg_matrix, smooth):
	unique_labels = list(set(train_label))

	positive_count, negative_count = label_count(train_label)

	pos_label_matrix = np.empty([len(train_data), 16], dtype=float)
	neg_label_matrix = np.empty([len(train_data), 16], dtype=float) 
	
	prior_pos = float(positive_count) / len(train_data)
	prior_neg = float(negative_count) / len(train_data)

	final_matrix, final_label = create_matrix(train_data, train_label)
	
	i=0
	for col in train_label:
		if col ==unique_labels[0]:
			pos_label_matrix[i,:] = final_matrix[i]
		else:
			neg_label_matrix[i,:] = final_matrix[i]
		i=i+1

	pos_matrix = pos_label_matrix[:,1:] #excluding empty first column
	neg_matrix = neg_label_matrix[:,1:] 

	num_pos = (np.sum(pos_matrix, axis=0) + float(smooth))
	den_pos = (positive_count + len(unique_labels) * float(smooth)) 
	pos_matrix = np.divide(num_pos, den_pos)
	
	num_neg = (np.sum(neg_matrix, axis=0) + float(smooth))
	den_neg = (negative_count + len(unique_labels) * float(smooth))
	neg_matrix = np.divide(num_neg, den_neg)

	return pos_matrix, neg_matrix

def naive_test(test_set, test_label, postive_matrix, negative_matrix):
	prediction = []
	postive_count, negative_count = label_count(test_label)
	test_matrix, true_label = create_matrix(test_set, test_label)
	
	prior_pos = float(postive_count) / len(test_set)
	prior_neg = float(negative_count) / len(test_set)

	test_mat = sp.dok_matrix((len(test_set), 16), dtype = np.float64)
	test_mat = test_matrix[:,1:]

	postive_matrix = [test_mat==1]*postive_matrix + [test_mat==0] * (1-postive_matrix)
	negative_matrix = [test_mat==1]*negative_matrix + [test_mat==0] * (1-negative_matrix)
	
	postive_matrix = np.squeeze(postive_matrix, axis=0)
	negative_matrix = np.squeeze(negative_matrix, axis=0)
	
	#res_pos = ma.filled(np.log(ma.masked_equal(postive_matrix, 0)), 0)
	#res_neg = ma.filled(np.log(ma.masked_equal(negative_matrix, 0)), 0)

	#postive_matrix = np.sum(res_pos, axis=1)
	#negative_matrix = np.sum(res_neg, axis=1)

	postive_matrix = np.sum(postive_matrix, axis=1)
	negative_matrix = np.sum(negative_matrix, axis=1)

	postive_matrix = postive_matrix + math.log(prior_pos) 
	if (prior_neg != 0):
		negative_matrix = negative_matrix + math.log(prior_neg)

	for row in range(len(test_set)):
		if postive_matrix[row] > negative_matrix[row]:
			prediction.append(1)
		else:
			prediction.append(-1)
	
	return prediction
	
def accuracy(data_label, predict):
	pr_ct = 0
	pr_wg = 0
	
	if len(data_label) != len(predict):
		print ("labels and predicts are not equal")
	
	for i in range(len(data_label)):
		
		if predict[i] == data_label[i]:
			pr_ct += 1
		else:
			pr_wg += 1

	if (pr_ct + pr_wg) == len(data_label):
		accuracy = round(((float(pr_ct) / len(data_label)) * 100),4)
	
	return accuracy


def create_data(filename):
	with open (filename) as f:
		data = OrderedDict()
		label = []
		num = 0
		for line in f:
			line = line.split()
			label.append(int(line[0]))
			line.pop(0)
			
			feat = []
			val = []
			row = {}

			for i in line:
				i = i.split(":")
				feat.append(i[0])
				val.append(i[1])
			row = OrderedDict(zip(feat, val))

			data[num] = row
			num = num + 1

	return data, label

def create_data_list(data_list):
	data = OrderedDict()
	label = []
	num = 0

	for line in data_list:
		line = line.split()
		label.append(int(line[0]))
		line.pop(0)
		
		feat = []
		val = []
		row = {}

		for i in line:
			i = i.split(":")
			feat.append(i[0])
			val.append(i[1])
		row = OrderedDict(zip(feat, val))

		data[num] = row
		num = num + 1

	return data, label


def cross_valisation_naive(smooth):
	cv_result = []
	for i in range(0,5,1):

		temp_train = []
		temp_test = []		

		if i != 0:
			with open("Dataset/CVSplits/training00.data") as f1:
				read1 = f1.readlines()
			temp_train = temp_train + read1
		else:
			with open("Dataset/CVSplits/training00.data") as f1:
				read1 = f1.readlines()
			temp_test = temp_test + read1

		if i != 1:
			with open("Dataset/CVSplits/training01.data") as f2:
				read2 = f2.readlines()
			temp_train = temp_train + read2
		else:
			with open("Dataset/CVSplits/training01.data") as f2:
				read2 = f2.readlines()
			temp_test = temp_test + read2

		if i != 2:
			with open("Dataset/CVSplits/training02.data") as f3:
				read3 = f3.readlines()
			temp_train = temp_train + read3
		else:
			with open("Dataset/CVSplits/training02.data") as f3:
				read3 = f3.readlines()
			temp_test = temp_test + read3

		if i != 3:
			with open("Dataset/CVSplits/training03.data") as f4:
				read4 = f4.readlines()
			temp_train = temp_train + read4
		else:
			with open("Dataset/CVSplits/training03.data") as f4:
				read4 = f4.readlines()
			temp_test = temp_test + read4

		if i != 4:
			with open("Dataset/CVSplits/training04.data") as f5:
				read5 = f5.readlines()
			temp_train = temp_train + read5
		else:
			with open("Dataset/CVSplits/training04.data") as f5:
				read5 = f5.readlines()
			temp_test = temp_test + read5

		training_set, training_label = create_data_list(temp_train)
		test_set, test_label = create_data_list(temp_test)


		postive_matrix = np.empty([len(training_set), 16], dtype=float)
		negative_matrix = np.empty([len(training_set), 16], dtype=float)	
	
		for i in range(CV_EPOCH):		
			updated_positive, updated_negative = naive_train(training_set, training_label, postive_matrix, negative_matrix, smooth)

		predicted_label = naive_test(test_set, test_label, updated_positive, updated_negative)

		accu = accuracy(test_label, predicted_label)

		cv_result.append(accu)

	return cv_result


smooth_factor = [2, 1.5, 1.0, 0.5]

def wrapper_naive():
	print ('CROSS-VALIDATION NAIVE BAYES')
	maximum = 0

	for i in smooth_factor:
		cv_out = cross_valisation_naive(i)
		average = round((sum(cv_out) / len(cv_out)),4)
		
		if (average > maximum):
			maximum = average
			best_smooth = i
			
	print ("Hyper parameters: smoothing parameter %s and its Accuracy %s" % (best_smooth, maximum))
	
	print ('\nNaive Bayes')

	train_set, train_label = create_data(sys.argv[1])
	test_set, test_label  = create_data(sys.argv[2])

	postive_matrix = np.empty([len(train_set), 16], dtype=float)
	negative_matrix = np.empty([len(train_set), 16], dtype=float)

	updated_postive, updated_negative = naive_train(train_set, train_label, postive_matrix, negative_matrix, best_smooth)
	predicted_label = naive_test(test_set, test_label, updated_postive, updated_negative)
	result = accuracy(test_label, predicted_label)

	#for i in predicted_label:
	#	print i

	print ("Best smoothing %s and its test_set accuracy is %.4f" %( best_smooth, result))

wrapper_naive()
