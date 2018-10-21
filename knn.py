import sys
import math

RANGE = 16
KNN = 19

def distance(data, example, neighbour):

	distance = []
	for i in range(len(data)):
		label = data[i][0]
		
		dist = 0
		tmp = []
		for j in range(len(data[i][1])):
			dist += pow((data[i][1][j] - example[1][j]), 2)
		tmp.append(math.sqrt(dist))
		tmp.append(int(label))
		distance.append(tmp)
		
	
	distance.sort()

	positive = negative = 0
	for i in range(neighbour):
		if distance[i][1] == 1:
			positive += 1
		elif distance[i][1] == -1:
			negative += 1


	if positive > negative:
		return 1
	else: 
		return -1


def accuracy(data, predict):
	pr_ct = 0
	pr_wg = 0
	
	for i in range(len(data)):
		label = int(data[i][0])
		
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

def wrapper_knn():
	print 'K-Nearest-neighbours'

	train_set = data_join(sys.argv[1])
	test_set = data_join(sys.argv[2])

	prediction = []
	for example in test_set:
		ret_val = distance(train_set, example, KNN)
		prediction.append(ret_val)

	#for i in prediction:
	#	print i

	result = accuracy(test_set, prediction)
	print ("Value of K is %s and its test_set accuracy is %f" %(KNN ,result))

wrapper_knn()
