# Twitter-Spambot-Detection
Programs are written in python2.7 and python3.6

Directory structure 
===================
```
Dataset
    Contains twitter dataset
Classifiers
    Algorithms used in this project,
    1. Perceptron (Dynamic, Aggressive)
    2. Supper Vector Machines
    3. Logistic Regression
    4. Naive Bayes
    5. K-Nearest-Neighbors
```
Program file information
====================
```
	1. 'perceptron.py' contains both dynamic and aggressive perceptron algorithm, and it needs development set.
	2. 'svm.py' is the python program file for SVM, and it requires development set.
	3. 'logistic.py' is the python program file for Logistic Regression and it requires development set.
	4. 'naive_bayes.py' is the python program file for Naive Bayes classification.
	5. 'knn.py' is the python program file for K-Nearest-Neighbors.
```
How to run
=====
cd Classifiers
```
eg. python perceptron.py Dataset/data.train Dataset/data.eval Dataset/data.test
```
