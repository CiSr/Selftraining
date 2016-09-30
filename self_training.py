#Self Training code

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split 
from sklearn.naive_bayes import GaussianNB

print "\nloading the training data........\n"
load_data=genfromtxt("feature_vectors.txt", delimiter=" ")
reload_data=genfromtxt("labels.txt",delimiter= " ")
true_label=genfromtxt("test_label.txt")
#load the test data
test_data=genfromtxt("test.txt", delimiter=" ")
[a,b]=load_data.shape

train_data=load_data[:,1:a]
label_data=reload_data[:,1]

[c,d]=train_data.shape


[row_test, col_test]=test_data.shape
#true_label=reload_data[:,0]

print "\nThe size of test data is:"
print row_test; print col_test

clf=SVC(probability=True)
#clf=Perceptron()
clf.fit(train_data,label_data)
actual_label=clf.predict(test_data)

#Finding the probabilities of test labels

confidence_label_svm=clf.predict_proba(test_data)
#plt.plot(confidence_label_svm[:,0],confidence_label_svm[:,1],'x');plt.axis('equal');plt.show();
#plt.plot(train_data);plt.show();
#plt.axis('equal');plt.show();

#Building a model using Random Forest Classifier


init=RandomForestClassifier(n_estimators=3)
init.fit(train_data,label_data)
label_test=init.predict(test_data)
confidence_label_rf=init.predict_proba(test_data)


def measures(true_label, actual_label):
	print "\n The accuracy score is:"
	print  accuracy_score(true_label,actual_label) 
	print "\nThe confusion_matrix is:"
	print  confusion_matrix(true_label, actual_label)	
	print "\nThe precision_score"
	print  precision_score(true_label, actual_label)
	print "\nThe recall score"
	print recall_score(true_label, actual_label)

print "\t\t**** SVM classifier ***" 
measures(true_label,actual_label)

print "\t\t****RandomForestClassifier ***"
measures(true_label, label_test)

tusk=np.empty((row_test,1))
Topkprobs=np.empty((row_test,1))
Topkprobr=np.empty((row_test,1))


append_label_array=np.empty((row_test,1))



for i in range(row_test):
	  store=actual_label[i]
	  append_label_array[i].fill(store)


#Finding the top maximum score in the labels.

def find_max(confidence_label):
	for i in range(row_test):
		temp=max(confidence_label[i,:]) 
		tusk[i].fill(temp)
	return tusk		

#Topkprobs=find_max(confidence_label_svm)


TopKprobr=find_max(confidence_label_rf)


intermediate_array=np.empty((row_test,col_test+1))
intermediate_confidence_array=np.empty((row_test, col_test+2))


intermediate_array=np.append(test_data,append_label_array,1)
intermediate_confidence_array=np.append(intermediate_array,Topkprobr ,1)
final_array=np.empty((row_test,col_test+2))

#Sort the test data set by the most confident labels

final_array=intermediate_confidence_array[np.argsort(intermediate_confidence_array[:,col_test+1])]

#print final_array


#This function just passes the test data to the training set

def find(new_train, new_label):
	cll=SVC(probability=True)
	#cll=RandomForestClassifier(n_estimators=3)
	#cll=Perceptron
	cll.fit(new_train, new_label)
	actual_label1=cll.predict(test_data)
	print "\n The selftraining  triplet <A, P, R> is: "
	print accuracy_score(true_label, actual_label1)
	print  precision_score(true_label, actual_label1)
	print recall_score(true_label, actual_label1)
	print confusion_matrix(true_label, actual_label1)


for k in range(0,500,50):
	new_train=np.concatenate((train_data,final_array[row_test-k:row_test,0:col_test]))
	new_label=np.concatenate((label_data,final_array[row_test-k:row_test,col_test+1]))

	#new_train=np.concatenate((train_data,final_array[1:k,0:col_test]))
	#new_label=np.concatenate((label_data,final_array[1:k,col_test+1]))
	#Select range determines how much top k to add.
	select_range=(row_test-(k-1))
	find(new_train,new_label)
	print k
	[xrow,xcol]=new_train.shape
	k=((k/10)*xrow)
#new_test=intermediate_array[0:select_range,0:col_train]




