#Name: Judy Jinn
#Assignment 1

#goal of assignment
#   1) create models and support vector machines from training sets of different sizes and test them on a subset of data
#   2) create a confusion matrix of the results of #1
#   3) perform k cross-validation tests to find the best parameter C to use for creating your model
#   4) repeat 1-3 for classifying spam e-mails


#import every package you'll need
import os #to change directories
import scipy.io as scio
from scipy.io import loadmat #importing matlab file
import numpy as np #to use arrays
from sklearn import svm #to perform svms
import matplotlib.pyplot as plt #to plot matrices
import pandas as pd #to use data frames
from sklearn.metrics import confusion_matrix #to use confusion matrices




#Check directory
os.getcwd() #get directory, should be the one the py file is saved
os.chdir("./data./digit-dataset") #change it


#load the matlab files with the samples for training
digits = loadmat("./train.mat") #training data with labels
digits_test = loadmat("./test.mat") #test data to submit to kaggle, no labels

#Formatting data for sklearn so it can be put into function. Training must be in 2D array, each row one sample. Labels must be a single column
labels = list(digits["train_labels"]) #pull out labels from dictionary
labels = np.array(labels)   #change to array to be used

#pull out images from dicationary, reshape to row form and 2D matrix
numbers = list(digits["train_images"])  #pull from dictionary
numbers = np.array(numbers)     #create an array with the data
numbers = numbers.reshape(1,784,60000) #change into rows instead of 28x28 data
numbers = numbers.swapaxes(1,2) #swap two axes
numbers = numbers.reshape(60000,784) #make it just 2D
numbers = numbers/255 #normalize data so pixel values are now 0-1

#concatenating labels and images together into training set along 1 axis to create a row of 785 elements, last column being the label
training_digits = np.concatenate((numbers, labels), axis=1)

#shuffle the training data so it can be used later for the SVM
np.random.shuffle(training_digits)



#############################
## --- Problem 1 and 2 --- ##
#############################



#Create a matrix of training sizes to test
tr_grp_size = np.array([100, 200, 500, 1000, 2000, 5000, 10000]).reshape(1,7)
#

#trainingSVM will iterate over several training sizes and test against a validation set with a size chosen by the user
#   tr_grp_size = matrix that holds the training groups sizes for the model to use
#   samples = the array that holds all the samples to be partitione into the training and validation sets

def trainingSVM (tr_grp_size, samples, v_size):
    results = []
    validation_set = samples[:v_size,:-1]        #splice out samples for validation set
    validation_labels = samples[:v_size, -1]       #set the validation's sets respective labels
    train_set = samples[v_size:]            #the rest of the data is the training set
    for size in np.nditer(tr_grp_size):     #must use np.nditer to loop through a np array
        np.random.shuffle(train_set)     #always shuffle prior to withdrawing training set
        train = train_set[:size,:-1]    #isolates training images
        labels = train_set[:size,-1]
        results.append(Q1linsvc(size, train, labels, validation_set, validation_labels))
    return results


#function for linear svcs for question1
    #tr_size: a matrix that holds the sizes of the groups for the model to be trained on
    #train: the data to be analyzed, should be indexed to only include data to be trained
    #labels: labels associated with training data only
    #testset: the subset of data matrix to be used as validation
    #testlabes: the labels assoiate with the validation set
def Q1linsvc(size, train, labels, validation_set, validation_labels):
    clf = svm.LinearSVC(multi_class='ovr')   #this creates a new model to train with a c value we designate as the one being passed in from our previously made C_parameters array (or just enter your own to test)
    clf.fit(train, labels)    #we train our model based on some data and labels
    test = clf.predict(validation_set)  #test on our validationset to grab an array of predicted results
    accuracy = sum(test == validation_labels)/len(validation_set) #this compares the predicted results to the true results that we know, then divides by the size of the validationset to give us an accuracy percentage, which we store and return
    print ("Accuracy of training group size ", size, " = ", accuracy) #in case you want to print out results as everything runs.
    confusionmatrix(size, validation_labels, test) #saves the confusion matrices to your folders as you run the function
    return accuracy 

#confusionmatrix prints out the confusion matrices for each training group size. It is implented within the Q1linsvc function
#   size: takes the size of the curren training set. Does nothing except name the file
#   validation_labels: these are the labels associated with the validation test set used in the linsvc
#   test: this is the saved predicted test labels from the Q1linsvc that will be used to build the confusion matrix
def confusionmatrix(size, validation_labels, test):    
    cm = confusion_matrix(validation_labels, test)
    #coloring confusion matrix
    plt.matshow(cm)
    plt.title("Confusion matrix for "+str(size)+" image training set", y=1.08)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.ylabel('True label')
    plt.yticks([0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('Predicted label')
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.savefig("test"+str(size)+".png")


#run the functions with your data!
SVMtraining = []       #store the results of trainingSVM in a new matrix
for i in range(1):     #how many times do you want to run function to save the results
   SVMtraining.append( trainingSVM (tr_grp_size, training_digits, 10000) ) #runs function
SVMtraining = np.array(SVMtraining) #create an array

#plot the data!
plt.plot(tr_grp_size, SVMtraining, "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
xlabels=[100, 200, 500, 1000,2000,5000,10000] #labeling the axes
plt.title("Classification Accuracy for Various Digit Training Group Sizes", y=1.02)#configure X axes
plt.xlabel('Training Group Sizes')
plt.xlim(0, 11000)
plt.xticks([100, 200, 500, 1000,2000,5000,10000]) #specificies x tick marks
plt.xticks(rotation=320, size='small') #rotates x axis labels
#configure  Y axes
plt.ylabel('Accuracy for Test Group Size 10000')
plt.ylim(.5,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('Q1training2.png')



#######################
## --- Problem 3 --- ##
#######################


#reshuffle 
np.random.shuffle(training_digits)
digitsubset = training_digits[:10000]


#this is the matrix that has all the values of C to test
C_parameters = ([(10**(-4)), (10**(-3)), (10**(-2)), (10**(-1)), 1, (10**(1)), (10**(2))])
#c1=[(.05, .1, .15, 1, 1.5)]
#c2=[(.09, .1, .11, .12, .13)]
#c3=[(.12)] #picked .12 for kaggle

    
##----------------------- Functions -------------------------------------------

#K_fold function that trains and tests lienar SVC over some set of data over a series of different C values.
#    data: a matrix of mxn of data to be trained/tested
#    labels: a matrix of mx1 of labels for the data
#    k: the number of folds to be tested
#    C_value: a mx1 matrix that contains values of C to be tested
def k_fold(samples, k, C_parameters):
    results_rows = len(C_parameters)    #use this value to create the # of rows for the empty matrix where the first element of each row will eventually hold each c_value
    results=np.zeros(shape=(results_rows,k+1), dtype=np.float)  #creates and empty array with rows = #of C values and columns equal to the number of partitions (test groups) that will be tested. Each element will hold the accuracy for the test 
    
    for c_value in C_parameters:    #iterate through all your C values starting with the first one in the matrix
        Cposition = C_parameters.index(c_value)     #this finds the position of the c_value in the matrix of c values which will be used to determine which row in your empty array it will sit (thus ensuring each c value has its own row of accuracies)
        results[Cposition,0]= c_value   #this puts the current c value into the appropriate row of your empty array.
        num_rows = len(samples)        #number of data being analyzed (eg we have 60000 images)
        row_block = (num_rows/k)    #the size of each partition based on how large the data is. (e.g. 100 data in 100 blocks = 10 data per block)
        print("C = ", c_value)        #when it runs this will help keep track of  which parameter is being tested
        for validationgroup in range(0,k):      #this loop only runs for how many partitions (k) are created. Validation group is the one being tested after training
                
                indexFirst= row_block * validationgroup     #this marks the element number which will be the beginning of a validation group. (e.g for validation group = 0, row_block = 60000*0 = 0, so our first element begins at 0)
                indexSecond = row_block * (validationgroup+1) #this marks the end of our block. so this will be equivalent to 6000*1 = 60000 for our first iteration in the loop
                
                validationdata = samples[indexFirst:indexSecond,:-1] #taking our index1 and index2, we now can tell the array that test group one is the values between 0 and 6000
                validationlabels = samples[indexFirst:indexSecond,-1] #this is the same for the labels
                trainingdata = np.vstack(( samples[:indexFirst,:-1] , samples[indexSecond:,:-1] )) #thus our training data is the concatenation of all the data prior to the validationdata and after the validationdata. (eg with first iteration, nothing comes before 0, but there is all the data after 6000. For the second iteration it will take 0-6000 and concatenate it with 12000-600000)
                traininglabels = np.hstack(( samples[:indexFirst,-1] , samples[indexSecond:,-1] ) ) #does same thing as prior
                
                results[Cposition, validationgroup+1] = k_fold_linsvc(trainingdata, traininglabels, validationdata, validationlabels, c_value, validationgroup) #this takes all the data we just assigned and then puts it into the k_fold_linsvc function. Our results, the accuracy for that validationgroup, given by our trained model will then be stored in the row of the respective C value being used, and in the column respective to the validation set just tested. (e.g. we ran a test on the 3rd c value in the matrix, and we are only validationset 5, thus the accuracy of our test results will be stored in the resultsarray[2,5])
    print (results) #and at the very end of all our c values and validation sets, we should print out array to make sure the entire thing was filled in.      
    return results


#function for linear svcs for the k_fold function
    #train: the data to be analyzed, should be indexed to only include data to be trained
    #labels: labels associated with training data only
    #testset: the subset of data matrix to be used as validation
    #testlabes: the labels assoiate with the validation set
    #c_value: this should not be modified
    #
def k_fold_linsvc(trainingdata, traininglabels, validationdata, validationlabels, c_value, validationgroup):
    clf = svm.LinearSVC(C = c_value, multi_class='ovr')   #this creates a new model to train with a c value we designate as the one being passed in from our previously made C_parameters array (or just enter your own to test)
    clf.fit(trainingdata, traininglabels)    #we train our model based on some data and labels
    test = clf.predict(validationdata)  #test on our validationset to grab an array of predicted results
    accuracy = sum(test == validationlabels)/len(validationdata[:len(validationdata)]) #this compares the predicted results to the true results that we know, then divides by the size of the validationset to give us an accuracy percentage, which we store and return
    print ("C parameter = ", c_value, "Accuracy of test group ", validationgroup, " = ", accuracy, "\n") #in case you want to print out results as everything runs.
    return accuracy 
    
##----------------------- End Functions -------------------------------------------
    
    
    
##----------------------- Run the Functions -------------------------------------------    
#Run the k cross validation set and store the results!
k_fold_results = []    #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
    k_fold_results.append(k_fold(digitsubset, 10, C_parameters)) #runs function
k_results = k_fold_results[0] #the previous script saves the array as a list, so this picks out the array

Cmeans = k_results[:,1:].mean(axis=1).reshape(7,1)     #find the means of the rows, store as array

FinalResultsDigits = np.concatenate((k_results, Cmeans), axis=1)     #concatenate your results
np.savetxt("FinalResultsDigits.csv", FinalResultsDigits, delimiter=",") #save the file
    
##plot the data!
plt.plot(C_parameters, Cmeans, "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
plt.xscale("log")
plt.title("Accuracy for Various C Parameters", y=1.02)#configure X axes
plt.xlabel('C values')
plt.xlim(10e-5, 10e2)
#configure  Y axes
plt.ylabel('Accuracy')
plt.ylim(.6,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('C_parametersdigits.png')

############################
## --- Digits Kaggle  --- ##
############################

#pull out test images from dicationary, reshape to row form and 2D matrix
test_images = list(digits_test["test_images"])
test_images = np.array(test_images)
test_images = test_images.reshape(1,784,10000)
test_images = test_images.swapaxes(1,2)
test_images = test_images.reshape(10000,784)
test_images = test_images/255

#shuffle the training images again
np.random.shuffle(training_digits)

def test_linsvc(samples, testset, c_value):
    train = samples[:,:-1] #split training data for the classifier
    labels = samples[:,-1]
    clf = svm.LinearSVC(C = c_value, multi_class='ovr')   #this creates a new model to train with a c value we designate as the one being passed in from our previously made C_parameters array (or just enter your own to test)
    clf.fit(train, labels)    #we train our model based on some data and labels
    test = clf.predict(testset)  #test on our validationset to grab an array of predicted results
    return test

##----------------------- Run the Functions ------------------------------------------- 
#Run the function with selectd C paramater value
test_images_predictions = []  #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
   test_images_predictions.append( test_linsvc(training_digits, test_images, 1.2) ) #runs function
test_images_predictions= test_images_predictions[0].astype(int)

#create a dataframe that has one column for the IDs and another for the predicted values. 
predictions = pd.DataFrame({"A": range(1,len(test_images)+1), 
                            "B": test_images_predictions})
predictions.columns = ['Id', 'Category'] #rename columns
predictions.to_csv("JinnJdigits3.csv", sep=',', index=False, line_terminator = "\r") #save data as csv in proper format







#######################
## --- Problem 4 --- ##
#######################
import os
os.chdir("./data./spam-dataset") #change to proper directory

import featurize #to import a py file, copy paste the blank ___inuit___.py file to your directory and make sure both inuit and py file are there. This specific module has a problem on line 18 where it must be edited to say with open(filename, encoding="utf8") as f: to work

##load the matlab files with the samples for training
spam_data = loadmat("./spam_data.mat") #training data with labels



#--------------------------Setting up the Data ----------------------------

#pull out relevant data
test_email = np.array(list(spam_data["test_data"])) #pull out test email, create np array
email = np.array(list(spam_data["training_data"])) #pull out email, create np array
email_labels = np.array(list(spam_data["training_labels"])).reshape(5172,1) #pull out labels for the e-mails 1=spam, 0=ham; create nparray, make column

#concatenating labels and images together into training set along 1 axis to create a row of 785 elements, last column being the label
training_email = np.concatenate((email, email_labels), axis=1)

#shuffle the training data so it can be used later for the SVM
np.random.shuffle(training_email)




--------------------------Testing with a single svc----------------------------


#Quick test of the svc to make sure it works
def emailSVC (samples, tr_grp_size):
    e_train = training_email[:tr_grp_size,:-1]
    e_labels = training_email[:tr_grp_size,-1]
    e_test = training_email[tr_grp_size:, :-1]
    e_testlabels = training_email[tr_grp_size:, -1]
    clf = svm.LinearSVC(multi_class='ovr')  
    clf.fit(e_train, e_labels)    
    test = clf.predict( e_test)  
    accuracy = sum(test ==  e_testlabels)/len(e_test) 
    print ("Accuracy of training group size ", tr_grp_size, " = ", accuracy)
    return


------------------------- K Cross-validation ----------------------------------

#reshuffle 
np.random.shuffle(training_email)

#this is the matrix that has all the values of C to test for spam
spam_c1 = ([(10**(-5)), (10**(-4)), (10**(-3)), (10**(-2)), (10**(-1)), 1, (10**(1)), (10**(2))])
spam_c2 = ([1, 10, 50, 100, 150])
spam_c3 = ([45, 50, 55, 60, 65]) 
spam_c4 = ([52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]) 
spam_c5 = ([61,62,63,64,65,66,67,68,69]) 
spam_c6 = ([61,62,63,66,67,68,69]) 
spam_c7 = ([67,68,69,70,71,72,73]) #Doesn't really seem to be some optimal C parameter.. it's all in this 50-70 range. Chose 68
spam_c = ([68])


#all the previously defined functions may be used for the spam/ham crossval


##----------------------- Run the Functions ------------------------------------------- 
#Run the k cross validation set and store the results!
k_fold_results = []    #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
    k_fold_results.append(k_fold(training_email, 12, spam_c1)) #runs function
k_results = k_fold_results[0] #the previous script saves the array as a list, so this picks out the array

spam_c1means = k_results[:,1:].mean(axis=1).reshape(len(spam_c1),1)     #find the means of the rows, store as array

FinalCResultsSpam = np.concatenate((k_results, spam_c1means), axis=1)     #concatenate your results
np.savetxt("FinalResultsSpam.csv", FinalCResultsSpam, delimiter=",") #save the file

#
#plot the data!
plt.plot(spam_c1, spam_c1means, "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
plt.xscale("log")
plt.title("Accuracy for Various C Parameters for Spam Data", y=1.02)#configure X axes
plt.xlabel('C values')
plt.xlim(10e-5, 10e2)
#configure  Y axes
plt.ylabel('Accuracy')
plt.ylim(.6,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('C_parametersSpam.png')

#------------------------- Predicting Testing Data  ----------------------------------

#reshuffle 
np.random.shuffle(training_email)

#Run the function with selectd C paramater value
test_email_predictions = []  #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
   test_email_predictions.append( test_linsvc(training_email, test_email, 68) ) #runs function, last number is c value of choise
test_email_predictions= test_email_predictions[0].astype(int)

#create a dataframe that has one column for the IDs and another for the predicted values. 
email_predict = pd.DataFrame({"A": range(1,len(test_email)+1), 
                            "B": test_email_predictions})
email_predict.columns = ['Id', 'Category'] #rename columns
email_predict.to_csv("JinnJemail4.csv", sep=',', index=False, line_terminator = "\r") #save data as csv in proper format






