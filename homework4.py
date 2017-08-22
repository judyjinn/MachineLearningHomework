"""
Name: Judy Jinn
Assignment 4
"""

""" 
####################
## --- Set up --- ##
####################
"""

"""import every package you'll need"""
import os #to change directories
import scipy.io as scio #scipy
from scipy.io import loadmat #importing matlab file
import numpy as np #to use arrays
import matplotlib.pyplot as plt #to plot matrices
import pandas as pd #to use data frames
import math
import sklearn as sk
from sklearn import preprocessing
import pickle
import random
import pylab 
import scipy.stats as stats


"""Check directory"""
os.getcwd() #get directory, should be the one the py file is saved


"""
#####################
 --- Problem 1 --- ##
#####################
"""
"""Using the X array given to us, use Newton's Method to find the new Us and Betas"""

B0 = (np.asmatrix([-2, 1, 0]).reshape(1,3)).T #our original beta matrix
X = np.matrix(([1, 0, 3], [1, 1, 3], [1, 0, 1], [1, 1, 1])) #our original X matrix with 1s appended onto the end
Y = np.asmatrix([1, 1, 0, 0]) #our original Y values
lam = 0.07 #also given to us

#this will find our new U each time.
def find_u(B, X):
    U = np.asarray(((1./(1+(np.exp(np.dot(-B.T, X.T)))))))
    return U


def Diagonalize(U):
    one_u = 1 - np.asarray(U)
    U = [np.asarray(U)*one_u for U, one_u in zip(U, one_u)]
    D = np.diag(np.asarray(U[0]))
    #print ("D = ", D)
    return D
    
def newton(steps, B, X, Y,lam):
    B_t = B


    print ("B_t is = ", B_t)
    for step in range(0, steps):
        U=find_u(B_t, X)
        print ("U_{0} is = ".format(step), U)
        
        deriv2 = (np.linalg.inv((2*lam*np.identity(3))+(X.T*Diagonalize(U)*X)))
        deriv1 = (2*lam*B_t - np.dot(X.T, (Y.reshape(4,1))-(U.T)))
        #print ("derivative of 1 = ", deriv1)
        #print ("derivative of 2 = ", deriv2)
        B_t = B_t - (deriv2 * deriv1)
        print ("B_{0} =".format(step+1), B_t)
    return
    
newton(2, B0, X, Y, lam)


"""
#####################
 --- Problem 2 --- ##
#####################
"""
""" Using a music data file that has features containing the musical tones, tenors, etc of songs, predict which year a song originated from """

os.chdir("./Music") #change the working directory
#
###music = np.genfromtxt("YearPredictionMSD.txt", delimiter=",", dtype="float64") #reads the data txt file into a np.matrix

#to speed up loading of the txt file, pickle the data
###music_dict = {'a': music} #put all the data into a dictionary under a key name ("a")
###pickle.dump(music_dict, open("music2", "wb")) #pickle dump it

music_dict = pickle.load(open("music2", "rb")) #load the pickled data into a dictionary for use
music = np.asarray(list(music_dict["a"])) #pull out the relevant data
music_train = music[:463715,] #separate the data into train and test accoring to what was given to us by the file instructions
music_test = music[463715:,]

#split the training data into the data and the years (labels)
music_x = music_train[:, 1:] #takes out the data, which is everything but column one
music_one = np.ones(len(music_x)).reshape(len(music_x),1) #adding a column of 1s in order to set the B0
music_x = np.concatenate((music_x, music_one), axis=1) #adds columns to the END of the data set. Important. this is how the linalg.solve function must take the 1s column

music_y = music_train[:, 0] #includes just the years

#do the same for the test set
music_testx = music_test[:, 1:]
music_testone = np.ones(len(music_testx)).reshape(len(music_testx),1) 
music_testx = np.concatenate((music_testx, music_testone), axis=1)
music_testy = np.asmatrix(music_test[:, 0]).T 


"""linear system of eqns for solving for B = X'XB = X'y
    this is solvable by using the function linalg.solve
    Everything below is organizing the data such that linalg.solve will work"""
#finding the Residual sum of squares (RSS) to use for modeling a linear regression. We must use linalg.solve to find the beta coefficients first.
block = len(music_x)/5 #need to break up music_x to resolve memory error from the dot product function. I did it in chunks of 5

x_sub = [] #storing all the chunks of the data as a list of arrays
for sub in range(0,5):
    x_sub.append(music_x[(block*sub):(block*(sub+1)),]) #this is breaking data up into block size we designated

#find the dot product of the data set in chunks
X_Xt = []
for dot in range(0,5):
    X_Xt.append(np.dot(x_sub[dot].T, x_sub[dot]))

X_Xt = sum(X_Xt) #sum up all 5 X'X dot product matrices to get the total X_Xt matrix


Xt_y = np.asmatrix(music_x.T) * np.asmatrix(music_y).reshape(463715,1) #finding X'y to use for linalg.solve. Make sure to convert np.arrays to matrices so it works.

B_coeff = np.linalg.solve(X_Xt, Xt_y) #finally solved for beta using linalg.solve

""" a linear regression is just y = Bx + Bx.... """
#We have our trained linear regression so test what years we would predict given the test set
est_y = np.dot(music_testx, B_coeff) #flipped the X and B matrix around so the dot product would work. Doesn't make a difference.

##RSS of test values
resid = np.asmatrix(music_testy - est_y) #now that we have our estimated years, subtract the difference from the actual year labels we have and find the residuals

#and now we find the residual sum of squares
RSS_test = resid.T * resid
"""RSS = 4669580.17950477"""


#range of the predicted values (the years predicted)
min_y = min(est_y) 
max_y = max(est_y) 
""" minimum [[ 1953.85419386]]"""
"""maximum [[ 2045.54569461]]"""

#plot the coeff
x = np.arange((len(B_coeff)-1)) 
plt.plot(x, B_coeff[:-1], "ro") #did not take the B0 because it was skewing the entire graph noted its value below
plt.title("Values of Beta", y=1.02)
plt.yscale("log")
plt.ylabel('B Coefficients')
plt.show()

""" Where B0 was equal to 1951.22""" # removed from the graph
#

#plot of residuals
plt.plot(music_testy, resid, "ro")
plt.title("Residuals of esimated years", y=1.02)
plt.ylabel("Residual value")
plt.xlabel('Song Year')
plt.show()


"""
#####################
 --- Problem 3 --- ##
#####################
"""

"""Problem 3 - Use different batch descent gradient methods (batch, stochastic, and variable learning rate) to minimize data that has been standardized, log transformed, and binarized"""

spam_data = loadmat("./spam.mat") #training data with labels

#pull out relevant data
email_train = np.asarray(list(spam_data["Xtrain"])) #pull out email, create np array
email_labels = spam_data["ytrain"] #pull out labels for the e-mails 1=spam, 0=ham; create nparray, make column

###standardize the columns
email_scale = np.asmatrix(sk.preprocessing.scale(email_train, axis=0, with_mean=True, with_std=True, copy=True)) #centers features around their mean and also converts the elements to unit variance

#log transform the data
email_log = np.asmatrix(np.log(email_train + 0.1))

##binarize
email_binary = np.asmatrix(sk.preprocessing.binarize(email_train, threshold=0.0, copy=True)) #binarize where values above 0 are 1 and below are 0

#


"""---3(1)----"""
#the functions below are based on newton's method and gradient descent. They are derived from problem 1 from the homework.

#find_u : this function is used to find the vector of Mu. It should output an array of 1xn where n is the length of the samples
#    B: the beta vector used to calvlate mu. Should be a mx1 where m is the number of features
#    X: the design matrix. Should be a mxn matrix where m is the number of samples and n is the number of features
def find_u(B, X):
    U = ((1./(1+(np.exp(np.dot(-np.asmatrix(B.T), np.asmatrix(X.T)))))))
    return U
    
#batch_descent : This calculates batch gradient descent. It takes 6 arguments and outputs a tuple with 4 lists consisting of all the beta matrices, the loss, all calculated mu vectors, and the iterations
#    B: the inital beta vector. Should be a mx1 where m is the number of features
#    step: the step sized desired
#    X: the design matrix. Should be a mxn matrix where m is the number of samples and n is the number of features
#    Y: the labels associated with the samples. Should be a mx1 matrix
#    lam: the regularization parameter lambda. Should be an integer or float
#    iterate: the number of iterations desired
def batch_descent(B, step, X, Y, lam, iterate):
    all_B = [B] #a list which will hold all calulated betas. Initialized with the starting beta vector.
    neglogL = [] #a list which will hold all the loss values
    U_all = []    # a list which will hold all the calculated mus
    
    B_t = B #beta initailized to what was selected by user
    

    for i in range(0, iterate): #for each iteration desired
        U=find_u(B_t, X) #find the mu vector associated with the initalized B, then on the next will find the mu based on the new beta vector
        U_all.append(U) #store the mu  

        deriv1 = (2*lam*B_t) - (np.dot(X.T, (Y - (np.asmatrix(U).reshape(len(X),1))))) #the derivative of the negative log likelihood from problem 1 in hw 4

        B_t = B_t - (step*deriv1) #update the beta vector using the arbitrary step size and derivative
        all_B.append(B_t) #put it into the list of betas
   
        neglog = (lam*B_t.T*B_t) - ((Y.T*np.log(np.asmatrix(U).T)) + ((1-Y.T)*np.log(1-np.asmatrix(U).T))) #calculate the loss using the original negative log likelihood function
        neglogL.append(neglog.item(0)) #store that value of loss into the list
      
    return all_B, neglogL, U_all, i


#lambda, initual beta, and step size are completely arbitrary. 
batch_lam = 1.0
email_B0 = np.asmatrix(np.zeros(len(email_train[0]))).T

#calculate the scaled data
batch_scale = batch_descent(email_B0, 0.00001, email_scale, email_labels, batch_lam, 1000)
#plot the data
x = np.arange(1, batch_scale[3]+2)
y = batch_scale[1]
plt.plot(x,y)
plt.title("Batch Gradient Descent (standardized data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()
  
  #calculate the log data
batch_log = batch_descent(email_B0, 0.00001, email_log, email_labels, batch_lam, 1000)
#plot the data
x = np.arange(1, batch_log[3]+2)
y = batch_log[1]
plt.plot(x,y)
plt.title("Batch Gradient Descent (log transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()

#calculate the binarized data
batch_binary = batch_descent(email_B0, 0.00001, email_binary, email_labels, batch_lam, 1000)
x = np.arange(1, batch_binary[3]+2)
y = batch_binary[1]
plt.plot(x,y)
plt.title("Batch Gradient Descent (binary transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()


"""---3(2)----"""   

def find_u(B, X):
    U = ((1./(1+(np.exp(np.dot(-np.asmatrix(B.T), np.asmatrix(X.T)))))))
    return U

#stochastic_descent : This calculates batch gradient descent. It takes 6 arguments and outputs a tuple with 4 lists consisting of all the beta matrices, the loss, all calculated mu vectors, and the iterations
#    B: the inital beta vector. Should be a mx1 where m is the number of features
#    step: the step sized desired
#    X: the design matrix. Should be a mxn matrix where m is the number of samples and n is the number of features
#    Y: the labels associated with the samples. Should be a mx1 matrix
#    lam: the regularization parameter lambda. Should be an integer or float
#    iterate: the number of iterations desired
def stochastic_descent(B, step, X, Y,lam, iterate):
    all_B = [B]
    neglogL = []
    U_all = []    
    
    B_t = B
    
    #print ("B_t is = ", B_t)
    a=1 #keeping count of our iterations
    for i in range(0, iterate):
     
        rand = random.randrange(0, len(X)) #stochastic descent randomly uses features to update beta. so randomly pick a sample usinga random number generator set to the length of the samples
#        print("rand = ", rand)
        yi = np.asarray(Y.item(rand)) #find the label associated with the random sample. Will be a scalar
        xi = np.asmatrix(X[rand,:]) #get the features associaed with the random sample. Will be a 1xn vector
        ui = np.asarray((B_t.T*X[rand,:].T).item(0)) #calculate the mu associated with that particular sample. Will be scalar
        #print ("U_{0} is = ".format(step), U)
        
        stoch = 2*lam*B_t.T - (yi - ui)*xi.T #stochastic descent calculations
        #print ("derivative of 1 = ", deriv1)
        
        B_t = B_t - (step*stoch) #update your beta
        all_B.append(B_t) #append it.
        #print ("B_{0} =".format(step+1), B_t) 
        
        #calculating the loss is the same as for batch gradient. Requires full matrices
        U=np.asmatrix(find_u(B_t, X).T) #calculate teh full mu vector now associated to the updated beta
        U_all.append(U) #append the vector
        
        neglog = (lam*B_t.T*B_t) - ((Y.T*np.log(np.asmatrix(U))) + ((1-Y.T)*np.log(1-np.asmatrix(U)))) #uses the same log likelihood function to find loss
        neglogL.append(neglog.item(0))
        a+=1  
  
    return all_B, neglogL, U_all, a


lam = 0.001
email_B0 = np.asmatrix(np.zeros(len(email_train[0]))).T
#
stoch_scale = stochastic_descent(email_B0, 0.001, email_scale, email_labels, lam, 1000)
#plot the data
x = np.arange(1, stoch_scale[3])
y = stoch_scale[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent (standardized data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()
##  

stoch_log = stochastic_descent(email_B0, 0.001, email_log, email_labels, lam, 1000)
#plot the data
x = np.arange(1, stoch_log[3])
y = stoch_log[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent (log transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()

stoch_binary = stochastic_descent(email_B0, 0.01, email_binary, email_labels, lam, 1000)
x = np.arange(1, stoch_binary[3])
y = stoch_binary[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent (binary transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()


#
"""---3(3)----"""

def find_u(B, X):
    U = ((1./(1+(np.exp(np.dot(-np.asmatrix(B.T), np.asmatrix(X.T)))))))
    return U


#stoch_descent_learn : This calculates stochastic gradeitn descent with variable step sizes which vary proportional with the iterations taken. It takes 6 arguments and outputs a tuple with 4 lists consisting of all the beta matrices, the loss, all calculated mu vectors, and the iterations
#    B: the inital beta vector. Should be a mx1 where m is the number of features
#    step: the step sized desired
#    X: the design matrix. Should be a mxn matrix where m is the number of samples and n is the number of features
#    Y: the labels associated with the samples. Should be a mx1 matrix
#    lam: the regularization parameter lambda. Should be an integer or float
#    iterate: the number of iterations desired
def stoch_descent_learn(B, step, X, Y, lam, iterate):
    all_B = [B]
    neglogL = []
    U_all = []    
    
    B_t = B
    
    #print ("B_t is = ", B_t)
    a=1
    for i in range(0, iterate):
        U=find_u(B_t, X)
        #print ("U_{0} is = ".format(step), U)
        deriv1 = (2*lam*B_t) - (np.dot(X.T, (Y - (np.asmatrix(U).reshape(len(X),1)))))
        #print ("derivative of 1 = ", deriv1)
        B_t = B_t - ((step/a)*deriv1) #only difference is right here. Divide step size by the current iteration
        all_B.append(B_t)
        #print ("B_{0} =".format(step+1), B_t)     
        
        neglog = (lam*B_t.T*B_t) - ((Y.T*np.log(np.asmatrix(U).T)) + ((1-Y.T)*np.log(1-np.asmatrix(U).T)))
        neglogL.append(neglog.item(0))
        a+=1
    
        U_all.append(U)
#      
    return all_B, neglogL, U_all, a
    all_B = [B]
    neglogL = []
    U_all = []    
    
    B_t = B
    
    #print ("B_t is = ", B_t)
    a=1
    for i in range(0, iterate):
     
        rand = random.randrange(0, len(X))
#        print("rand = ", rand)
        yi = np.asarray(Y.item(rand))
        xi = np.asmatrix(X[rand,:])
        ui = np.asarray((B_t.T*X[rand,:].T).item(0))
        #print ("U_{0} is = ".format(step), U)
        
        stoch = 2*lam*B_t.T - (yi - ui)*xi.T
        #print ("derivative of 1 = ", deriv1)
        
        B_t = B_t - ((step/a)*stoch)
        all_B.append(B_t)
        #print ("B_{0} =".format(step+1), B_t) 
        
        U=np.asmatrix(find_u(B_t, X).T)
        neglog = (lam*B_t.T*B_t) - ((Y.T*np.log(np.asmatrix(U))) + ((1-Y.T)*np.log(1-np.asmatrix(U))))
        neglogL.append(neglog.item(0))
        a+=1
    
        U_all.append(U)
    
  
    return all_B, neglogL, U_all, a

lam = 0.001
email_B0 = np.asmatrix(np.zeros(len(email_train[0]))).T

learn_scale = stoch_descent_learn(email_B0, 0.001, email_scale, email_labels, lam, 100)
#plot the data
x = np.arange(1, learn_scale[3])
y = learn_scale[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent Variable Learning Rate (standardized data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()
###  
learn_log = stoch_descent_learn(email_B0, 0.00001, email_log, email_labels, lam, 100)
#plot the data
x = np.arange(1, learn_log[3])
y = learn_log[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent Variable Learning Rate (log transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()
#
learn_binary = stoch_descent_learn(email_B0, 0.001, email_binary, email_labels, lam, 100)
x = np.arange(1, learn_binary[3])
y = learn_binary[1]
plt.plot(x,y)
plt.title("Stochastic Gradient Descent Variable Learning Rate (binary transformed data)", y=1.02)
plt.ylabel('Negative Log Likelihood')
plt.xlabel('Iterations')
plt.show()
#
#
#
#
#
#
