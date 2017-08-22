"""
Name: Judy Jinn
Assignment 3
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
from sklearn import svm #to perform svms
import matplotlib.pyplot as plt #to plot matrices
import pandas as pd #to use data frames
from sklearn.metrics import confusion_matrix #to use confusion matrices
from numpy.random import RandomState #to set seeds and save
from scipy.stats import multivariate_normal #to use multivariate normal functions

"""Check directory"""
os.getcwd() #get directory, should be the one the py file is saved


"""
#######################
## --- Problem 1 --- ##
#######################
"""

"""Draw 100 samples from each distribution"""

#Setting up random numbers  using RandomState so each sample is same for each run.
rng1 = np.random.RandomState(1) #for x1
rng2 = np.random.RandomState(2) #for x2

#creating X1
x1_u = 3 #x1 mean
x1_var = 9 #x1 variance
x1_sd = x1_var**0.5 #x1 sd because normal() only takes sd
x1 = rng1.normal(x1_u, x1_sd, 100)
#print(x1[0:3]) #Check to see if numbers are being sampled the same each time.
#plotting x1
count, bins, ignored = plt.hist(x1, 30, normed=True) #create a count and the number of bins used to hold count of data
plt.plot(bins, 1/(x1_sd * np.sqrt(2 * np.pi)) * 
                np.exp( - (bins - x1_u)**2 / (2 * x1_sd**2) ),
          linewidth=2, color='r') # plots the bins we designated as well as a curve for the the function for a normal distribution given our designated means and variance with line width 2 and color red
plt.show()

#find the means and sd of the sampled X1 distribution
x1samp_u = sum(x1)/len(x1) #mean of X1 samples
x1samp_var = np.var(x1) #variance of our sample

#creating X2
#Summing two gaussians = (u1+u2, var1+var2)
x2_u = 4
x2_var = 4
x2_sd = x2_var**0.5
x2 = rng2.normal(x2_u, x2_sd, 100) + .5*x1 #just add half of each x to the distribution
#print(x2[0:3]) #Check to see if numbers are being sampled the same each time.
#plotting x2
count, bins, ignored = plt.hist(x2, 30, normed=True)
plt.plot(bins, 1/(x2_sd * np.sqrt(2 * np.pi)) *
                np.exp( - (bins - x2_u)**2 / (2 * x2_sd**2) ),
          linewidth=2, color='r')
plt.show()

""" (a) Computing Means of the sampled data """
x1samp_u = sum(x1)/len(x1) #mean of X1 samples
x2samp_u = sum(x2)/len(x2) #mean of X2 samples

""" (b) Covariance Matrix of X1 and X2 """
X = np.vstack((x1, x2)) #stack x1 and x2 into a 2x100 matrix
covX = np.cov(X) #find a covariance. Make sure the dimensions = number of variables being assessed

"""(c) eigenvalues and eigen vectors """
evalX, evecX = np.linalg.eig(covX) #gives eigen values and eigen vectors of our covariance
#first col of evecX corresponds to X1
#first element of eval corresponds to X1

""" (d) Plotting values and their eigenvectors and eigen values"""
ax = plt.subplot(111)
plt.plot(x1, x2, "o", zorder=1) #plot X1 and X2 data using circles
e1 = ax.arrow(3, 4, evecX[0,0]*evalX[0], evecX[1,0]*evalX[0], width=0.01, lw=2, color='r', zorder=2)  # X1 eigenvector multipled by the length of the eigen value to create an arrow of corresponding length
e2 = ax.arrow(3, 4, evecX[0,1]*evalX[1], evecX[1,1]*evalX[1], width=0.01, lw=2, color='g', zorder=3)  # X1 eigenvector
legend = ax.legend(bbox_to_anchor=(1.1, 1.05), loc='2')
plt.title("Relationship between Two Gaussian Distributions", y=1.02) #Title
#plt.legend([e1, e2], ['Eigen vector 1', 'Eigen vector 2'])
plt.xlabel('X1')
plt.xlim(-15, 15) #set the limits
plt.ylabel('X2')
plt.ylim(-15, 15)
plt.savefig("1d.png")
plt.show()


""" (e) Rotate the sample points based on our eigenvectors """
#center the data first
centerX = np.vstack((X[0]-3, X[1]-4)) #center the data by subtracting mean of X1 from X1  data and mean of X2 from X2 data
#rotate based on equation given in 1e
rotX = np.matrix(evecX.T) * np.matrix(centerX ) #this was given in the homework problem
plt.plot(rotX[0], rotX[1], "ro")
plt.title("Rotated and Shifted X1 and X2", y=1.02) #Title
plt.xlabel('X1')
plt.xlim(-15, 15) #set the limits
plt.ylabel('X2')
plt.ylim(-15, 15)
plt.savefig("1e.png")
plt.show()



"""
#######################
## --- Problem 3 --- ##
#######################
"""

#set your mesh grid coordinates so you can graph with the right axes
delta = 0.025
x = np.arange(-6.0, 6.0, delta)
y = np.arange(-6.0, 6.0, delta)
X, Y = np.meshgrid(x, y)

def l2norm(vector):
	total = 0 
	for elem in vector: 
		total += elem**2
	return np.sqrt(np.sqrt(total))
	
"""---3a---"""
u_3a = np.array([1, 1])#these are the means
cov_3a = np.array([[2,0], [0,1]]) #these is the covariance matrix
f_3a = plt.mlab.bivariate_normal(X,Y, l2norm(cov_3a[0]), l2norm(cov_3a[1]), u_3a[0], u_3a[1]) #this is our function 
contour = plt.contour(X,Y,f_3a) #graph using the X and Y we set earlier for the grid axes and the function we created
plt.clabel(contour, inline =1, fontsize = 10) #labels each isocontour
plt.show()
	
"""---3b---"""
u_3b = np.array([-1, 2]) 
cov_3b = np.array([[3,1], [1,2]]) 
f_3b = plt.mlab.bivariate_normal(X,Y, l2norm(cov_3b[0]), l2norm(cov_3b[1]), u_3b[0], u_3b[1]) 
contour = plt.contour(X,Y,f_3b) 
plt.clabel(contour, inline =1, fontsize = 10)
plt.show()
	
"""---3c---"""
u1_3c = np.array([0, 2])
cov1_3c = np.array([[1,1], [1,2]])
f1_3c = plt.mlab.bivariate_normal(X,Y, l2norm(cov1_3c[0]), l2norm(cov1_3c[1]), u1_3c[0], u1_3c[1])
u2_3c = np.array([2, 0])
cov2_3c = np.array([[1,1], [1,2]])
f2_3c = plt.mlab.bivariate_normal(X,Y, l2norm(cov2_3c[0]), l2norm(cov2_3c[1]), u2_3c[0], u2_3c[1])
f_3c = f1_3c-f2_3c #subtracts the two distributions before graphing
contour = plt.contour(X,Y,f_3c)
plt.clabel(contour, inline =1, fontsize = 10)
plt.show()
	
"""---3d---"""
u1_3d = np.array([0, 2])
cov1_3d = np.array([[1,1], [1,2]])
f1_3d = plt.mlab.bivariate_normal(X,Y, l2norm(cov1_3d[0]), l2norm(cov1_3d[1]), u1_3d[0], u1_3d[1])
u2_3d = np.array([2, 0])
cov2_3d = np.array([[3,1], [1,2]])
f2_3d = plt.mlab.bivariate_normal(X,Y, l2norm(cov2_3d[0]), l2norm(cov2_3d[1]), u2_3d[0], u2_3d[1])
f_3d = f1_3d-f2_3d
contour = plt.contour(X,Y,f_3d)
plt.clabel(contour, inline =1, fontsize = 10)
plt.show()
	
"""---3e---"""
u1_3e = np.array([1, 1])
cov1_3e = np.array([[1,0], [0,2]])
f1_3e = plt.mlab.bivariate_normal(X,Y, l2norm(cov1_3e[0]), l2norm(cov1_3e[1]), u1_3e[0], u1_3e[1])
u2_3e = np.array([-1, -1])
cov2_3e = np.array([[2,1], [1,2]])
f2_3e = plt.mlab.bivariate_normal(X,Y, l2norm(cov2_3e[0]), l2norm(cov2_3e[1]), u2_3e[0], u2_3e[1])
f_3e = f1_3e-f2_3e
contour = plt.contour(X,Y,f_3e)
plt.clabel(contour, inline =1, fontsize = 10)
plt.show()






"""
#######################
## --- Problem 4 --- ##
#######################
"""
#
""" Organizing the data """
os.chdir("./data./digit-dataset") #change directory

#load the matlab files with the samples for training
digits = loadmat("./train.mat") #training data with labels
digits_test = loadmat("./test.mat") #test data to submit to kaggle, no labels

#Formatting data for sklearn so it can be put into function. Training must be in 2D array, each row one sample. Labels must be a single column
labels = list(digits["train_label"]) #pull out labels from dictionary
labels = np.array(labels)   #change to array to be used

#pull out images from dicationary, reshape to row form and 2D matrix
numbers = list(digits["train_image"])  #pull from dictionary
numbers = np.array(numbers)     #create an array with the data
numbers = numbers.reshape(1,784,60000) #change into rows instead of 28x28 data
numbers = numbers.swapaxes(1,2) #swap two axes
numbers = numbers.reshape(60000,784) #make it just 2D


#normalizat by l2 
numbers_l2 = np.linalg.norm(numbers, axis=1)
numbers_norm = numbers/numbers_l2[:,None] #divide each element in our numbers array by the l2 corresponding to the image

#concatenate the norm images and labels together
labels = np.array(labels)
numbers_norm = np.array(numbers_norm)
training_digits = np.concatenate((numbers_norm, labels), axis = 1)


#testing digits
test_digits = list(digits_test["test_image"])  #pull from dictionary
test_digits = np.array(test_digits)     #create an array with the data
test_digits = test_digits.reshape(1,784,5000) #change into rows instead of 28x28 data
test_digits = test_digits.swapaxes(1,2) #swap two axes
test_digits = test_digits.reshape(5000,784) #make it just 2D
#normalizat by l2 
test_l2 = np.linalg.norm(test_digits, axis=1)
test_digits_norm = test_digits/test_l2[:,None]

test_labels = list(digits_test["test_label"]) #pull out labels from dictionary
test_labels = np.array(test_labels)   #change to array to be used


""" 4a """

#subset out each classes
num_classes = []
for i in range (0,10):
    num_classes.append(training_digits[training_digits[:,-1]==i]) #checks for labels in the last row to equal designated class

#get the means
num_means = []
for i in range (0,10):
    num_means.append(np.mean(num_classes[i][:,:-1],axis=0))
#
##
#Get the covariances
num_cov = []
for i in range (0,10):
    num_cov.append(np.cov((num_classes[i][:,:-1]).T)) #do not access via variable explorer or spyder will crash

#Covariance is currently positive semidefinite. We need it to be definite. To get around this, add a tiny constant to the diagonals
I = np.identity(784)
num_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
for i in range(0, 10):
    num_cov_pd.append(num_cov[i]+(I*0.00001)) #this 0.00001 value is our alpha which we can then modify to improve accuracy

#create probability density functions based on each u and cov we have. Stores it, but will not show up in variable explorer or return anything.
pdfs = []
for i in range(0,10):
    pdfs.append(multivariate_normal(num_means[i], num_cov_pd[i]))




"""4b Calculating priors for all classes"""
#
prior = []
for i in range (0,10):
    prior.append(float(len(num_classes[i]))/60000.00) #priors are just the frequency of the class divided by all samples being used.
    

"""4c visualizing a covariance matrix"""
digits_notnorm = np.concatenate((numbers, labels), axis = 1) #use non normalized data to visualize. Raw pixel values. 
class_0 = (digits_notnorm[digits_notnorm[:,-1]==0]) #I picked class 0 to visualize
cov_0 = np.cov(class_0[:,:-1].T) #created a new covariance matrix. Should be 784x784

fig, ax = plt.subplots()
im = ax.imshow(cov_0, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=0, vmax=100) #creates a heat map of values
fig.colorbar(im) #the legend
plt.title("Covariance matrix for class 0", y=1.02)
plt.xlabel('Features 0-784')
plt.ylabel('Features 0-784')
plt.show()



"""4d i """
#find the accuracy of digit classification for different training groups using an overall covariance.

tr_grp_size = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]).reshape(1,9) #the training sizes

#gauss_cov_overall - Takes in 4 arguments. Uses an overall covariance of all the classes to classify samples. 
#   tr_grp_size: and array containing the values for training group sizes to be used.
#   samples: data to be trained. Should be a mxn array with labels as the final column
#   te_digits: testing digits. These are used as validation. Should be a mxn-1 array with no labels
#   te_labels: testing labels which correspond to the testing digits
def gauss_cov_overall (tr_grp_size, samples, te_digits, te_labels):
    accuracy = []   #a blank array to hold values for accuracy for each training group
    for size in np.nditer(tr_grp_size): #iterate over the group sizes
        np.random.shuffle(samples)  #for each group size, shuffle training set before partitioning
        tr_classes = []     #empty array to sort classes into. Makes a list of # classes in data
        for digit in range(0, 10): #in this case, i had 10 digits. Set range accordingly
            tr_classes.append(samples[samples[:size,-1]==digit]) #reads the last column for the digit class, sorts in numerical order from 0 to 9 digits
#        print("tr_classes = ", np.shape(tr_classes))   #check the size. Should be a mxn list where m is the number of classes and n is the number of features
#        
        tr_means = []   #to store means for features (this is average of columns). Will results in a list of mxn with m = # classes and n = # of features
        tr_cov = [] #to store covariances. Should be a mxn matrix where m = # of classes. Each class should be a nxn where n = number of features. In digits it's 784x784
        tr_priors = [] #stores priors, results in a mx1 array where m is just the prior for each class. A scalar value.
        for digit in range(0, 10): #iterate through each class and find the means, covariances, and priors for each
            tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0)) 
            tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
            tr_priors.append((len(tr_classes[digit]))/tr_grp_size)
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0])) #in case you want to check that the final sizes are correct.
#            
        cov_overall = tr_cov[0] #find the overall covariance by averaging across all covariances for each class. Start with first covariance
        for digit in range(1, len(tr_classes)): #iterate and add each covariance to total of cov_overall until all cov are summed
            cov_overall += tr_cov[digit] #will results in a single nxn matrix whre n = #of features
        cov_overall = cov_overall/len(tr_classes) #divide it by the number of classes to get the average
        I = np.identity(784) #the cov_overall is positive semidefinite. It must be definite. Must add a small value. To do so create identity matrix of same size as cov_overall
        cov_overall += (I*0.00001) #multiple I by a small alpha value that we will later optimize. Add alphaI to cov_overall to make it positive definitie
#        print("cov_overall shape = ", np.shape(cov_overall)) #check to make sure this came out okay too
#        
        tr_pdfs = [] #create your pdfs the same way in part a
        for digit in range(0, len(tr_classes)):
            tr_pdfs.append(multivariate_normal(tr_means[digit], cov_overall)) #use cov_overall for pdfs because that's waht the problem says
        
        #this part will now find errors due to misclassification
        errors = 0  #set errors to 0 to begin
        flawless = len(te_labels) #this just holds the total # of testing images for dividing purposes to find accuracy.
        for i in range(0, len(te_digits)): #this will iterate over each row (an image to be classified)
            correct = te_labels[i] #our correct label is given by the testing labels
            max = float("-inf") #this will hold the highest probality of correct match we find for a specific image
            max_class = None #this holds the value of the digit that our image had the highest probality of being
            for digit in range (0, 10): #now we will iterate through the pdfs of each class to compare our testing image to
                prob = tr_pdfs[digit].logpdf(te_digits[i]) #probability of a testing image given a PDF of a certain class (eg what's the probability of an image being 1 given our PDF for 1s that we created based on our known training samples for images of 1?)
                if prob > max: #if your probability is higher than the one that was previously stored, update your finding. For example, all images will be initially classified as 0 becuase max prob is currently at -inf. however, if the true test image is not 0, the PDF for another image will predict it better and max will be updated accordingly.
                    max = prob #update your probability accordingly
                    max_class = digit  #also update your digit
            if correct != max_class: #loop ends, now check if the guessed class was actually the true class
                errors += 1 #if not, then add it to the errors
                #now error holds # of misclassified over all testing digits while max and max_class will reset for each image.
        print("errors for size", size, " = ", errors) #gives an idea how classification is going for each training size
        print("size of flawless = ", flawless)
        accuracy.append(1-(errors/flawless))        #calculates the accuracy of our classification for each training group size
    return accuracy #spits out accuracy after function ends

#run the functions with your data!
test_errors = []       #store the results of gauss_cov_overall in a new matrix
for i in range(1):     #how many times do you want to run function to save the results
  test_errors.append(gauss_cov_overall (tr_grp_size, training_digits, test_digits_norm, test_labels)) #runs function
test_errors = np.array(test_errors) #create an array of our accuracies instead of a list

accuracy_results_4di = np.vstack((tr_grp_size, test_errors)) #matches training group sizes to our accuracies


#plot the data!
plt.plot(accuracy_results_4di[0], accuracy_results_4di[1], "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
xlabels=[100, 200, 500, 1000,2000,5000,10000, 30000, 60000] #labeling the axes
plt.title("Digit Classification Accuracy for Various Training Group Sizes using Overall Covariance", y=1.02)#configure X axes
plt.xlabel('Training Group Sizes')
plt.xlim(0, 61000)
plt.xticks([100, 200, 500, 1000,2000,5000,10000, 30000, 60000]) #specificies x tick marks
plt.xticks(rotation=320, size='small') #rotates x axis labels
#configure  Y axes
plt.ylabel('Accuracy for Test Group Size 10000')
plt.ylim(.5,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('Q1training2.png')



"""4d ii """
tr_grp_size = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]).reshape(1,9)

#gauss_covi - Takes in 4 arguments. Uses an each classes' covariance to classify samples. Should be more accurate than overall covariance from 4di
#   tr_grp_size: and array containing the values for training group sizes to be used.
#   samples: data to be trained. Should be a mxn array with labels as the final column
#   te_digits: testing digits. These are used as validation. Should be a mxn-1 array with no labels
#   te_labels: testing labels which correspond to the testing digits
def gauss_covi (tr_grp_size, samples, te_digits, te_labels):
    accuracy = []
    for size in np.nditer(tr_grp_size):
        np.random.shuffle(samples)
        tr_classes = []
        for digit in range(0, 10):
            tr_classes.append(samples[samples[:size,-1]==digit])
#        print("tr_classes = ", np.shape(tr_classes))
#        
        tr_means = []
        tr_cov = []
        tr_priors = []
        for digit in range(0, 10):
            tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0))
            tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
            tr_priors.append((len(tr_classes[digit]))/tr_grp_size)
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0]))
#                        
        I = np.identity(784)
        tr_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
        for i in range(0, 10):
            tr_cov_pd.append(tr_cov[i]+(I*0.00001))
            
#        
        tr_pdfs = []
        for digit in range(0, len(tr_classes)):
            tr_pdfs.append(multivariate_normal(tr_means[digit], tr_cov_pd[digit]))
        
        errors = 0
        flawless = len(te_labels)
        for i in range(0, len(te_digits)):
            correct = te_labels[i]
            max = float("-inf") #not sure about this line
            max_class = None
            for digit in range (0, 10):
                prob = tr_pdfs[digit].logpdf(te_digits[i])
                if prob > max:
                    max = prob #not sure what this line is
                    max_class = digit
            if correct != max_class:
                errors += 1
        print("errors for size", size, " = ", errors)
        print("size of flawless = ", flawless)
        accuracy.append(1-(errors/flawless))        
    return accuracy

#run the functions with your data!
accuracy = []       #store the results of gauss_covi in a new matrix
for i in range(1):     #how many times do you want to run function to save the results
  accuracy.append(gauss_covi(tr_grp_size, training_digits, test_digits_norm, test_labels)) #runs function
accuracy = np.array(accuracy) #create an array

accuracy_results_4dii = np.vstack((tr_grp_size, accuracy))


#plot the data!
plt.plot(accuracy_results_4dii[0], accuracy_results_4dii[1], "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
xlabels=[100, 200, 500, 1000,2000,5000,10000, 30000, 60000] #labeling the axes
plt.title("Classification Accuracy for Various Digit Training Group Sizes with individual covariances", y=1.02)#configure X axes
plt.xlabel('Training Group Sizes')
plt.xlim(0, 61000)
plt.xticks([100, 200, 500, 1000,2000,5000,10000, 30000, 60000]) #specificies x tick marks
plt.xticks(rotation=320, size='small') #rotates x axis labels
#configure  Y axes
plt.ylabel('Accuracy for Test Group Size 10000')
plt.ylim(.5,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('Q1training2.png')



"""4d iv """
#Now we should have found that the individual covariances per class give highest accuracies. So let's optimize our alpha that we mutipled by our I matrix to get even better results! Do this with k-fold cross validation. 
np.random.shuffle(training_digits)
digitsubset = training_digits[:10000]

#Parameters of alpha to test. Classily called c_paramters out of laziness to rename.
C_parameters = ([(10**(-6)), (10**(-5)), (10**(-4)), (10**(-3)), (10**(-2)), (10**(-1)), 1, (10**(1))])
C_parameters2 = ([0.2, 0.4, 0.6, .8, 1.2, 1.4]) 

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
        for i in range(0,k):      #this loop only runs for how many partitions (k) are created. Validation group is the one being tested after training
                
                indexFirst= row_block * i     #this marks the element number which will be the beginning of a validation group. (e.g for validation group = 0, row_block = 60000*0 = 0, so our first element begins at 0)
                indexSecond = row_block * (i+1) #this marks the end of our block. so this will be equivalent to 6000*1 = 60000 for our first iteration in the loop
                
                te_data = samples[indexFirst:indexSecond,:-1] #taking our index1 and index2, we now can tell the array that test group one is the values between 0 and 6000
                te_labels = samples[indexFirst:indexSecond,-1] #this is the same for the labels
                tr_data = np.vstack(( samples[:indexFirst,:] , samples[indexSecond:,:] )) #thus our training data is the concatenation of all the data prior to the validationdata and after the validationdata. (eg with first iteration, nothing comes before 0, but there is all the data after 6000. For the second iteration it will take 0-6000 and concatenate it with 12000-600000)

                results[Cposition, i+1] = k_fold_gauss_predict(tr_data, te_data, te_labels, c_value) #this takes all the data we just assigned and then puts it into the k_fold_linsvc function. Our results, the accuracy for that validationgroup, given by our trained model will then be stored in the row of the respective C value being used, and in the column respective to the validation set just tested. (e.g. we ran a test on the 3rd c value in the matrix, and we are only validationset 5, thus the accuracy of our test results will be stored in the resultsarray[2,5])
    print (results) #and at the very end of all our c values and validation sets, we should print out array to make sure the entire thing was filled in.      
    return results
#
#
def k_fold_gauss_predict(tr_data, te_digits, te_labels, c_value):
    tr_classes = []
    for digit in range(0, 10):
        tr_classes.append(tr_data[tr_data[:,-1]==digit])
#        print("tr_classes = ", np.shape(tr_classes))
#        
    tr_means = []
    tr_cov = []
    tr_priors = []
    for digit in range(0, 10):
        tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0))
        tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
        tr_priors.append((len(tr_classes[digit]))/len(tr_data))
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0]))
#                        
    I = np.identity(784)
    tr_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
    for i in range(0, 10):
        tr_cov_pd.append(tr_cov[i]+(I*c_value))
        
    tr_pdfs = []
    for digit in range(0, len(tr_classes)):
        tr_pdfs.append(multivariate_normal(tr_means[digit], tr_cov_pd[digit]))            
            
            #unncessary section
#    cov_overall = tr_cov[0]
#    for digit in range(1, len(tr_classes)):
#        cov_overall += tr_cov[digit]
#    cov_overall = cov_overall/len(tr_classes)
#    I = np.identity(784)
#    cov_overall += (I*c_value)
##        print("cov_overall shape = ", np.shape(cov_overall))
#        
#    tr_pdfs = []
#    for digit in range(0, len(tr_classes)):
#        tr_pdfs.append(multivariate_normal(tr_means[digit], cov_overall))
#                     
        
    errors = 0
    flawless = len(te_labels)
    for i in range(0, len(te_digits)):
        correct = te_labels[i]
        max = float("-inf") #not sure about this line
        max_class = None
        for digit in range (0, 10):
            prob = tr_pdfs[digit].logpdf(te_digits[i])
            if prob > max:
                max = prob #not sure what this line is
                max_class = digit
        if correct != max_class:
                errors += 1
    print("errors for group = ", errors)
    print("size of flawless = ", flawless)
    accuracy = (1-(errors/flawless))        
    return accuracy

#Run the k cross validation set and store the results!
k_fold_results = []    #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
    k_fold_results.append(k_fold(digitsubset, 10, C_parameters)) #runs function
k_results = k_fold_results[0] #the previous script saves the array as a list, so this picks out the array

Cmeans = k_results[:,1:].mean(axis=1).reshape(len(k_results),1)     #find the means of the rows, store as array

accuracy_matrix_c = np.hstack((np.asarray(C_parameters).reshape(len(C_parameters),1), Cmeans)).T #stacks hyperparameters with resulting accuracies


##plot the data!
plt.plot(accuracy_matrix_c[0], accuracy_matrix_c[-1], "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
plt.xscale("log")
plt.title("Accuracy of Digit Classification for Different Alpha values", y=1.02)#configure X axes
plt.xlabel('Alpha')
plt.xlim(10e-7, 10e2)
#configure  Y axes
plt.ylabel('Accuracy')
plt.ylim(.6,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('C_parametersdigits.png')

"""best appears to be around 0.6"""


"""
############################
## --- Digits Kaggle  --- ##
############################
"""
#submitting for kaggle

kaggle_digits= loadmat("./kaggle.mat")
kaggle_images = list(kaggle_digits["kaggle_image"])
kaggle_images = np.array(kaggle_images)
kaggle_images = kaggle_images.reshape(1,784,5000)
kaggle_images = kaggle_images.swapaxes(1,2)
kaggle_images = kaggle_images.reshape(5000,784)
kaggle_l2 = np.linalg.norm(kaggle_images, axis=1)
kaggle_images = kaggle_images/kaggle_l2[:,None]

np.random.shuffle(training_digits)


def gauss_kaggle(tr_data, te_digits, c_value):
    tr_classes = []
    for digit in range(0, 10):
        tr_classes.append(tr_data[tr_data[:,-1]==digit])
#        print("tr_classes = ", np.shape(tr_classes))
#        
    tr_means = []
    tr_cov = []
    tr_priors = []
    for digit in range(0, 10):
        tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0))
        tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
        tr_priors.append((len(tr_classes[digit]))/len(tr_data))
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0]))
#                        
    I = np.identity(784)
    tr_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
    for i in range(0, 10):
        tr_cov_pd.append(tr_cov[i]+(I*c_value))
        
    tr_pdfs = []
    for digit in range(0, len(tr_classes)):
        tr_pdfs.append(multivariate_normal(tr_means[digit], tr_cov_pd[digit]))            
             
             #this section now no longer has an error check. It simply guesses a class for each image and then appends that guess into an empty array
    kaggle_predictions = []
    for i in range(0, len(te_digits)):
        max = float("-inf") #not sure about this line
        max_class = None
        for digit in range (0, 10):
            prob = tr_pdfs[digit].logpdf(te_digits[i])
            if prob > max:
                max = prob #not sure what this line is
                max_class = digit
        kaggle_predictions.append(max_class)       
    return kaggle_predictions



#Run the function with selectd C paramater value
kaggle_predictions = []  #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
   kaggle_predictions.append( gauss_kaggle(training_digits, kaggle_images, 0.6) ) #runs function
kaggle_predictions = kaggle_predictions[0]

#create a dataframe that has one column for the IDs and another for the predicted values. 
predictions = pd.DataFrame({"A": range(1,len(kaggle_predictions)+1), "B": kaggle_predictions})
predictions.columns = ['Id', 'Category'] #rename columns
predictions.to_csv("JinnJ_kaggledigitsH3.csv", sep=',', index=False, line_terminator = "\r") #save data as csv in proper format





"""--- 4dv ---"""
#Training and testing E-mail!

os.chdir("./data./spam-dataset") #change to proper directory
spam_data = loadmat("./spam_data.mat") #training data with labels

#pull out relevant data
test_email = np.array(list(spam_data["test_data"])) #pull out test email, create np array
email = np.array(list(spam_data["training_data"])) #pull out email, create np array
email_labels = np.array(list(spam_data["training_labels"])).reshape(5172,1) #pull out labels for the e-mails 1=spam, 0=ham; create nparray, make column

#concatenating labels and images together into training set along 1 axis to create a row of 785 elements, last column being the label
training_email = np.concatenate((email, email_labels), axis=1)

#shuffle the training data so it can be used later for the SVM
np.random.shuffle(training_email)


#

"""testing for best alpha"""
#same k-cross method to test for best alphas
C_parameters = ([(10**(-6)), (10**(-5)), (10**(-4)), (10**(-3)), (10**(-2)), (10**(-1)), 1, (10**(1))])
C_parameters2 = ([0.002, 0.004, 0.006, 0.008, 0.02, 0.04, 0.06])

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
        for i in range(0,k):      #this loop only runs for how many partitions (k) are created. Validation group is the one being tested after training
                
                indexFirst= row_block * i     #this marks the element number which will be the beginning of a validation group. (e.g for validation group = 0, row_block = 60000*0 = 0, so our first element begins at 0)
                indexSecond = row_block * (i+1) #this marks the end of our block. so this will be equivalent to 6000*1 = 60000 for our first iteration in the loop
                
                te_data = samples[indexFirst:indexSecond,:-1] #taking our index1 and index2, we now can tell the array that test group one is the values between 0 and 6000
                te_labels = samples[indexFirst:indexSecond,-1] #this is the same for the labels
                tr_data = np.vstack(( samples[:indexFirst,:] , samples[indexSecond:,:] )) #thus our training data is the concatenation of all the data prior to the validationdata and after the validationdata. (eg with first iteration, nothing comes before 0, but there is all the data after 6000. For the second iteration it will take 0-6000 and concatenate it with 12000-600000)

                results[Cposition, i+1] = k_fold_gauss_predict(tr_data, te_data, te_labels, c_value) #this takes all the data we just assigned and then puts it into the k_fold_linsvc function. Our results, the accuracy for that validationgroup, given by our trained model will then be stored in the row of the respective C value being used, and in the column respective to the validation set just tested. (e.g. we ran a test on the 3rd c value in the matrix, and we are only validationset 5, thus the accuracy of our test results will be stored in the resultsarray[2,5])
    print (results) #and at the very end of all our c values and validation sets, we should print out array to make sure the entire thing was filled in.      
    return results



#Note that in this function I have changed range to only 0-2 becuase e-mail is only classified as spam or ham and the number of features is now 32 and not 784!
def k_fold_gauss_predict(tr_data, te_digits, te_labels, c_value):
    tr_classes = []
    for digit in range(0, 2):
        tr_classes.append(tr_data[tr_data[:,-1]==digit])
#        print("tr_classes = ", np.shape(tr_classes))
#        
    tr_means = []
    tr_cov = []
    tr_priors = []
    for digit in range(0, 2):
        tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0))
        tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
        tr_priors.append((len(tr_classes[digit]))/len(te_labels))
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0]))
#                        
    I = np.identity(32)
    tr_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
    for i in range(0, 2):
        tr_cov_pd.append(tr_cov[i]+(I*c_value))
        
    tr_pdfs = []
    for digit in range(0, len(tr_classes)):
        tr_pdfs.append(multivariate_normal(tr_means[digit], tr_cov_pd[digit]))            
            
#    cov_overall = tr_cov[0]
#    for digit in range(1, len(tr_classes)):
#        cov_overall += tr_cov[digit]
#    cov_overall = cov_overall/len(tr_classes)
#    I = np.identity(784)
#    cov_overall += (I*c_value)
##        print("cov_overall shape = ", np.shape(cov_overall))
#        
#    tr_pdfs = []
#    for digit in range(0, len(tr_classes)):
#        tr_pdfs.append(multivariate_normal(tr_means[digit], cov_overall))
#                     
        
    errors = 0
    flawless = len(te_labels)
    for i in range(0, len(te_digits)):
        correct = te_labels[i]
        max = float("-inf") #not sure about this line
        max_class = None
        for digit in range (0, 2):
            prob = tr_pdfs[digit].logpdf(te_digits[i])
            if prob > max:
                max = prob #not sure what this line is
                max_class = digit
        if correct != max_class:
                errors += 1
    print("errors for group = ", errors)
    print("size of flawless = ", flawless)
    accuracy = (1-(errors/flawless))        
    return accuracy

#Run the k cross validation set and store the results!
k_fold_results = []    #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
    k_fold_results.append(k_fold(training_email, 12, C_parameters2)) #runs function
k_results = k_fold_results[0] #the previous script saves the array as a list, so this picks out the array

Cmeans = k_results[:,1:].mean(axis=1).reshape(len(k_results),1)     #find the means of the rows, store as array

accuracy_matrix_c2 = np.hstack((np.asarray(C_parameters2).reshape(len(C_parameters2),1), Cmeans)).T


##plot the data!
plt.plot(accuracy_matrix_c2[0], accuracy_matrix_c2[-1], "ro") #takes the two arrays and plots x,y with the characters being red circles "ro"
plt.xscale("log")
plt.title("Accuracy of E-mail Classification for Different Alpha values", y=1.02)
plt.xlabel('Alpha')
plt.xlim(10e-7, 10e2)
#configure  Y axes
plt.ylabel('Accuracy')
plt.ylim(.6,1.0)
plt.subplots_adjust(bottom=0.15)
plt.savefig('C_parametersdigits.png')

""" best appears to be 0.008 """



"""kaggle submission for e-mail""" 
np.random.shuffle(training_email)
te_sub = training_email[:431,:-1]
te_sub_labs = training_email[:431,-1]
tr_sub = training_email[431:,:]

def gauss_kaggle(tr_data, te_digits, c_value):
    tr_classes = []
    for digit in range(0, 2):
        tr_classes.append(tr_data[tr_data[:,-1]==digit])
#        print("tr_classes = ", np.shape(tr_classes))
#        
    tr_means = []
    tr_cov = []
    tr_priors = []
    for digit in range(0, 2):
        tr_means.append(np.mean(tr_classes[digit][:,:-1],axis=0))
        tr_cov.append(np.cov(tr_classes[digit][:,:-1].T))
        tr_priors.append((len(tr_classes[digit]))/len(tr_data))
#        print(  "tr_means len 0 = ", len(tr_means[0]), 
#                "\n tr_cov = ", np.shape(tr_cov[0]),
#                "\n tr_priors= ", np.shape(tr_priors[0]))
#                        
    I = np.identity(32)
    tr_cov_pd = [] #num_cov_pd = positive definite matrix by adding a small value to the diagonals
    for i in range(0, 2):
        tr_cov_pd.append(tr_cov[i]+(I*c_value))
        
    tr_pdfs = []
    for digit in range(0, 2):
        tr_pdfs.append(multivariate_normal(tr_means[digit], tr_cov_pd[digit]))            
             
    kaggle_predictions = []
    for i in range(0, len(te_digits)):
        max = float("-inf") #not sure about this line
        max_class = None
        for digit in range (0, 2):
            prob = tr_pdfs[digit].logpdf(te_digits[i])
            if prob > max:
                max = prob #not sure what this line is
                max_class = digit
        kaggle_predictions.append(max_class)       
    return kaggle_predictions

#Run the function with selectd C paramater value
kaggle_predictions = []  #stores all the results of the k_fold function in a list
for i in range(1):     #how many times do you want to run function to save the results
   kaggle_predictions.append( gauss_kaggle(training_email, test_email, 0.008) ) #runs function
kaggle_predictions = kaggle_predictions[0]

#create a dataframe that has one column for the IDs and another for the predicted values. 
predictions = pd.DataFrame({"A": range(1,len(kaggle_predictions)+1), "B": kaggle_predictions})
predictions.columns = ['Id', 'Category'] #rename columns
predictions.to_csv("JinnJ_kagglespamH3.csv", sep=',', index=False, line_terminator = "\r") #save data as csv in proper format




