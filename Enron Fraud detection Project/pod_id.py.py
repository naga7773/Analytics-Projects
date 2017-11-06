
# coding: utf-8

# In[1]:

import sys
from time import time 
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:

#Task 1 -- What features to use?
# But beofore that I wanted to see the basic details of the data set. 
#Load in the data set as a dictionary
# So import the data set
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[3]:

# See how many observations are present in the data set.
print 'There are',len(data_dict.keys()),'observations in the data set.'


# In[4]:

#How many features are given for each observation?
NumberofFeatures=0
for i in data_dict.keys():
    for j in data_dict[i].keys():
        NumberofFeatures +=1
    break
print 'There are',NumberofFeatures,'features for each observation.'


# In[5]:

# Print the names of those 21 features. 
Actualfeatures=[]
for i in data_dict.keys():
    for j in data_dict[i].keys():
        Actualfeatures.append(j)
    break
print Actualfeatures


# In[174]:

#converttocsv(data_dict)


# In[6]:

# Let us see how many people are actually marked as POI in our dataset.
NumberofPOI=0
for i in data_dict.keys():
    if data_dict[i]['poi'] ==1:
        NumberofPOI +=1
print 'Number of POI in the dataset is',NumberofPOI
    
            


# In[7]:

# A visual showed that missing values are encoded as 'NaN'.
# So before continuing my analysis I want to see how many missing values are there in each feature.

MissingValueDict={}
for i in Actualfeatures:
    MissingValueDict[i]=0
for i in data_dict.keys():
    for j in Actualfeatures:
        if data_dict[i][j]=='NaN':
            MissingValueDict[j] +=1

print 'Missing Values Per feature: '
for i in MissingValueDict.keys():
    print i, '---', MissingValueDict[i]


# In[177]:

# This is very useful. There are only 146 records in the dataset and as seen above many of them have missing values. 
# As seen director_fees,deferred_income, deferral_payments, restricted_stock_deferred,loan_advances 
# have more number of missing values.
# I would not consider these features in my feature selection as they will not bring any useful information to my model.


# In[8]:

import matplotlib.pyplot as plt
# Removing Outliers 
# This step was already performed in mini project. But I am still including it for my readers. 
#I want to look at the salry and bonus to identif any outliers in the data
Myfeatures=['bonus','salary']
data = featureFormat(data_dict, Myfeatures, sort_keys = True)
BonusValues, SalaryValues = targetFeatureSplit(data)
get_ipython().magic(u'matplotlib inline')
plt.scatter(BonusValues,SalaryValues)
plt.ylabel('Salary in US$')
plt.xlabel('bonus in us$')
plt.title('Salary vs Bonus')


# In[9]:

# As seen in the above plot there is one point which is very far from the other observations. 
# I checked the data and this is because there is a "Total" row in the file that is causing this outlier. 
# SO I'm removing this observation. 
data_dict.pop('TOTAL',0)


# In[10]:

# remaining observations in the data set 
print 'There are',len(data_dict.keys()),'observations remaining in the data set'


# In[11]:

# removing another outlier which had most of its values NaN 
# and its name doesn't look like a person name

data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)


# In[12]:

print 'Remaining observations in the dataset are: ',len(data_dict.keys())


# In[13]:

# Manual checks revealed that there is one more observation with all values missing.
# removing this oiutlier from the datatset.
data_dict.pop('LOCKHART EUGENE E',0)


# In[14]:

print "There are ",len(data_dict.keys()),"observations remaining in the data."


# In[15]:

# Keeping a copy of data set for final submission.
my_dataset = data_dict


# In[16]:

# Some of the features included in the data set gives us information about the 
# number of emails and messages sent by the person and received by the person.
# Also provided is the number of messages sent and received between this person and the 
# POI. My guess is that there will be more communication between the POI's and that taking
# a ratio of these would be helpful in our POI Identifier. 
# for example to_messages gives us the information of how many messages this 
# particular person received. And from_poi_to_this_person gives us 
# the number of messages this person received from the POI. 
# So The ratio would be from_poi_to_this_person / to_messages 
# This gives the percent of messages this person gets from the POI.

# Updating: My earlier function failed to give correct ratio because there are missing values. 
# SO updating the function.
from operator import truediv
def fraction(numerator,denominator):
    if numerator == 'Nan' or denominator =='NaN':
        return 0.0
    p = truediv(numerator,denominator)
    return p


# In[17]:

# Getting the required fields out from the copy of my data set.
# update: first I created two list one for numerator 
# and one for denominator. But the division did not work.
to_messages =[]
from_poi_to_this_person=[]
from_this_person_to_poi=[]
from_messages=[]
Ratio_sent_to_POI=[]
Ratio_received_from_POI=[]
poi=[]
for i in my_dataset.keys():
    to_messages.append(my_dataset[i]['to_messages'])
    from_poi_to_this_person.append(my_dataset[i]['from_poi_to_this_person'])
    from_this_person_to_poi.append(my_dataset[i]['from_this_person_to_poi'])
    from_messages.append(my_dataset[i]['from_messages'])
    Ratio_sent_to_POI.append(fraction(my_dataset[i]['from_this_person_to_poi'],my_dataset[i]['from_messages']))
    Ratio_received_from_POI.append(fraction(my_dataset[i]['from_poi_to_this_person'],my_dataset[i]['to_messages']))
    poi.append(my_dataset[i]['poi'])
print len(from_messages)
print len(from_this_person_to_poi)
print len(from_poi_to_this_person)
print len(to_messages)
print len(Ratio_received_from_POI)
print len(Ratio_sent_to_POI)
    


# In[18]:

# Now we have the ratios. 
colo=[]
for r in poi:
    if r==0:
        colo.append('c')
    else:
        colo.append('b')
plt.scatter(Ratio_sent_to_POI,Ratio_received_from_POI,c=colo)
#blue for POI's and Cyan for non POI's


# In[53]:

# Now this really looks like a good information 
# and it supports my intuition as well.
# We can see that POI's have higher ratio's. 

# creating a feature list as described in the project description.
financialfeatures= ['salary', 'total_payments', 'bonus',
                      'total_stock_value', 'expenses',
                     'exercised_stock_options', 'other', 'long_term_incentive',
                     'restricted_stock']

emailfeatures= ['to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi','Ratio_sent_to_POI',
                 'Ratio_received_from_POI']
Target= ['poi']

features_list=Target+financialfeatures+emailfeatures
# As  I mentioned previously I excluded those variables that had very large amount of missing values. 


# In[54]:

print features_list


# In[55]:

# we started with 21 features and I added two more to make a toatl of 23.
# But I also ignored 5 features. 
# Thus making it 18 features in total. 

# One thing I forgot to do is adding th enewly created features to the dataset.
# As of now they are individual variables.

for i in my_dataset.keys():
    my_dataset[i]['Ratio_sent_to_POI'] = Ratio_sent_to_POI[my_dataset.keys().index(i)]
    my_dataset[i]['Ratio_received_from_POI'] = Ratio_received_from_POI[my_dataset.keys().index(i)]
    


# In[56]:

# Of all the variable I wanted to choose 
# 10 best variables that can help
# in creating a good classifier.

# I am usingSelectKBest algorithm to find the best features

#method 1 -- ExtraTreesClassiier
from sklearn.ensemble import ExtraTreesClassifier
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

model1 = ExtraTreesClassifier()
model1.fit(features,labels)
print(model1.feature_importances_)


# In[51]:

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(features, labels)
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), features_list), 
             reverse=True)


# In[57]:

print features_list


# In[58]:

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(features,labels)
print rf.feature_importances_


# In[59]:

from sklearn.feature_selection import SelectKBest,f_classif

k_best = SelectKBest(k=10)
k_best.fit(features,labels)
scores= k_best.scores_


# In[60]:

print scores


# In[68]:

#Effect of new features
#step 1 
# Logistic regression using original features
Original_features = ['poi','salary', 'to_messages', 'deferral_payments', 
                     'total_payments', 'exercised_stock_options', 
                     'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                     'restricted_stock_deferred', 'total_stock_value', 
                     'expenses', 'loan_advances', 'from_messages', 'other',
                     'from_this_person_to_poi', 'director_fees', 
                     'deferred_income', 'long_term_incentive',
                     'from_poi_to_this_person']
Added_features = ['poi','salary', 'to_messages', 'deferral_payments', 
                     'total_payments', 'exercised_stock_options', 
                     'bonus', 'restricted_stock', 'shared_receipt_with_poi',
                     'restricted_stock_deferred', 'total_stock_value', 
                     'expenses', 'loan_advances', 'from_messages', 'other',
                     'from_this_person_to_poi', 'director_fees', 
                     'deferred_income', 'long_term_incentive',
                     'from_poi_to_this_person','Ratio_sent_to_POI','Ratio_received_from_POI']
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score
#exercised_stock_options, Ratio_sent_to_POI, expenses, bonus, salary, 
my_feature_list_=Target+['salary','bonus','total_stock_value','expenses',
                        'exercised_stock_options','Ratio_sent_to_POI','shared_receipt_with_poi']
data1 = featureFormat(my_dataset,Original_features, sort_keys = True)
labels, features = targetFeatureSplit(data1)
features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)

LR_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

LR_clf.fit(features_train,labels_train)
pred = LR_clf.predict(features_test)
acc_ScaledLR=accuracy_score(labels_test,pred)
rec_ScaledLR=recall_score(labels_test,pred)
pr_ScaledLR = precision_score(labels_test,pred)
print 'Accuracy :',acc_ScaledLR
print 'Precision :',pr_ScaledLR
print 'Recall :',rec_ScaledLR

data2 = featureFormat(my_dataset,Added_features, sort_keys = True)
labels, features = targetFeatureSplit(data2)
features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)

LR_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

LR_clf.fit(features_train,labels_train)
pred = LR_clf.predict(features_test)
acc_ScaledLR=accuracy_score(labels_test,pred)
rec_ScaledLR=recall_score(labels_test,pred)
pr_ScaledLR = precision_score(labels_test,pred)
print 'Accuracy :',acc_ScaledLR
print 'Precision :',pr_ScaledLR
print 'Recall :',rec_ScaledLR


# In[61]:

# taking the combination of features from all the three methods
# 'salary','bonus','total_stock_value','expenses','exercised_stock_options','Ratio_sent_to_POI','shared_receipt_with_poi'

my_feature_list=Target+['salary','bonus','total_stock_value','expenses',
                        'exercised_stock_options','Ratio_sent_to_POI','shared_receipt_with_poi']
data = featureFormat(my_dataset,my_feature_list)
labels,features = targetFeatureSplit(data)


# In[196]:

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import cross_validation
from sklearn.metrics import accuracy_score,precision_score, recall_score


# In[197]:

# divide the data in to training set and test set.
# I'm considering 30% data (randomly chosen) as test data.

features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)


# In[198]:

from sklearn.svm import SVC
s_clf = SVC(kernel = 'rbf',C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')


# In[199]:

s_clf.fit(features_train,labels_train)
pred = s_clf.predict(features_test)
acc=accuracy_score(labels_test,pred)
rec=recall_score(labels_test,pred)
pr = precision_score(labels_test,pred)


# In[200]:

print 'Accuracy :',acc
print 'Precision :',pr
print 'Recall :',rec


# In[201]:

# As seen eventhough the accuracy is over 88% precision and 
#recall are zero becuase the prediction did not our class of interest POI
# This is why we should be considering precision and recall in 
# data sets like these which are skewed or imbalanced in terms of the
# class frequency.


# In[202]:

# Earlier I did not use feature scaling, but SVC performs 
# better when features are scaled.


# In[203]:

from sklearn.preprocessing import StandardScaler
features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
s_clf_scaled = SVC(kernel = 'rbf',C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')
s_clf_scaled.fit(features_train_scaled,labels_train)
pred = s_clf_scaled.predict(features_test)
acc_ScaledSVC=accuracy_score(labels_test,pred)
rec_ScaledSVC=recall_score(labels_test,pred)
pr_ScaledSVC = precision_score(labels_test,pred)
print 'Accuracy :',acc_ScaledSVC
print 'Precision :',pr_ScaledSVC
print 'Recall :',rec_ScaledSVC


# In[204]:

# Even after scaling the features the performance is still bad.
# SO I considered tuning the parameters so that 
# I can give the algorithm a chance to fit according to the data.
# I am using GridSearchCV for tuning the parameters.


# In[205]:

from sklearn.svm import SVC
parameter_candidates = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)

clf.fit(features_train_scaled, labels_train) 


# In[206]:

print('Best score for data1:', clf.best_score_)
print('Best C:',clf.best_estimator_.C)
print('Best Kernel:',clf.best_estimator_.kernel)
print('Best Gamma:',clf.best_estimator_.gamma)


# In[207]:

# As seen GridSearch CV also outputs the optimum values for the parameters
# obtained after tuning.


# In[208]:

from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)

scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)

k_clf.fit(features_train_scaled,labels_train)
pred = k_clf.predict(features_test)
acc_ScaledKM=accuracy_score(labels_test,pred)
rec_ScaledKM=recall_score(labels_test,pred)
pr_ScaledKM = precision_score(labels_test,pred)
print 'Accuracy :',acc_ScaledKM
print 'Precision :',pr_ScaledKM
print 'Recall :',rec_ScaledKM


# In[209]:

# Even with Kmeans, the performance is bad as Precision is 0.1


# In[75]:

# Using Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score, recall_score
#exercised_stock_options, Ratio_sent_to_POI, expenses, bonus, salary, 
my_feature_list=Target+['salary','bonus','total_stock_value','expenses',
                        'exercised_stock_options','Ratio_sent_to_POI','shared_receipt_with_poi']
data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)

LR_clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(tol = 0.001, C = 10**-8, penalty = 'l2', random_state = 42))])

LR_clf.fit(features_train,labels_train)
pred = LR_clf.predict(features_test)
acc_ScaledLR=accuracy_score(labels_test,pred)
rec_ScaledLR=recall_score(labels_test,pred)
pr_ScaledLR = precision_score(labels_test,pred)
print 'Accuracy :',acc_ScaledLR
print 'Precision :',pr_ScaledLR
print 'Recall :',rec_ScaledLR


# In[76]:

acc_ScaledLR=[]
rec_ScaledLR=[]
pr_ScaledLR=[]
from numpy import mean
for i in range(500):
    features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.3)
    LR_clf.fit(features_train,labels_train)
    pred = LR_clf.predict(features_test)
    acc_ScaledLR.append(accuracy_score(labels_test,pred))
    rec_ScaledLR.append(recall_score(labels_test,pred))
    pr_ScaledLR.append(precision_score(labels_test,pred))


# In[77]:

print mean(acc_ScaledLR)
print mean(rec_ScaledLR)
print mean(pr_ScaledLR)


# In[78]:

clf = LR_clf
features_list=my_feature_list
dump_classifier_and_data(clf, my_dataset, features_list)


# In[79]:

features_list


# In[80]:

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:



