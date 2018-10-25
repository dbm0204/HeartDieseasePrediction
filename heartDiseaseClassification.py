
# coding: utf-8
# # Heart Disease Classification#
# ## Abstract
# Data classification can often be applied to medical data, helping detect the prevalence of disease.
# The following analysis predicts the prevalence of heart disease from a dataset drawn from four different sources: the Cleveland Clinic Foundation,
# the Hungarian Institute of Cardiology, Budapest and the University Hospital, Zurich, Switzerland and is drawn from the UCI Machine Learning Repository.
# This project focuses on the classification of heart disease by using several machine learning algorithms, such as random forests, kth-nearest neighbors, support vector machine and logistic regression.
# The analysis implements Python and Python libraries including these algorithms to come up with a model that best predicts the diagnosis (0 = not present, 1 = present). Through the investigation, we will
# find which algorithm most effectively and consistently predicts the presence of heart disease.
# We will examine 11 out of 76 total attributes, including age, sex, chest pain type, resting blood pressure, cholesterol level, etc.
# 

# ### Import Modules
# We begin by importing python modules.
import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd

from sklearn.cross_validation import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import svm #SVM
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository

import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt
plt.style.use('ggplot')
pt.set_credentials_file(username='bmathew2014', api_key='bckdZ4APoakTageKPaJG')


# ### Import Data
# 
# We imported the data directly from the UCI Machine Learning Repository website.
# Because we are working with three different datasets, we decided to merge the data into one array.
# Here is a short output of the data, just the first five rows.
# In[27]:


Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'


np.set_printoptions(threshold=np.nan) #see a whole array when we output it

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
longbeachHeartDisease = pd.read_csv(urlopen(Longbeach_data_URL),names=names) #gets Long beach data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease] #combines all arrays into a list
heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()


# ## Exploratory Analysis 
# Let's start with the exploratory analysis of our dataset. 
# We don't want to predict on all the variables from the original data so we are getting rid of 'ca', 'slope', and 'thal'. For the variables we kept, there are still some "?" in the data, so we're going to replace them with a NAN. 
# We can also see the data types of the variables in the data set. This way, we can differentiate between discrete or categorical representations of our variables. Although the entire set is numerical, some outputs of the datatypes are objects. 
#

del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']
heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes


#loop to count the "?" per variable. More for interpretation.

count = 0
for item in heartDisease:
    for i in heartDisease[item]:
        count += (i == '?')


# ### Class Imbalance
# This is a function from another project by Ravi.
# This tests for class imbalance in the data, but since we are going to simplify the data later,
# this shouldn't be an issue. But let's check anyway.
# Class Imbalance refers to when a class within a data set is outnumbered by the other class (or classes).
# Class Imbalance is present when a class populates 10-20% of the data set. We can see that is a problem here! :(
def classImbalance(item):
    item_0 = 0
    item_1 = 0
    item_2 = 0
    item_3 = 0
    item_4 = 0
    item_5 = 0
    for item_i in heartDisease[item]:
        for i in range(0,6):
            if (item == i):
                item_i +=1
    heartDisease_i = 0
    for i in  range (0,6):
        heartDisease_i = (item_i/len(heartDisease)) * 100
        print("The percentage of level", i, "in the response variable is: {0:.2f}".format(heartDisease_i)) 
 
classImbalance('heartdisease')

trace0 = go.Box(y=heartDisease['age'],name='age')
trace1 = go.Box(y=heartDisease['sex'],name='sex')
trace2 = go.Box(y=heartDisease['cp'],name='cp')
trace3 = go.Box(y=heartDisease['trestbps'],name='trestbps')
trace4 = go.Box(y=heartDisease['chol'],name='chol')
trace5 = go.Box(y=heartDisease['fbs'],name='fbs')
trace6 = go.Box(y=heartDisease['restecg'],name='restecg')
trace7 = go.Box(y=heartDisease['thalach'],name='thalach')
trace8 = go.Box(y=heartDisease['exang'],name='exang')
trace9 = go.Box(y=heartDisease['oldpeak'],name='oldpeak')
trace10 = go.Box(y=heartDisease['heartdisease'],name='heart disease status')
plotdata = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
py.iplot(plotdata)


# ## Preprocessing
# ### Normalizing Data
# 
# Everything else seems okay. So we begin the preprocessing of th data.
# All of our data is numerical, so we are going to standardize the variables to approach our analysis more objectively. In doing so, the data is scaled to be only between 0 and 1, to objectify the distribution.  
# 

# In[31]:


for item in heartDisease: #converts everything to floats
    heartDisease[item] = pd.to_numeric(heartDisease[item])

def normalize(heartDisease, toNormalize): #normalizes 
    result = heartDisease.copy()
    for item in heartDisease.columns:
        if (item in toNormalize):
            max_value = heartDisease[item].max()
            min_value = heartDisease[item].min()
            result[item] = (heartDisease[item] - min_value) / (max_value - min_value)
    return result
toNormalize = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak'] #columns to normalize
heartDisease = normalize(heartDisease, toNormalize)
heartDisease = heartDisease.dropna()
heartDisease.head()
# This is a classification problem, so to simplify our project we are going to convert the predictor
# column into 1 for "heart disease is present" and 0 for "heart disease is not present."
# Before, the scope of the disease ran from 0 - 5 for the intensity of the heart disease but this shit's too hard
# so we're going to replace it.
# In[32]:


for i in range(1,5):
    heartDisease['heartdisease'] = heartDisease['heartdisease'].replace(i,1)


# ###### Boxplot visualization of the Transformed Dataset and the Distribution of the Attributes 

# In[33]:


f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')
plt.title("Box Plot of Transformed Data Set")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = heartDisease[1:29], orient = 'h', palette = 'Set2')
plt.show()


# In[34]:


trace0 = go.Box(y=heartDisease['age'],name='age')
trace1 = go.Box(y=heartDisease['sex'],name='sex')
trace2 = go.Box(y=heartDisease['cp'],name='cp')
trace3 = go.Box(y=heartDisease['trestbps'],name='trestbps')
trace4 = go.Box(y=heartDisease['chol'],name='chol')
trace5 = go.Box(y=heartDisease['fbs'],name='fbs')
trace6 = go.Box(y=heartDisease['restecg'],name='restecg')
trace7 = go.Box(y=heartDisease['thalach'],name='thalach')
trace8 = go.Box(y=heartDisease['exang'],name='exang')
trace9 = go.Box(y=heartDisease['oldpeak'],name='oldpeak')
trace10 = go.Box(y=heartDisease['heartdisease'],name='heart disease status')
plotdata = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
py.iplot(plotdata)


# ## Model Estimation
# ### Training and Testing

train, test = train_test_split(heartDisease, test_size = 0.20, random_state = 42)
# Create the training test omitting the diagnosis

training_set = train.ix[:, train.columns != 'heartdisease']
# Next we create the class set 
class_set = train.ix[:, train.columns == 'heartdisease']

# Next we create the test set doing the same process as the training set
test_set = test.ix[:, test.columns != 'heartdisease']
test_class_set = test.ix[:, test.columns == 'heartdisease']


# ### Decision Trees
# 
# Decision trees have a hierarchical structure, where each leaf of the tree represents
# a class label while the branches represent represent the process the tree used to deduce the class labels.
#

dt = tree.DecisionTreeClassifier()
dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
predictright = 0
predictions_dt.shape[0]
for i in range(0,predictions_dt.shape[0]-1):
    if (predictions_dt[i]== test.iloc[i][10]):
        predictright +=1
accuracy = predictright/predictions_dt.shape[0]
print("accuracy of Decision Tree "+str(accuracy))
# The accuracy for a decision tree is 95.64%. This is high, but I want to see if we can get higher with a random forest.
# In[37]:


print("Table comparing actual vs. predicted values for our test set:\n",pd.crosstab(predictions_dt, test_class_set['heartdisease'],
                  rownames=['Predicted Values'], 
                  colnames=['Actual Values']))


# In[38]:


#ROC curve calculation 
fpr2, tpr2, _ = roc_curve(predictions_dt, test_class_set)

#AUC curve calcuation
auc_dt = auc(fpr2, tpr2)


# #### Note on ROC and AUC curves
# ###### Receiver Operating Characteristc Curve calculations 
# The function roc_curve is for calculating the False Positive Rates and True Positive Rates for each model. 
# The Area under the Curve was also calculated (in this case the curves are the ROC Curves). These are going to be calculated for each model so we can fit a graph to visualize which model ends up working best for the data. 

# ### Random Forest
# 
# A random forest is an entire forest of random decision trees. This will perform better than just a single tree because it corrects the problem of overfitting. 
# 
# Decision Trees tend to have low bias and high variance, a process known as Bagging Trees (Bootstrap Aggregating). Random Forest aims to reduce this correlation by choosing only a subsample of the feature space at each split. Essentially aiming to make the trees more independent and thus reducing the variance.
# 
# 
# ##### Tree Concepts 
# + Single Decision Tree (Single tree)
# + Bagging Trees (Multiple trees) [Model with all features, M, considered at splits, where M = all features]
# + Random Forest (Multiple trees) [Model with m features considered at splits, where m < M]

# In[39]:


fitRF = RandomForestClassifier(random_state = 42, criterion='gini',n_estimators = 500,max_features = 5)

# In[40]:


fitRF.fit(training_set, class_set['heartdisease'])


# We have to gather the variable importance. This is essential in decision trees and random forests for seeing which attributes played an important role in our algorithm.

# In[41]:


importancesRF = fitRF.feature_importances_
indicesRF = np.argsort(importancesRF)[::-1]
indicesRF


# ##### Gini impurity 
# Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly
# labeled according to the distribution of labels in the subset. Gini impurity can be computed by summing the probability fi of an item with label i being chosen times the probability 1- fi of a
# mistake in categorizing that item.
# It reaches its minimum (zero) when all cases in the node fall into a single target category. Here we compute the importance of each gini impurity and plot it. 

# In[42]:


namesInd = names[:11]
print("Feature ranking:")

for f in range(10):
    i = f
    print("%d. The feature '%s' has a Gini Importance of %f" % (f + 1, 
                                                                namesInd[indicesRF[i]], 
                                                                importancesRF[indicesRF[f]]))


# In[45]:


feature_space = []
for i in range(9, -1, -1):
    feature_space.append(namesInd[indicesRF[i]])


# Print the accuracy of the random forest. 
# 

# In[46]:


indRf = sorted(importancesRF) #sort into decreasing order 
index = np.arange(10) #arrange the importance

f, ax = plt.subplots(figsize=(11, 11))

ax.set_facecolor('#fafafa')
plt.title('Feature importances for Random Forest Model')
plt.barh(index, indRf,align="center", color = '#875FDB')
plt.yticks(index, ('cp', 'exang', 'oldpeak', 'chol', 'thalach', 'age', 'trestbps', 'sex', 'restecg', 'fbs'))
plt.ylim(-1, 10)
plt.xlim(0, 0.15)
plt.xlabel('Gini Importance')
plt.ylabel('Feature')
plt.show()


predictions_RF = fitRF.predict(test_set)
print(pd.crosstab(predictions_RF, test_class_set['heartdisease'], rownames=['Predicted Values'], colnames=['Actual Values']))


accuracy_RF = fitRF.score(test_set, test_class_set['heartdisease'])

print("Here is our mean accuracy on the test set:\n",'%.3f' % (accuracy_RF * 100), '%')



# Here we calculate the test error rate!
test_error_rate_RF = 1 - accuracy_RF
print("The test error rate for our model is:\n",
     '%.3f' % (test_error_rate_RF * 100), '%')


# In[50]:

#R
#ROC curve calculation
fpr1, tpr1, _ = roc_curve(predictions_RF, test_class_set)
#AUC curve calcuation
auc_rf = auc(fpr1, tpr1)


fig, ax = plt.subplots(figsize=(10, 10))

plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve (area = %.4f)' % auc_dt, color = 'navy', linewidth=2)
plt.plot(fpr2, tpr2,label='Random Forest ROC Curve (area = %.4f)' % auc_rf, color = 'red', linestyle=':', linewidth=2)

ax.set_facecolor('#fafafa')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison For All Models')
plt.legend(loc="lower right")
plt.show()
lw = 2

trace1 = go.Scatter(x=fpr1, y=tpr1, mode='lines', line=dict(color='darkorange', width=lw),name='Decision Tree ROC curve (area = %0.4f)' % auc_dt)
trace2 = go.Scatter(x=fpr2, y=tpr2, mode='lines', line=dict(color='red', width=lw),name= 'Random Forest ROC curve (area = %0.4f)' % auc_rf)


layout = go.Layout(title='Receiver operating characteristic example',xaxis=dict(title='False Positive Rate'),yaxis=dict(title='True Positive Rate'))
fig = go.Figure(data=[trace1, trace2], layout=layout)
py.iplot(fig)
