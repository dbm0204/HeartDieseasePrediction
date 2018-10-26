import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt # Visuals
import seaborn as sns 
import sklearn as skl
import pandas as pd
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split # Create training and test sets
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn import tree 
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn.model_selection import learning_curve
from sklearn import cross_validation  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as pt 
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image 

plt.style.use('ggplot')
pt.set_credentials_file(username='dbm0204', api_key='7582Q9k8ReXNw4wfqdR8')

# ### Import Data 
# We imported the data directly from the UCI Machine Learning Repository website.
# Because we are working with three different datasets, we decided to merge the data into one array.
# Here is a short output of the data, just the first five rows.
Cleveland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
np.set_printoptions(threshold=np.nan) #see a whole array when we output it
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv(urlopen(Cleveland_data_URL), names = names) #gets Cleveland data
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease] #combines all arrays into a list
heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()

## Exploratory Analysis 
# Let's start with the exploratory analysis of our dataset. 
# We don't want to predict on all the variables from the original data so we are getting rid of 'ca', 'slope', and 'thal'. For the variables we kept, there are still some "?" in the data, so we're going to replace them with a NAN. 
# We can also see the data types of the variables in the data set. This way, we can differentiate between discrete or categorical representations of our variables. Although the entire set is numerical, some outputs of the datatypes are objects. #
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
## Preprocessing
### Normalizing Data
# Everything else seems okay. So we begin the preprocessing of th data.
# All of our data is numerical, so we are going to standardize the variables to approach our analysis more objectively. In doing so, the data is scaled to be only between 0 and 1, to objectify the distribution.  # 
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
for i in range(1,5):
    heartDisease['heartdisease'] = heartDisease['heartdisease'].replace(i,1)

###### Boxplot visualization of the Transformed Dataset and the Distribution of the Attributes 
f, ax = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
plt.title("Box Plot of Transformed Data Set")
ax.set(xlim=(-.05, 1.05))
ax = sns.boxplot(data = heartDisease[1:29], orient = 'h', palette = 'Set2')
plt.show()
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

### Decision Trees
# Decision trees have a hierarchical structure, where each leaf of the tree represents
# a class label while the branches represent represent the process the tree used to deduce the class labels.
dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=None,splitter='random')
dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])

predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
predictright = 0
for i in range(0,predictions_dt.shape[0]-1):
    if (predictions_dt[i]== test.iloc[i][10]):
        predictright +=1
accuracy = predictright/predictions_dt.shape[0]
print("accuracy of Decision Tree "+str(accuracy))
print("Table comparing actual vs. predicted values for our test set:\n",pd.crosstab(predictions_dt, test_class_set['heartdisease'],rownames=['Predicted Values'], colnames=['Actual Values']))
predictions = dt.predict(test_set)
print(pd.crosstab(predictions_dt, test_class_set['heartdisease'], rownames=['Predicted Values'], colnames=['Actual Values']))
accuracy_dt = dt.score(test_set, test_class_set['heartdisease'])
print("Here is our mean accuracy on the test set:\n",'%.3f' % (accuracy_dt * 100), '%')

# Here we calculate the test error rate!
test_error_rate_dt = 1 - accuracy_dt
print("The test error rate for our model is:\n",'%.3f' % (test_error_rate_dt * 100), '%')
#ROC curve calculation
fpr1, tpr1, _ = roc_curve(predictions_dt,test_class_set)

#AUC curve Calculation
auc_dt = auc(fpr1,tpr1)

#Plotting the ROC curve
fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(fpr1, tpr1,label='Decision Trees ROC Curve (area = %.4f)' % auc_dt, color = 'navy', linewidth=2)
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
trace1 = go.Scatter(x=fpr1,y=tpr1,mode='lines',line=dict(color='darkorange', width=2),name='Decision Tree ROC curve (area = %0.4f)' % auc_dt)
layout = go.Layout(title='Receiver operating characteristic example',xaxis=dict(title='False Positive Rate'),yaxis=dict(title='True Positive Rate'))
fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig)
