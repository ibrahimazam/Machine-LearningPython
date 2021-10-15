#Import Libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------------------

#load breast cancer data

BreastData = load_breast_cancer()

#X Data
X = BreastData.data
#print('X Data is \n' , X[:10])
#print('X shape is ' , X.shape)
#print('X Features are \n' , BreastData.feature_names)

#y Data
y = BreastData.target
#print('y Data is \n' , y[:10])
#print('y shape is ' , y.shape)
#print('y Columns are \n' , BreastData.target_names)

#----------------------------------------------------
#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
#print('X_train shape is ' , X_train.shape)
#print('X_test shape is ' , X_test.shape)
#print('y_train shape is ' , y_train.shape)
#print('y_test shape is ' , y_test.shape)

#----------------------------------------------------
#Applying QDA Model 

'''
#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(priors=None,reg_param=0.0,store_covariance=False,
#                                                            tol=0.0001,store_covariances=None)
'''

QDAModel = QuadraticDiscriminantAnalysis(tol=0.0001)
QDAModel.fit(X_train, y_train)

#Calculating Details
print('QDAModel Train Score is : ' , QDAModel.score(X_train, y_train))
print('QDAModel Test Score is : ' , QDAModel.score(X_test, y_test))
print('QDAModel means are : ' , QDAModel.means_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = QDAModel.predict(X_test)
y_pred_prob = QDAModel.predict_proba(X_test)
print('Predicted Value for QDAModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for QDAModel is : ' , y_pred_prob[:10])

#----------------------------------------------------
#Calculating Confusion Matrix
CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()