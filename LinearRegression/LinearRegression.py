
#Linear Regression
# Linear regression is a quiet and simple statistical regression method used for predictive analysis and shows the 
# relationship between the continuous variables. Linear regression shows the linear relationship between the 
# independent variable (X-axis) and the dependent variable (Y-axis), consequently called linear regression.
# If there is a single input variable (x), such linear regression is called simple linear regression. 
# And if there is more than one input variable, such linear regression is called multiple linear regression. 
# The linear regression model gives a sloped straight line describing the relationship within the variables.

# Use case
# In this, I will take random numbers for the dependent variable (salary) and an independent variable (experience) and will predict the impact of a year of experience on salary.
# Steps to implement Linear regression model
# Import some required libraries
# y= Dependent Variable.
# x= Independent Variable.
# a0= intercept of the line.
# a1 = Linear regression coefficient.

# Need of a Linear regression
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape
import pandas as pd
import numpy as np

# Define the dataset
x=  np.array([1,2,2,3,4,5,5])
y = np.array([7,8,7,9,11,10,12])
n = np.size(x)
experience = x #Independent Variable.
salary = y #Dependent Variable.

# print(experience)
# print(salary)

# Plot the data points
plt.style.use('dark_background')
plt.scatter(x,y, color = 'red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

# The main function to calculate values of coefficients
# Initialize the parameters.
# Predict the value of a dependent variable by given an independent variable.
# Calculate the error in prediction for all data points.
# Calculate partial derivative w.r.t a0 and a1.
# Calculate the cost for each number and add them.
# Update the values of a0 and a1.

#initialize the parameters
a0 = 5                  # Intercept Theta 0 
a1 = 2                  # Slop Theta 1
lr = 0.0001             # Learning rate
iterations = 1000       # Number of iterations
error = []              # Error array to calculate cost for each iterations.
for itr in range(iterations):
    error_cost = 0
    cost_a0 = 0
    cost_a1 = 0
    for i in range(len(experience)):
        y_pred = a0+a1*experience[i]   # Predict value for given x
        error_cost = error_cost +(salary[i]-y_pred)**2
        for j in range(len(experience)):
            partial_wrt_a0 = -2 *(salary[j] - (a0 + a1*experience[j]))                # partial derivative w.r.t a0 المشتقات الجزئية 
            partial_wrt_a1 = (-2*experience[j])*(salary[j]-(a0 + a1*experience[j]))   # partial derivative w.r.t a1 المشتقات الجزئية 
            cost_a0 = cost_a0 + partial_wrt_a0      # calculate cost for each number and add
            cost_a1 = cost_a1 + partial_wrt_a1      # calculate cost for each number and add
        a0 = a0 - lr * cost_a0    # update a0
        a1 = a1 - lr * cost_a1    # update a1
        print(itr,a0,a1)          # Check iteration and updated a0 and a1
    error.append(error_cost)      # Append the data in array

# At approximate iteration 50- 60, we got the value of a0 and a1.

print(a0)
print(a1)

#Plotting the error for each iteration.
plt.figure(figsize=(10,5))
plt.plot(np.arange(1,len(error)+1),error,color='red',linewidth = 5)
plt.title("Iteration vr error")
plt.xlabel("iterations")
plt.ylabel("Error")

#Predicting the values.
pred = a0+a1*experience
print(pred)

#Plot the regression line.
plt.scatter(experience,salary,color = 'red')
plt.plot(experience,pred, color = 'green')
plt.xlabel("experience")
plt.ylabel("salary")

#Analyze the performance of the model by calculating the mean squared error.
error1 = salary - pred
se = np.sum(error1 ** 2)
mse = se/n
print("mean squared error is", mse)

#Use the scikit library to confirm the above steps.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
experience = experience.reshape(-1,1)
model = LinearRegression()
model.fit(experience,salary)
salary_pred = model.predict(experience)
Mse = mean_squared_error(salary, salary_pred)
print('slop', model.coef_)
print("Intercept", model.intercept_)
print("MSE", Mse)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

import sklearn.linear_model as lm
# Create linear regression object

#B = np.reshape(x_train, (-1, 2))
#C = np.reshape(y_train, (-1, 2))
#D = np.reshape(x_test,(-1,2))
lr = lm.LinearRegression()
lr.fit(x_train, y_train)

# Predicting the Test set results
y_pred = lr.predict(x_test)

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#lr.predict([[12]])


a_1d_array = np.array([1, 2, 3, 4])
print(a_1d_array)
reshaped_to_2d = np.reshape(a_1d_array, (-1, 2))
print(reshaped_to_2d)

a = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
for i in range(len(a)):
    for j in range(len(a[i])):
        print(a[i][j], end=' ')
    print()