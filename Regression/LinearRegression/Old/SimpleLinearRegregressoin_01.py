
#Simple Linear Regression#
#Simple linear regression is an approach for predicting a response using a single feature.
#It is assumed that the two variables are linearly related. Hence, we try to find a linear function that predicts the response value(y) as accurately as possible as a function of the feature or independent variable(x).
#Let us consider a dataset where we have a value of response y for every feature x: 
from PIL import Image
img = Image.open('python-linear-regression.png').convert('LA')
img.show (img)

# For generality, we define:
# x as feature vector, i.e x = [x_1, x_2, …., x_n],
# y as response vector, i.e y = [y_1, y_2, …., y_n]
# for n observations (in above example, n=10).
# A scatter plot of the above dataset looks like:-