import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=";")  #in the csv file the values are seprated by ; and we do not want to read that
print(data.head())
data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head())

predict= "G3"   #this is called as label (basically what we want to predict using the dataset)
#we can have multiple labels if we need

x = np.array(data.drop([predict],1)) #this will reaturn a new data frame that does not have G3 in it
y = np.array(data[predict]) #this will be our lables
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

#Now we will split these up into 4 variables - x test, y test, x train, y train

"""best=0

for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
    #we have taken our lables and attributes and split them up into 4 arrays x_train will be a section of x arryy vice versa.
    #x_test/y_test will test acuracy of our model.
    #test_size=0.1 means that we split the data set into 10 percent for each test because if the computer already knows the whole data set the result will be inaccurate

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train) #will find best fit line using the xtrain and y train data
    acc= linear.score(x_test, y_test) #This will return a value that tells the accuracy of the model
    print(acc)
    if acc>best:
        best = acc
    with open("studentmodel.pickle", "wb") as f: #this will save model for later use
        pickle.dump(linear, f)"""

pickle_in = open("studentmodel.pickle", "rb") #read in pickle file
linear= pickle.load(pickle_in) #load pickel in linear model

print("Coefficient: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions= linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#ploting things on a grid

p="studytime"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"]) #sets up scatter plot
pyplot.xlabel(p)
pyplot.ylabel("Final grade")
pyplot.show()