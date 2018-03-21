"""
Linear regression with multi variable

Suppose you are selling your house and you
want to know what a good market price would be. One way to do this is to
first collect information on recent houses sold and make a model of housing
prices.

The file dataset.csv contains a training set of housing prices in Port-
land, Oregon. The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.

"""
import csv
import matplotlib.pyplot as plt
import numpy as np

def compute_cost(X,y,theta):
    m = len(y)
    J = 0
    hx = list()
    squarederrors = list()
    for i in X: 
        hofx = theta[0]*i[0] + theta[1]*i[1] + theta[2]*i[2]
        hx.append(hofx)
    for i in range(m):
        temp = float(hx[i]) - float(y[i])
        result = temp*temp
        squarederrors.append(result)
    J = (1/(2*m))*sum(squarederrors)
    J = np.array(J)
    return J    
        

def gradient_descent(X,y,theta,alpha,iterations):
    m = len(y)
    J_old_values = list()
    J_old_values.append(compute_cost(X,y,theta))
    for i in range(iterations):
        hofx = list()
        errorvalues = list()
        sum2 = 0
        sum3 = 0
        for i in X:
            temp1 =  theta[0]*i[0] + theta[1]*i[1] + theta[2]*i[2]
            hofx.append(temp1)
        for i in range(m):
            error = float(hofx[i]) - float(y[i])
            errorvalues.append(error)
        theta0 = theta[0] - ((alpha/m)*sum(errorvalues))
        for i in range(len(X)):
            sum2+=(errorvalues[i]*X[i][1])
            sum3+=(errorvalues[i]*X[i][2])
            temp2 = (alpha/m)*sum2
            temp3 = (alpha/m)*sum3
        theta1 = float(theta[1]) - float(temp2)
        theta2 = float(theta[2]) - float(temp3)
        theta = [theta0,theta1,theta2]
        J_old_values.append(compute_cost(X,y,theta))
    return theta,J_old_values

#reading the data from dataset.csv
print('\nLinear regression with multi variable to predict the price of the house')
dataset = list()
fp=open('dataset.csv','r')
reader = csv.reader(fp , delimiter=',')
for row in reader:
    dataset.append(row)   
m=len(dataset)
print('No number training samples : ',m) 
xvalues = list()
X = list()
y = list()
    
for i in range(m):
    X.append([1])

index = 0    
for i in dataset:
    X[index].append(float(i[0]))
    X[index].append(float(i[1]))
    y.append(i[2])
    index+=1

x1 = list()
x2 = list()

# Feature scaling in range of 0 to 1
for i in X:
    x1.append(i[1])
    x2.append(i[2])
    
min1 = min(x1)
min2 = min(x2)
max1 = max(x1)
max2 = max(x2)

t1 = max1 - min1
t2 = max2 - min2

for i in range(len(X)):
    X[i][1] = (X[i][1])/t1
    X[i][2] = (X[i][1])/t2


temp1 = ([0],[0],[0])
theta = np.array(temp1)
Xmat = np.array(X) 

no_of_iterations = 1500
alpha = 0.01

theta,J_cost = gradient_descent(X, y, theta, alpha, no_of_iterations)
print("\ntheta values are: \n",float(theta[0]),"\n",float(theta[1]),"\n",float(theta[2]))

plotx = list()
ploty = list()
for i in range(len(J_cost)):
    plotx.append(i)
for i in J_cost:
    ploty.append(i)
plt.xlabel('No of iterations')
plt.ylabel('Cost function "J" ')
plt.plot(plotx,ploty,'b')
plt.show()

while(True):
    input_size= float(input("\nEnter the size of house: (square-feet)\n"))
    input_noofbed = float(input("Enter the number of bedrooms : \n"))
    input_size = input_size/t1
    input_noofbed = input_noofbed/t2
    predict = theta[0] + input_size * theta[1] + input_noofbed * theta[2]
    print("\nPredicted house price based on given information is\n",round(float(predict),2),"$")   
    abc = int(input("-1 to exit , 5 to continue with one more house price prediction!\n"))
    if(abc==-1):
        break




    




    
    
