###############################################################################
##
## Logistic Regression
##
## @author: Matthew Cline
## @version: 20171010
##
## Description: Logistic regression models mapping features of a flu survey
##              to the risk of contracting flu. Uses single variable models
##              as well as multivariable models. Also makes use of
##              regularization and feature scaling.
##
###############################################################################

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    predictions = []
    for value in x:
        prediction = (1/(1+math.exp(-value)))
        '''
        if prediction < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
        '''
        predictions.append(prediction)
    return predictions

def predict(weights, features):
    return sigmoid(features.dot(weights))

def gradient(weights, features, labels):
    predictions = predict(weights, features)
    error = predictions - labels
    elementError = error.T.dot(features)
    return elementError

def costFunction(weights, features, labels, threshold = 0.5):
    m = features.shape[0]
    predictions = predict(weights, features)
    cost = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for hx, y in zip(predictions, labels):
        if(y == 1 and hx >= threshold):
            tp += 1
        elif(y == 0 and hx < threshold):
            tn += 1
        elif(y == 1 and hx < threshold):
            fn += 1
        else:
            fp += 1
        cost += (y*math.log(hx) + (1-y)*math.log(1-hx))
    cost = cost*-1/m
    return cost, tp, tn, fp, fn

def gradientDescent(initWeights, alpha, features, labels, maxIterations, lam=0):
    weights = initWeights
    m = features.shape[0]
    costHistory = []
    tp, tn, fp, fn = 0, 0, 0, 0
    cost, tp, tn, fn, fp = costFunction(weights, features, labels)
    costHistory.append([0, cost])
    costChange = 1
    i = 1
    while(costChange > 0.000001 and i < maxIterations):
        oldCost = cost
        weights = map(lambda x: x * (1-alpha*lam/m), weights) - (alpha * gradient(weights, features, labels))
        cost, tp, tn, fp, fn = costFunction(weights, features, labels)
        costHistory.append([i, cost])
        costChange = oldCost - cost
        i+=1
    return weights, np.array(costHistory), tp, tn, fp, fn


####### GLOBAL VARIABLES ########
weights = np.array([0.63, 0.924])
dummyData = pd.DataFrame({'A':[1, 3, 4], 'B':[5, 6, 7]})
intercept = np.ones(len(dummyData['A']))
dummyData.insert(0, 'Intercept', intercept)

####### IMPORT DATA FROM EXCEL FILE INTO PANDAS STRUCTURE #######
data = pd.read_excel('fluML.xlsx', sheetname='Sheet1', parse_cols=[2,9,13,17,16])

#Clean out any records that have null values
data.dropna(subset=['HndWshQual', 'Risk', 'KnowlTrans', 'Gender', 'Flu'], inplace=True)

#Normalize HndWshQual Data
data['HndWshQual'] = (data['HndWshQual'] - np.mean(data['HndWshQual'] / np.std(data['HndWshQual'])))


####### EVALUATE RISK ONLY MODEL ########
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features = features.drop('HndWshQual', 1)
features = features.drop('KnowlTrans', 1)
features = features.drop('Gender', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.001, features, labels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk Model")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("\n\n")


######## EVALUATE RISK AND HAND WASH QUALITY MODEL #######
weights = [0.9483, -0.7584, 0.4965]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features = features.drop('KnowlTrans', 1)
features = features.drop('Gender', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.001, features, labels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk and Hand Wash Quality")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("\n\n")

######## EVALUATE RISK, HAND WASH QUALITY, AND KNOWLEDGE OF TRANS MODEL #######
weights = [0.9483, -0.7584, 0.4965, 0.9639]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features = features.drop('Gender', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.001, features, labels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, Hand Wash Quality, and Knowledge of Transmission Model")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("\n\n")


######## EVALUATE RISK, HNDWSHQUAL, KNOWLTRANS, AND GENDER #######
weights = [0.9483, -0.7584, 0.4965, 0.9639, 1.3542]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.001, features, labels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, Hand Wash Quality, Knowledge of Transmission, and Gender Model")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("\n\n")


######## EVALUATE RISK, HNDWSHQUAL, KNOWLTRANS, AND GENDER WITH REGULARIZATION #######
weights = [0.9483, -0.7584, 0.4965, 0.9639, 1.3542]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.001, features, labels, 100000, 0.8)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, Hand Wash Quality, Knowledge of Transmission, and Gender Model with Regularization")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("\n\n")
