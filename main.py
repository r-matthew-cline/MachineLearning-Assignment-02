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

####### IMPORTS ########
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

def precision(tp, fp):
    if(tp+fp == 0):
        return 0
    return tp / (tp+fp*1.0)

def recall(tp, fn):
    if(tp + fn == 0):
        return 0
    return tp / (tp+fn*1.0)

def f1Score(tp, fp, fn):
    return 2*(precision(tp, fp) * recall(tp, fn) / (precision(tp, fp) + recall(tp, fn)))

def predict(weights, features):
    return sigmoid(features.dot(weights))

def gradient(weights, features, labels):
    predictions = predict(weights, features)
    error = predictions - labels
    elementError = error.T.dot(features)
    return elementError

def costFunction(weights, features, labels, threshold = 0.3, lam=0.0):
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
    cost = (cost*-1/m) + (lam/(2*m)*sum(weights))
    return cost, tp, tn, fp, fn

def gradientDescent(initWeights, alpha, features, labels, maxIterations, lam=0):
    weights = initWeights
    m = features.shape[0]
    costHistory = []
    tp, tn, fp, fn = 0, 0, 0, 0
    cost, tp, tn, fn, fp = costFunction(weights, features, labels, lam=lam)
    costHistory.append([0, cost])
    costChange = 1
    i = 1
    while(costChange > 0.000001 and i < maxIterations):
        oldCost = cost
        weights = map(lambda x: x * (1-alpha*lam/m), weights) - (alpha / m * gradient(weights, features, labels))
        cost, tp, tn, fp, fn = costFunction(weights, features, labels, lam=lam)
        costHistory.append([i, cost])
        costChange = oldCost - cost
        i+=1
    return weights, np.array(costHistory), tp, tn, fp, fn

def optimizeThreshold(weights, valData, valLabels, lam=0.0):
    threshold = 0.1
    finalThresh = 0
    f1 = 0
    while(threshold < 1):
        cost, tp, tn, fp, fn = costFunction(weights, valData, valLabels, threshold, lam=lam)
        if((precision(tp, fp) == 0) and (recall(tp, fn) == 0)):
            threshold += 0.001
            continue
        tempF1 = f1Score(tp, fp, fn)
        if(tempF1 > f1):
            f1 = tempF1
            finalThresh = threshold
        threshold += 0.001
    return finalThresh


def splitData(data, trainingSplit=0.6, validationSplit=0.8):
    training, validation, test = np.split(data, [int(trainingSplit*len(data)), int(validationSplit*len(data))])
    return training, validation, test


####### GLOBAL VARIABLES ########
weights = np.array([0.63, 0.924])
dummyData = pd.DataFrame({'A':[1, 3, 4], 'B':[5, 6, 7]})
intercept = np.ones(len(dummyData['A']))
dummyData.insert(0, 'Intercept', intercept)


####### IMPORT DATA FROM EXCEL FILE INTO PANDAS STRUCTURE #######
data = pd.read_excel('fluML.xlsx', sheetname='Sheet1', parse_cols=[2,9,13,17,16])

#Clean out any records that have null values
data.dropna(subset=['HndWshQual', 'Risk', 'KnowlTrans', 'Gender', 'Flu'], inplace=True)

# #Normalize HndWshQual Data
# data['HndWshQual'] = (data['HndWshQual'] - np.mean(data['HndWshQual'] / np.std(data['HndWshQual'])))

# Shuffle Data and reset the indicies to 0:len
data.reindex(np.random.permutation(data.index))
data.reset_index(drop=True)



####### EVALUATE RISK ONLY MODEL ########
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features = features.drop('HndWshQual', 1)
features = features.drop('KnowlTrans', 1)
features = features.drop('Gender', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']

trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk Model Training")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels)

print("Risk Model Validation")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels)

cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold)

print("Risk Model Test")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
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

# Split data for training test and validation
trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

# Train the model
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk and HndWshQual Model Training")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

#Optimize the classification threshold on the validation set
cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels)

print("Risk and HndWshQual Model Validation")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels)

# Evaluate the model against the test data
cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold)

print("Risk and HndWshQual Model Test")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")


####### EVALUATE RISK, HAND WASH QUALITY, AND KNOWLEDGE OF TRANS MODEL #######
weights = [0.9483, -0.7584, 0.4965, 0.9639]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features = features.drop('Gender', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']

# Split data for training test and validation
trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

# Train the model
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, HndWshQual, and KnowlTrans Model Training")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

#Optimize the classification threshold on the validation set
cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels)

print("Risk, HndWshQual, and KnowlTrans Model Validation")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels)

# Evaluate the model against the test data
cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold)

print("Risk, HndWshQual, and KnowlTrans Model Test")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")


######## EVALUATE RISK, HNDWSHQUAL, KNOWLTRANS, AND GENDER #######
weights = [0.9483, -0.7584, 0.4965, 0.9639, 1.3542]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']

# Split data for training test and validation
trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

# Train the model
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, HndWshQual, KnowlTrans, and Gender Model Training")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

#Optimize the classification threshold on the validation set
cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Validation")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels)

# Evaluate the model against the test data
cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Test")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")



######## EVALUATE RISK, HNDWSHQUAL, KNOWLTRANS, AND GENDER WITH REGULARIZATION #######
weights = [0.9483, -0.7584, 0.4965, 0.9639, 1.3542]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']

# Split data for training test and validation
trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

# Train the model
lamValues = [1, 2, 5, 10, 20, 50, 100]
regValue = 5
previousCost = 100
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000, lam=regValue)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, HndWshQual, KnowlTrans, and Gender Model Training With Regularization")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

#Optimize the classification threshold on the validation set
cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels, lam=regValue)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Validation With Regularization")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels, lam=regValue)

# Evaluate the model against the test data
cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold, lam=regValue)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Test with Regularization")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")



##### EVALUATE RISK, HNDWSHQUAL, KNOWLTRANS, AND GENDER WITH REGULARIZATION AND FEATURE SCALING #######

#Normalize HndWshQual Data
data['HndWshQual'] = (data['HndWshQual'] - np.mean(data['HndWshQual'] / np.std(data['HndWshQual'])))

#Normalize HndWshQual Data
data['Risk'] = (data['Risk'] - np.mean(data['Risk'] / np.std(data['Risk'])))

#Normalize HndWshQual Data
data['KnowlTrans'] = (data['KnowlTrans'] - np.mean(data['KnowlTrans'] / np.std(data['KnowlTrans'])))

weights = [0.9483, -0.7584, 0.4965, 0.9639, 1.3542]
dataLen = data.shape[0]
intercept = np.ones(dataLen)
features = data.drop('Flu', 1)
features.insert(0, 'Intercept', intercept)
labels = data['Flu']

# Split data for training test and validation
trainingFeatures, validationFeatures, testFeatures = splitData(features)
trainingLabels, validationLabels, testLabels = splitData(labels)

# Train the model
lamValues = [1, 2, 5, 10, 20, 50, 100]
regValue = 5
previousCost = 100
costs = []
tp, tn, fp, fn = 0, 0, 0, 0
weights, costs, tp, tn, fp, fn = gradientDescent(weights, 0.01, trainingFeatures, trainingLabels, 100000, lam=regValue)
plt.plot(costs.transpose()[0], costs.transpose()[1], 'o')
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()

print("Risk, HndWshQual, KnowlTrans, and Gender Model Training With Regularization With Feature Scaling")
print("Iterations", len(costs))
print("Cost", costs[-1][1])
print("Final Weights", weights)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

#Optimize the classification threshold on the validation set
cost, tp, tn, fp, fn = costFunction(weights, validationFeatures, validationLabels, lam=regValue)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Validation With Regularization With Feature Scaling")
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")

newThreshold = optimizeThreshold(weights, validationFeatures, validationLabels, lam=regValue)

# Evaluate the model against the test data
cost, tp, tn, fp, fn = costFunction(weights, testFeatures, testLabels, newThreshold, lam=regValue)

print("Risk, HndWshQual, KnowlTrans, and Gender Model Test with Regularization With Feature Scaling")
print("Final Weights", weights)
print("Final Threshold", newThreshold)
print('True Positive', tp)
print('True Negative', tn)
print('False Positive', fp)
print('False Negative', fn)
print("Precision", precision(tp, fp))
print("Recall", recall(tp, fn))
print("F1 Score", f1Score(tp, fp, fn))
print("\n\n")
