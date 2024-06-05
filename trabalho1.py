# -*- coding: utf-8 -*-
import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as matplot

#Dataset

url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
dataset = pandas.read_csv(url_data, header=None)
dataset #expoe o dataset

subset1 = dataset.iloc[0:50, 4]
subset2 = dataset.iloc[100:150, 4]
label = pandas.concat([subset1, subset2]).values

label = numpy.where(label == 'Iris-setosa', 0, 1)

widthsetosa = dataset.iloc[0:50, [1, 3]]
widthtvirginica = dataset.iloc[100:150, [1, 3]]

width = pandas.concat([widthsetosa, widthtvirginica]).values

matplot.scatter(width[:50, 0], width[:50, 1], color='red', marker='o')
matplot.scatter(width[50:100, 0], width[50:100, 1], color='blue', marker='x')
matplot.show()

matplot.scatter(width[:50, 0], width[:50, 1], color='red', marker='o', label='setosa')
matplot.scatter(width[50:100, 0], width[50:100, 1], color='blue', marker='x', label='virginica')
matplot.xlabel('sepal width [cm]')
matplot.ylabel('petal width [cm]')
matplot.legend(loc='upper right')
matplot.show()

#Building the model - Perceptron

def hardlim(label):
    return 1 if label >= 0 else 0

def predict(p, w, b):
    return hardlim(numpy.dot(w, p) + b)

def fit(x, label, epochs):
    rgen = numpy.random.RandomState(7)
    w = rgen.normal(loc=0.0, scale=0.1, size=x.shape[1])
    b = rgen.normal(loc=0.0, scale=0.1, size=1)
    for _ in range(epochs):
        for p, t in zip(x, label):
            labelp = predict(p, w, b)
            e = t - labelp
            w += e * p
            b += e
    return w, b

#Accuracy

def calculate_accuracy(x, label, w, b):
    predictions = [predict(p, w, b) for p in x]
    accuracy = numpy.mean(predictions == label)
    return accuracy

#Train model

w, b = fit(width, label, 10)

matplot.scatter(width[:50, 0], width[:50, 1], color='red', marker='o', label='Setosa')
matplot.scatter(width[50:100, 0], width[50:100, 1], color='blue', marker='x', label='Virginica')

x1 = numpy.linspace(width[:, 0].min(), width[:, 0].max(), 100)
x2 = (-w[0] * x1 - b) / w[1]
matplot.plot(x1, x2, color='green')

matplot.xlabel('sepal width [cm]')
matplot.ylabel('petal width [cm]')
matplot.legend(loc='upper right')
matplot.show()

accuracy = calculate_accuracy(width, label, w, b)
print(f'Model accuracy: {accuracy:.2f}')
