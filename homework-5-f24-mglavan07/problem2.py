import numpy as np
import math as m
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t

# import data
myFile = open('city_vehicle_survey.txt')
data1 = myFile.readlines()
data1 = [float(x) for x in data1]
myFile.close()

# code for question 2
print('Problem 2 Answers:')

# hypothesis test
mu = 5

# calculate sample statistics
n = len(data1)
x_bar = np.mean(data1)
stdev = np.std(data1, ddof=1)

# standardize the sample mean
SE = stdev / m.sqrt(n)
z_score = (x_bar - 5) / SE

# compute the p value
p = 2 * norm.cdf(-abs(z_score))

# print statements
print(f'The sample size was: {n} \nThe sample mean was: {x_bar:.4f} \nThe standard error was: {SE:.4f} \nThe standard score was: {z_score:.4f} \nThe p-value was: {p:.4f}\n')

# code for question 3
print('Problem 3 Answers:')

# define a significance level and respective 2-tailed p
a = 0.05
p = a / 2

# standardize the p value
z_score = abs(norm.ppf(p))

# compute the error
SE = (x_bar - mu) / z_score

# compute n
n = (stdev / SE) ** 2

# print statements
print(f'The largest standard error is: {SE:.4f} \nThis occurs at n = {n:.0f}\n')

# code for question 5
print('Problem 5 Answers:')

# import data
myFile1 = open('vehicle_data_1.txt')
data1 = myFile1.readlines()

myFile2 = open('vehicle_data_2.txt')
data2 = myFile2.readlines()

data1 = [float(x) for x in data1]
data2 = [float(y) for y in data2]
myFile1.close()
myFile2.close()

# collect sample statistics
n0 = len(data1) # emissions
n1 = len(data2) # no emissions
x_bar0 = np.mean(data1)
x_bar1 = np.mean(data2)
stdev0 = np.std(data1, ddof=1)
stdev1 = np.std(data2, ddof=1)

# standardize the sample difference in mean
SE = m.sqrt(stdev0 ** 2 / n0 + stdev1 ** 2 / n1)
z_score = (x_bar0 - x_bar1 - 0) / SE

# compute p value
p = 2 * norm.cdf(-abs(z_score))

# print statement
print(f'For exhaust: \nn = {n0}\nmean = {x_bar0:.4f}\n\nFor no exhaust:\nn = {n1}\nmean = {x_bar1:.4f}\n\nFor the hypothesis test:\nStandard Error = {SE:.4f}\nZ-Score = {z_score:.4f}\nP-value = {p:.4e}\n')








