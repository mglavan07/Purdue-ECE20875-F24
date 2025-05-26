import math as m
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t

# import or paste dataset here
data = np.array([135, 140, 130, 145, 150, 138, 142, 137, 136, 148, 141, 139, 143, 147, 149, 134, 133, 146, 144, 132])


# code for Question 1
print('Problem 1 Answers:')

# C level
C = 0.9

# compute sample statistics
n = len(data)
x_bar = np.mean(data)
stdev = np.std(data, ddof=1)

# standardize the test statistics
SE = stdev / m.sqrt(n)
t_c = t.ppf(1 - (1-C)/2, n-1)

# constuct the bounds of the CI
interval = t_c * SE 
(lower, upper) = x_bar - interval, x_bar + interval

# print statement
print(f'The mean = {x_bar:.4f}\nThe standard error = {SE:.4f}\nThe critical t-score = {t_c:.4f}\nThe confidence interval bounds = ({lower:.4f}, {upper:.4f})\n')

# code for Question 2
print('Problem 2 Answers:')

# C level
C = 0.95

# compute sample statistics
n = len(data)
x_bar = np.mean(data)
stdev = np.std(data, ddof=1)

# standardize the test statistics
SE = stdev / m.sqrt(n)
t_c = t.ppf(1 - (1-C)/2, n-1)

# constuct the bounds of the CI
interval = t_c * SE 
(lower, upper) = x_bar - interval, x_bar + interval

# print statement
print(f'The mean = {x_bar:.4f}\nThe standard error = {SE:.4f}\nThe critical t-score = {t_c:.4f}\nThe confidence interval bounds = ({lower:.4f}, {upper:.4f})\n')

# code for Question 3
print('Problem 3 Answers:')

# C level
C = 0.95

# compute sample statistics
n = len(data)
stdev = 5
x_bar = np.mean(data)

# standardize the sample
SE = stdev / m.sqrt(n)
z_c = norm.ppf(1 - (1-C)/2)

# construct the bounds of the CI
interval = z_c * SE 
(lower, upper) = x_bar - interval, x_bar + interval

# print statement
print(f'The mean = {x_bar:.4f}\nThe standard error = {SE:.4f}\nThe critical z-score = {z_c:.4f}\nThe confidence interval bounds = ({lower:.4f}, {upper:.4f})\n')