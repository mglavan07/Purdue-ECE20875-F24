import numpy as np
import matplotlib.pyplot as plt

# Function that returns fitted model parameters to the dataset at datapath for each choice in degrees.
# 

def main(datapath, degrees):
    
    # Input
    # --------------------
    # datapath : A string specifying a .txt file 
    # degrees : A list of positive integers.
    #    
    # Output
    # --------------------
    # paramFits : a list with the same length as degrees, where paramFits[i] is the list of
    #             coefficients when fitting a polynomial of d = degrees[i].
    
    paramFits = []

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))
        
    # iterate through each n in the list degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.
    for n in degrees:
        A = feature_matrix(x, n)
        B = least_squares(A, y)
        paramFits.append(B)

    return paramFits

# Function that returns the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
#

def feature_matrix(x, d):
    
    # Input
    # --------------------
    # x: A list of the independent variable samples
    # d: An integer
    #
    # Output
    # --------------------
    # X : A list of features for each sample, where X[i][j] corresponds to the jth coefficient
    #     for the ith sample. Viewed as a matrix, X should have dimension (samples, d+1).

    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    # Please be aware of which matrix colum corresponds to which degree polynomial when completing the writeup.
 
    X = [[samp ** (d - degree) for degree in range(d+1)] for samp in x]
    return X

# Function that returns the least squares solution based on the feature matrix X and corresponding target variable samples in y.
def least_squares(X, y):
    # Input
    # --------------------
    # X : A list of features for each sample
    # y : a list of target variable samples.
    #
    # Outut
    # --------------------
    # B : a list of the fitted model parameters based on the least squares solution.
    
    X = np.array(X)
    y = np.array(y)

    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = list(np.linalg.inv(X.T @ X) @ X.T @ y)
    return B

if __name__ == "__main__":
    datapath = "poly.txt"

    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))

    ### Problem 1 ###
    # TODO: Complete 'main, 'feature_matrix', and 'least_squares' functions above

    ### Problem 2 ###
    # The paramater values for degrees 2 and 4 have been provided as test cases in the README.
    # The output should match up to at least 3 decimal places rounded 
    
    # Write out the resulting estimated functions for each d.
    degrees = [1, 2, 3, 4, 5, 6] # TODO: Update the degrees,d to include 1, 3, 5 and 6. i.e. [1,2,3,4,5,6] and
    paramFits = main(datapath, degrees)
    for idx in range(len(degrees)):
        print("y_hat(x_"+str(degrees[idx])+")")
        print(paramFits[idx])
        print("****************")

    ### Problem 3 ###
    # TODO: Visualize the dataset and these fitted models on a single graph
    # Use the 'scatter' and 'plot' functions in the `matplotlib.pyplot` module.

    # Draw a scatter plot
    plt.scatter(x, y, color='black', label='data')
    x.sort() # Ensures the scatter plot looks nice

    # define 6 colors for the new plots
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    # iterate to generate 6 new lines
    for params in paramFits:

        # get a feature matrix
        A = np.array(feature_matrix(x, len(params) - 1))

        # perform matrix multiplicaiton
        y_predicted = list(A @ np.array(params).T)

        # plot the new line
        plt.plot(x, y_predicted, label=f'Model Degree: {len(params)-1}', color = colors[len(params)-2])
        
    
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title("Polynomial Regression of Degrees 1-6")
    plt.legend(fontsize=10, loc='upper left')

    plt.show()

    ### Problem 4 ###
    # TODO: when x = 2; what is the predicted output
    # Use the degree that best matches the data as determined in Problem 3 above.
    
    # use degree 3
    y_2 = 0
    degree = 3
    coeffs = paramFits[degree - 1]
    for i in coeffs:
        y_2 += i * 2 ** degree
        degree -= 1

    print(f'y_hat(2) at Degree = 3 Regression is: {y_2}')



