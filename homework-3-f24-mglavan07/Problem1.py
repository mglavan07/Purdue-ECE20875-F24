import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(histogram):
    """
    takes a list of counts and converts to a list of probabilities, outputs the probability list.
    :param histogram: a numpy ndarray object
    :return: list
    """ 
    # convert the np array into a list (optional but eases debugging)
    hist_list = list(histogram)

    # compute the sample size
    n = 0
    for val in hist_list:
        n += float(val) # prevent integer truncating 

    # create an empty list for storing probabilities
    p = []

    # fill in the probabilities (bin height / n)
    for h in hist_list:
        p.append(float(h) / float(n))

    # return the normalized p values
    return p

def compute_j(histogram, bin_width, num_samples):
    """
    takes list of counts, uses norm_histogram function to output the histogram of probabilities, 
    then calculates compute_j for one specific bin width (reference: histogram.pdf page19)
    :param histogram: list
    :param bin_width: float
    :param num_samples: int
    :return: float
    """

    # gather variables in the equation as it appears on p19
    m = num_samples
    w = bin_width
    p = norm_histogram(histogram)

    # find sum of squared probabilities
    ssp = 0.0
    for prob in p:
        ssp += float(prob ** 2)

    # use the fomula and return j
    return (2.0 / float((m - 1) * w) - (float(m + 1) / float((m - 1) * w)) * ssp)

def sweep_n(data, min_val, max_val, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep
    
    The variable "data" is the raw data that still needs to be "processed"
    with matplotlib.pyplot.hist to output the histogram

    You must utilize the variables (data, min_val, max_val, min_bins, max_bins) 
    in your code for 'sweep_n' to determine the correct input to the function 'matplotlib.pyplot.hist',
    specifically the values to (x, bins, range).
    Other input variables of 'matplotlib.pyplot.hist' can be set as default value.
    
    :param data: list
    :param min_val: int
    :param max_val: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    # make a destination for J(w) values
    j_w_list = []

    # iterate from min bins to max bins
    for n in range(min_bins, max_bins + 1):

        # n will be the number of bins --> convert this into bin width w
        w = float((max_val - min_val)) / float(n)

        # make the histogram array for compute_j to accept
        hist, edges, patches  = plt.hist(data, n)

        # compute j and add it to the list
        j_w_list.append(compute_j(hist, w, len(data)))

    # return the list of j values
    return j_w_list


def find_min(l):
    """
    takes a list of numbers and returns the three smallest number in that list and their index.
    return as a list of tuples i.e. 
    [(index_of_the_smallest_value, the_smallest_value), (index_of_the_second_smallest_value, the_second_smallest_value)]
    
    For example:
        A list(l) is [14,27,15,49,23,41,147]
        Then you should return [(0, 14), (2, 15), (4, 23)]

    :param l: list
    :return: list of tuples
    """
    # the function will iterate through the list three times, changing the min to max + 1
    l_copy = l.copy()

    # find the maximum in the list
    l_max = l_copy[0]
    for item in l_copy:
        if item > l_max:
            l_max = item

    # create a list for the minima 
    minima = []
    min_count = 3

    # find three minima
    for i in range(0,min_count):

        # set the loop variables
        l_min = l_copy[0]
        idx_min = 0

        # iterate through the list
        for idx, val in enumerate(l_copy):
            if val < l_min:
                l_min = val
                idx_min = idx

        # pack the list and send to "minima"
        min_tuple = (idx_min, l_min) 
        minima.append(min_tuple)

        # set the val of l in the copy to the max value 
        l_copy[idx_min] = l_max

    # return the list of minima
    return minima 

if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))