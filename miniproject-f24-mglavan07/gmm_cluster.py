import numpy as np
from sklearn.mixture import GaussianMixture


def best_k(data):

    """finds the best K value by the BIC test for clustering

    Args:
      data: an n-by-2 numpy array of numbers with n data points

    Returns:
      A single digit (which is an element from n_components, i.e., the optimal number of clusters) that results in the lowest
      BIC when it is used as the number of clusters to fit a GMM

    """
    d = data

    n_components = [1,2,3,4,5,6,7,8]
    
    gm = GaussianMixture(n_components = n_components[0], random_state=0).fit(d)
    best_bic = gm.bic(d)
    best_no_clusters = n_components[0]

    # for all different k values in n_components, make GMM model and calculate BIC
    for k in n_components:

      # fit GMM (remember to set random_state=0 when you call GaussianMixture())
      gm_ = GaussianMixture(n_components = k, random_state=0).fit(d)

      # calculate BIC
      bic = gm_.bic(d)

      # if current BIC is lower than best BIC, make it the best BIC and make its corresponding k the best_no_clusters
      if bic < best_bic:
        best_bic = bic
        best_no_clusters = k

    return best_no_clusters



