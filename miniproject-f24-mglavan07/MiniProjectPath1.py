import pandas
from sklearn.cluster import KMeans
from data_sorting import make_video_dictionary, make_student_dictionary, at_least_n_videos, normalize_features
from gmm_cluster import *
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df

# sort data into dictionaries
video_dict = make_video_dictionary(dataset_1)
student_dict = make_student_dictionary(dataset_1)

# print(video_dict.keys()) --> there are 92 different videos - NO VIDEO #29
# print(student_dict.keys()) --> there are 3976 different student IDs

# identify students under 5 videos
video_dict_g5 = at_least_n_videos(video_dict, student_dict, 5)

# for each video
total_euclidean_distance = 0  # Initialize total Euclidean distance
total_clusters = 0

# cluster belonging data
cluster_matrix = []

for video_id in video_dict_g5.keys():

    # make a array for the features of interest
    X = []

    X.append(video_dict_g5[video_id][1]) # fracSpent
    X.append(video_dict_g5[video_id][2]) # fracComp
    X.append(video_dict_g5[video_id][4]) # fracPaused
    X.append(video_dict_g5[video_id][5]) # numPauses
    X.append(video_dict_g5[video_id][6]) # avgPBR
    X.append(video_dict_g5[video_id][8]) # numRWs
    X.append(video_dict_g5[video_id][9]) # numFFs

    # normalize the feature maxtrix x
    X_n = normalize_features(X)

    X_n = np.array(X_n).T

    # find the optimal cluster count by the BIC score
    k_bic = best_k(X_n)

    # Fit KMeans model
    kmeans = KMeans(n_clusters=k_bic, random_state=0)
    kmeans.fit(X_n)

    # Print the video ID and the number of clusters
    print(f"Video ID: {video_id}, Number of clusters: {k_bic}")

    # Add the number of clusters to the total count
    total_clusters += k_bic

    # Calculate the Euclidean distance for each point in each cluster
    for i in range(k_bic):
        cluster_points = X_n[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
        total_euclidean_distance += np.sum(distances)  # Sum the distances for this cluster

    # Create a list for each data point's cluster number
    cluster_labels = [label + 1 for label in kmeans.labels_]  # Convert labels to 1-based index
    cluster_matrix.append(cluster_labels)

# At the end of the loop, display the total Euclidean distance and the total number of clusters
print(f"Total Euclidean distance from all points to their centers across all clusters: {total_euclidean_distance:.4f}")
print(f"Total number of clusters across all models: {total_clusters}")

# Regression Model

# re-sort data into dictionaries
video_dict = make_video_dictionary(dataset_1)
student_dict = make_student_dictionary(dataset_1)

# Filter videos for students with at least 5 videos
video_dict_g5 = at_least_n_videos(video_dict, student_dict, 5)

# For storing regression results
ridge_results = {}
ridge_with_clusters_results = {}

# Ridge regularization strength
ridge_alpha = 1.0

# Loop through each video
for video_id in video_dict_g5.keys():
    video_data = video_dict_g5[video_id]

    z = []

    z += (video_data[1:3])
    z+= (video_data[4:7])
    z+= (video_data[8:10])

    # Create X and Y
    X = np.array(z).T  # Features: fracSpent to numFFs
    Y = np.array(video_data[10])  # Target: s (average score)

    # Perform KMeans clustering
    k_bic = best_k(X)  # Optimal number of clusters
    kmeans = KMeans(n_clusters=k_bic, random_state=0)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster information as a feature
    encoder = OneHotEncoder(sparse_output=False)
    cluster_features = encoder.fit_transform(cluster_labels.reshape(-1, 1))
    X_with_clusters = np.hstack([X, cluster_features])  # Concatenate features with clusters

    # Ridge Regression without clusters
    ridge_model = Ridge(alpha=ridge_alpha)
    ridge_model.fit(X, Y)
    Y_pred = ridge_model.predict(X)
    mse_ridge = mean_squared_error(Y, Y_pred)
    ridge_results[video_id] = {'mse': mse_ridge, 'coefficients': ridge_model.coef_, 'intercept': ridge_model.intercept_}

    # Ridge Regression with clusters
    ridge_with_clusters_model = Ridge(alpha=ridge_alpha)
    ridge_with_clusters_model.fit(X_with_clusters, Y)
    Y_pred_with_clusters = ridge_with_clusters_model.predict(X_with_clusters)
    mse_ridge_with_clusters = mean_squared_error(Y, Y_pred_with_clusters)
    ridge_with_clusters_results[video_id] = {
        'mse': mse_ridge_with_clusters,
        'coefficients': ridge_with_clusters_model.coef_
    }
    # Print coefficients for this video
    print(f"Video ID: {video_id}")
    print(f"  Normalized Coefficients without Clusters: {ridge_results[video_id]['coefficients']}, [{ridge_results[video_id]['intercept']}]")

# Summarize results
print("\nSummary of Ridge Regression Results:")
for video_id in ridge_results:
    print(f"Video ID: {video_id}")
    print(f"  MSE without clusters: {ridge_results[video_id]['mse']:.4f}")
    print(f"  MSE with clusters: {ridge_with_clusters_results[video_id]['mse']:.4f}\n")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Define a range of lambda values to test
lambda_values = np.logspace(-3, 3, 50)  # 50 values from 10^-3 to 10^3
sum_mse_per_lambda = []  # To store the sum of MSEs for all videos for each lambda

# Loop through each lambda value
for lambda_val in lambda_values:
    total_mse = 0  # Initialize the sum of MSEs for this lambda
    
    # Loop through each video
    for video_id in video_dict_g5.keys():
        video_data = video_dict_g5[video_id]
        z=[]

        z += (video_data[1:3])
        z+= (video_data[4:7])
        z+= (video_data[8:10])
        
        # Create X and Y
        X = np.array(z).T  # Features: fracSpent to numFFs
        Y = np.array(video_data[10])  # Target: s (average score)
        
        # Fit Ridge Regression with current lambda
        ridge_model = Ridge(alpha=lambda_val)
        ridge_model.fit(X, Y)
        Y_pred = ridge_model.predict(X)
        
        # Calculate MSE for this video and add to total
        mse = mean_squared_error(Y, Y_pred)
        total_mse += mse
    
    # Store the sum of MSEs for this lambda
    sum_mse_per_lambda.append(total_mse)

# Plot the sum of MSEs against lambda values
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, sum_mse_per_lambda, marker='o', color='purple', label="Sum of MSEs Across Videos")
plt.xscale('log')
plt.xlabel("Lambda (Regularization Strength)")
plt.ylabel("Sum of Mean Squared Errors")
plt.title("Lambda vs Sum of MSEs Across All Videos")
plt.grid(True)
plt.legend()
plt.show()

# Naieve Bayes Classification Model
error_total = 0

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# train a classifier for each video
for video_id in video_dict.keys():
    
    # make a array for the features of interest
    X = []

    X.append(video_dict_g5[video_id][1]) # fracSpent
    X.append(video_dict_g5[video_id][2]) # fracComp
    X.append(video_dict_g5[video_id][4]) # fracPaused
    X.append(video_dict_g5[video_id][5]) # numPauses
    X.append(video_dict_g5[video_id][6]) # avgPBR
    X.append(video_dict_g5[video_id][8]) # numRWs
    X.append(video_dict_g5[video_id][9]) # numFFs

    X = np.array(X).T
    X = scaler.fit_transform(X)

    y = np.array(video_dict_g5[video_id][len(video_dict_g5[video_id]) - 1]) # score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    print(f"Gaussian Naive Bayes model accuracy(in %) for video {video_id}:", metrics.accuracy_score(y_test, y_pred)*100)
    error_total += metrics.accuracy_score(y_test, y_pred)*100

print(f"average accuracy is {error_total / len(list(video_dict.keys()))}%")