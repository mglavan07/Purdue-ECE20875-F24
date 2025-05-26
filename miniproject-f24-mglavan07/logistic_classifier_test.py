
import pandas 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

# Filter students who completed at least 5 videos
video_counts = df.groupby('userID')['VidID'].count()
valid_users = video_counts[video_counts >= 5].index
filtered_data = df[df['userID'].isin(valid_users)].copy()  # Explicitly create a copy

# Normalize features
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
scaler = StandardScaler()
normalized_features = scaler.fit_transform(filtered_data[features])

# Add normalized features back to the DataFrame
for i, feature in enumerate(features):
    filtered_data.loc[:, f'norm_{feature}'] = normalized_features[:, i]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=7, random_state=0)  # Adjust `n_clusters` as needed
filtered_data['cluster'] = kmeans.fit_predict(normalized_features)

# Evaluate cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Calculate average score for each student
avg_scores = filtered_data.groupby('userID')['s'].mean().reset_index()
avg_scores.rename(columns={'s': 'avg_score'}, inplace=True)

# Merge average scores back to filtered data
merged_data = filtered_data.merge(avg_scores, on='userID')

# Prepare features and target
features_with_cluster = normalized_features.copy()
features_with_cluster = pandas.DataFrame(features_with_cluster, columns=features)
features_with_cluster['cluster'] = filtered_data['cluster']
target = merged_data['avg_score']

X_train, X_test, y_train, y_test = train_test_split(features_with_cluster, target, test_size=0.2, random_state=0)

# Calculate mean and std for each feature
means = filtered_data[features].mean()
stds = filtered_data[features].std()

print("Feature Means:\n", means)
print("Feature Standard Deviations:\n", stds)

# Classify a student's behavior based on likelihood of belonging to a Gaussian cluster
student_example = filtered_data.iloc[0][features]
z_scores = (student_example - means) / stds
print("Z-scores for Student:\n", z_scores)

# Train model
X = normalized_features
y  = target
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
print("Mean Squared Error:", mean_squared_error(y, y_pred))

# Prepare data for classification
X = normalized_features
y = filtered_data['s']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train classifier
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
