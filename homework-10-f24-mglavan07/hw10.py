import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import scikit-learn metrics module for accuracy and AUROC calculation
from sklearn import metrics

from utils import load_Dataset

def conf_matrix(y_pred, y_true, num_class):
    """
    agrs:
    y_pred : List of predicted classes
    y_true : List of corresponding true class labels
    num_class : The number of distinct classes being predicted

    Returns:
    M : Confusion matrix as a numpy array with dimensions (num_class, num_class)
    """
    # initialize a num_class x num_class array of zeros
    M = np.zeros((num_class, num_class), dtype=float)

    # iterate through the parallel lists
    for i in range(len(y_pred)):

        # get the actual condition (row)
        actual = y_true[i]

        # get the predicted conditon (col)
        prediction = y_pred[i]

        # determine the row,col coordinate assuming actual[i] and prediction[i] are intergers from 0-num_class
        row, col = actual, prediction

        # increment one position in M at this coordinate
        M[int(row)][int(col)] += 1

    # return the confusion matrix
    return M

def get_model(name, params):
    """
    args:
    name : Model name (string)
    params : list of parameters corresponding to given model

    Returns:
    model : sklearn model object
    """
    model = None
    if name == "KNN":
        k = params # Note that the expected parameters have already been extracted here
        # Define KNN model using sklearn KNeighborsClassifier object 
        # Note: you should include n_neighbors=k as an argument when initializing the model
        model = KNeighborsClassifier(n_neighbors=k)
    
    elif name == "SVM":
        rand_state, prob = params # Note that the expected parameters have already been extracted here
        # Define SVM model using sklearn SVC object
        # Note: you should include random_state=rand_state and probability=prob as arguments when initializing the model
        model = SVC(random_state = rand_state, probability=prob)
    
    elif name == "MLP":
        hl_sizes, rand_state, act_func = params # Note that the expected parameters have already been extracted here
        # Define MLP model using sklearn MLPClassifier object
        # Note: you should include hidden_layer_sizes=hl_sizes, random_state=rand_state, and activation=act_func when initializing the model
        model = MLPClassifier(hidden_layer_sizes = hl_sizes, random_state = rand_state, activation = act_func)

    else:
        print("ERROR: Model name not recognized/supported. Returned None")

    return model

def get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class):
    """
    args:
    model_name : Model name as a string
    params : List of parameters corresponding to the given model 
    train_Data : 5000x784 numpy array of FMNIST training images
    train_labels : corresponding 5000 numpy array of strings containing ground truth labels
    test_Data : 1000x784 numpy array of FMNIST test images
    test_labels : corresponding 1000 numpy array of strings containing ground truth labels
    num_class : integer number of unique classes being predicted

    Returns: 
    accuracy : Total model accuracy (numpy float) 
    confusion matrix: numpy array of dimensions (num_class,num_class)
    auc_score : Area under the curve of the ROC metric (numpy float)
    """
    # 1. Create Classifier model
    model = get_model(model_name, params)

    # 2. Train the model using the training sets 
    model.fit(train_data, train_labels)

    # 3. Predict the response for test dataset
    y_pred = model.predict(test_data)
   
    # 4. Model Accuracy, how often is the classifier correct? You may use metrics.accuracy_score(...)
    acc = metrics.accuracy_score(y_pred, test_labels, normalize=True)

    # 5. Calculate the confusion matrix by using the completed the function above 
    conf_mat = conf_matrix(y_pred, test_labels, num_class)

    # 6. Compute the AUROC score. You may use metrics.roc_auc_score(...)
    probs = model.predict_proba(test_data)
    auc_scor = metrics.roc_auc_score(test_labels, probs, multi_class='ovr')

    return acc, conf_mat, auc_scor


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = load_Dataset()
    num_class = 10
    
    model_name = "KNN"
    for k in range(1,6):
        print(str(k)+"-neighbors result:")
        params = k
        accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
        print("Accuracy:", accuracy)
        print("AUROC Score:", auc_score)
        print(confusion_matrix)
        print()
        
    model_name = "SVM"
    params = [1, True]
    accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
    print("SVM Result")
    print("Accuracy:", accuracy)
    print("AUROC Score:", auc_score)
    print(confusion_matrix)
    print()
    
    model_name = "MLP"
    params = [(15,10), 1, "relu"]
    accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
    print("MLP Result")
    print("Accuracy:", accuracy)
    print("AUROC Score:", auc_score)
    print(confusion_matrix)