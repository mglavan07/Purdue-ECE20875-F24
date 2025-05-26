
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

'''
IMPORTANT : Move the data folder inside the solution before running the solution.
Only one copy is kept under students folder for saving memory.  
'''


######################################################################
def load_Dataset():
    with open('./data/train_data.npy', 'rb') as f:
        filtered_trained_data=np.load(f,allow_pickle=True)
    with open('./data/test_data.npy', 'rb') as f:
        filtered_test_data=np.load(f,allow_pickle=True)
    with open('./data/train_labels.npy', 'rb') as f:
        filtered_trained_labels=np.load(f,allow_pickle=True)
    with open('./data/test_labels.npy', 'rb') as f:
        filtered_test_labels=np.load(f,allow_pickle=True)


    return filtered_trained_data, filtered_trained_labels, filtered_test_data, filtered_test_labels

######################################################################
if __name__ == "__main__":


    # filtering original MNIST to make it 5000 for training and 1000 for testing.
    train_data, train_labels, test_data, test_labels = load_Dataset()

    print(train_data.shape) # (5000, 784) 5000 images, 28x28 = 784 in dimension
    # display some random training images in a 28x28 grid
    num_plot = 7 # show me 49 images
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx].reshape(28,28),cmap="Greys")
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

    # You can do the same for test images.
    # You can even show more/less number of images in your plot.
    # Explore the images in the dataset and see what they look like.
