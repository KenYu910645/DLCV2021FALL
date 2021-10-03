import cv2
from sklearn.decomposition import PCA
import numpy as np
import os 
from shutil import rmtree
import math
from sklearn.neighbors import KNeighborsClassifier
import sklearn
# Reference
# SKlearn PCA 
# https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
# SKlearn NN neighbor
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# N_COMPONENTS = 360
N_TRAIN = 360
N_TEST = 40
H, W = (56, 46)
# Input directory
IMAGE_DIR = "./p1_data/"

# Output directory
EIGENFACE_DIR = "./eigenfaces/"
RESTORE_DIR = "./restore_image/"

# Clean eigenfaces directory
print("Clean directory : " + str(EIGENFACE_DIR))
rmtree(EIGENFACE_DIR, ignore_errors=True)
os.mkdir(EIGENFACE_DIR)
# Clean restore image directoy
print("Clean directory : " + str(RESTORE_DIR))
rmtree(RESTORE_DIR, ignore_errors=True)
os.mkdir(RESTORE_DIR)

# Load training images
training_images = []
with open('training.txt', 'r') as f:
    for i in f.readlines():
        training_images.append(cv2.imread(IMAGE_DIR + i.split()[0], cv2.IMREAD_GRAYSCALE).flatten())

# Load testing images
testing_images = []
with open('testing.txt', 'r') as f:
    for i in f.readlines():
        testing_images.append(cv2.imread(IMAGE_DIR + i.split()[0], cv2.IMREAD_GRAYSCALE).flatten())

# Fit PCA
pca = PCA(n_components=N_TRAIN,
          svd_solver='randomized',
          whiten=True).fit(np.array(training_images))

######################
### First Question ###
######################
# Get mean face
mean_face = np.mean(np.array(training_images), axis=0)
cv2.imwrite(EIGENFACE_DIR + "mean_face.png", np.reshape(mean_face, (H, W)))

# Get eigenfaces
eigenfaces = pca.components_.reshape((N_TRAIN, H, W))
for i, face in enumerate(eigenfaces):
    face_norm = np.zeros(eigenfaces[0].shape, dtype=np.float32)
    cv2.normalize(face, face_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(EIGENFACE_DIR + str(i) + ".png", face_norm)

##############################
### Second, Third Question ###
##############################
IMG_NUM = 63 # 7*9
pca_result = pca.transform(np.array([training_images[IMG_NUM]]))
for i in [3, 50 ,170, 240, 345]:
    # Restore image from cleaned PCA
    pca_clean = pca_result.copy()
    pca_clean[0][i:] = 0
    pca_restore = pca.inverse_transform(np.array([pca_clean]))
    # Calculate MSE
    mse = (np.square(training_images[IMG_NUM] - pca_restore)).mean(axis=None)
    print("image_" + str(i) + " MSE = " + str(mse))
    # Output image
    cv2.imwrite(RESTORE_DIR + "image_" + str(i) + ".png", np.reshape(pca_restore, (H, W)))

##############################
### Fourth, Fifth Question ###
##############################
# transform testing/training images by PCA
train_pca = pca.transform(np.array(training_images))
test_pca = pca.transform(np.array(testing_images))

# Get training pca coordinate
for n in [3, 50, 170]:
    # Prune train_pca, remove unwanted cooridinate
    train_pca_prune = np.zeros((N_TRAIN, n))
    train_label = np.zeros((N_TRAIN))
    for i in range(N_TRAIN):
        train_pca_prune[i][:] = train_pca[i][:n]
        train_label[i] = math.floor(i/9.0) + 1

    # Shuffle training dataset
    train_pca_prune, train_label = sklearn.utils.shuffle(train_pca_prune, train_label)

    # N nearest neighborhood
    for k in [1, 3, 5]:
        sum_recog = 0.0
        for fold in [0, 1, 2]: # Cross validation
            # Get val_fold
            val_fold   = train_pca_prune[fold*120:(fold+1)*120].copy()
            val_label_fold = train_label[fold*120:(fold+1)*120].copy()
            # Get train_fold
            train_fold = []
            train_label_fold = []
            for i in [0, 1, 2]:
                if fold != i: # Goes to train set
                    train_fold.extend(train_pca_prune[i*120:(i+1)*120])
                    train_label_fold.extend(train_label[i*120:(i+1)*120])
            train_fold = np.array(train_fold)
            train_label_fold = np.array(train_label_fold)

            # Fit NN neighborhood
            nn_neigh = KNeighborsClassifier(n_neighbors=k)
            nn_neigh.fit(train_fold, train_label_fold)
            
            # predict validation images
            val_pred = nn_neigh.predict(val_fold)

            # Calculate recognition rate of validation result 
            num_correct = 0
            for i, result in enumerate(val_pred):
                if val_label_fold[i] == result:
                    num_correct += 1
            sum_recog += num_correct/120.0

            # Print out results
            print('============= (n, k ,fold) = ' + str((n ,k , fold))+ ' =================')
            print("Recognition rate on val = " + str(round(num_correct/120.0, 4)))
            print("Number of correct prediction = " + str(num_correct))
        
        avg_recog = sum_recog/3.0
        print("********************************")
        print("avg_recog (n = " + str(n) + ", k = " + str(k) +") = " + str(avg_recog))
        print("********************************")

######################
### Fifth Question ###
######################
# n = 50, k = 1, Has the highest recongition rate on validation
n , k = (50 ,1)

# Get Training data
train_pca_prune = np.zeros((N_TRAIN, n))
train_label = np.zeros((N_TRAIN))
for i in range(N_TRAIN):
    train_pca_prune[i][:] = train_pca[i][:n]
    train_label[i] = math.floor(i/9.0) + 1

# Get testing data
test_pca_prune = np.zeros((N_TEST, n))
test_label = np.zeros((N_TEST))
for i in range(N_TEST):
    test_pca_prune[i][:] = test_pca[i][:n]
    test_label[i] = i+1

# Train NN neighborhood
nn_neigh = KNeighborsClassifier(n_neighbors=k)
nn_neigh.fit(test_pca_prune, test_label)

# prediction
test_pred = nn_neigh.predict(test_pca_prune)

# Evaluation
num_correct = 0.0
for i, result in enumerate(test_pred):
    if test_label[i] == result:
        num_correct += 1
print("Recongnition rate on testing set = " + str(num_correct/N_TEST))
