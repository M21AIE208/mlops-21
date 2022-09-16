# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import resize
#model parameters
param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
 ]



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
print(f"current Image size is {digits.images[0].shape}")
n_samples = len(digits.images)

img_array=[]
for images in digits.images:
    image_resized = resize(images, (32,32),
                           anti_aliasing=True)
    img_array.append(image_resized)
img_array=np.array(img_array)
data = img_array.reshape((n_samples, -1))
print(f"Scaled Image size is {img_array[0].shape}")

#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
dev_test_frac = 1-train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True
)


#PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()


best_model_accuracy  = 0

for c_ in param_grid[0]["C"]:
    for gamma_ in param_grid[0]["gamma"]:
        #PART: setting up hyperparameter
        hyper_params = {'gamma':gamma_,'C':c_}
        clf.set_params(**hyper_params)


        #PART: Train model
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        #PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)

#         #PART: Sanity check of predictions
#         _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#         for ax, image, prediction in zip(axes, X_test, predicted):
#             ax.set_axis_off()
#             ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#             ax.set_title(f"Prediction: {prediction}")
        
        model_accuracy = metrics.accuracy_score(y_test, predicted)
        if model_accuracy>best_model_accuracy:
            best_model_accuracy = model_accuracy
            best_param = {"gamma":gamma_,"C":c_}

        #PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf} with Gamma {gamma_} and C {c_} is :\n"
            f"{metrics.accuracy_score(y_test, predicted)}\n"
        )

print(f"best model accuracy is {best_model_accuracy} with parameters {best_param}")