import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from joblib import load
import glob


def get_all_h_param_comb_svm(params):
    h_param_comb = [{"gamma": g, "C": c} for g in params['gamma'] for c in params['C']]
    return h_param_comb

def get_all_h_param_comb_tree(params):
    h_param_comb = [{"max_depth": a, "min_samples_leaf": b,"max_features":c} \
                    for a in params['max_depth'] \
                    for b in params['min_samples_leaf'] \
                    for c in params['max_features']]
    return h_param_comb
    
def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

# other types of preprocessing
# - image : 8x8 : resize 16x16, 32x32, 4x4 : flatteing
# - normalize data: mean normalization: [x - mean(X)]
#                 - min-max normalization
# - smoothing the image: blur on the image


def data_viz(dataset):
    # PART: sanity check visualization of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


# PART: Sanity check of predictions
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def train_dev_test_split(data, label, train_frac, dev_frac,random_state):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True,random_state=random_state
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True,random_state=random_state
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)

        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("Found new best metric with :" + str(cur_h_params))
            print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params


def tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path):
    if type(clf) == svm.SVC:
        model_type = 'svm'
        best_model, best_metric, best_h_params = h_param_tuning(
            h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
        )
    if type(clf) == tree.DecisionTreeClassifier:
        model_type = "dt"
        best_model, best_metric, best_h_params = h_param_tuning(
            h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric
        )
    # save the best_model
    best_param_config = "_".join([h + "=" + str(best_h_params[h]) for h in best_h_params])

    best_model_name = model_type + "_" + best_param_config + ".joblib"
    if model_path == None:
        model_path = "./models/"+best_model_name
    dump(best_model, model_path)

    print("Best hyperparameters were:")
    print(best_h_params)

    print("Best Metric on Dev was:{}".format(best_metric))

    return model_path

def save_result(best_model_path,x_test,y_test,clf_name,random_state):
    clf = load(best_model_path)
    y_pred = clf.predict(x_test)
    report = classification_report(y_test, y_pred,output_dict=True)
    with open(f"./results/{clf_name}_{random_state}.txt","+w") as fobj:
        fobj.write(f"test accuracy : {report['accuracy']}\n")
        fobj.write(f"test macro-f1 : {report['macro avg']['f1-score']}\n")
        fobj.write(f"model save at {best_model_path}\n")


def find_best_model():
    max_f1 = 0
    file_path = ""
    files = glob.glob("./results/*.txt")
    for file in files:
        with open(file, "r") as fileObj:
            temp_list = fileObj.readlines()
            lines_2 = temp_list[1]
            lines_3 = temp_list[2]
            f1_score = lines_2.split(":")[1]

            f1_score = f1_score.replace("\n", "")
            if float(f1_score) > max_f1:
                max_f1 = float(f1_score)
                file_path = lines_3.split(":")[1]
    return file_path