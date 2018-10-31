import numpy as np
import cv2
import os
import sys

from preprocessing import Preprocessing
from hog import Hog
from sklearn.model_selection import StratifiedKFold

def get_data():
    '''
    Returns np array of labelled data
    '''
    labels = []
    x = []
    test = []
    paths = []
    pp = Preprocessing()

    # get list of data files
    flare_list = pp.get_training_data(pp.flare_path)
    blurry_list = pp.get_training_data(pp.blurry_path)
    good_list = pp.get_training_data(pp.good_path)

    paths = [pp.flare_path]*len(flare_list)
    paths = paths + [pp.blurry_path]*len(blurry_list)
    paths = paths + [pp.good_path]*len(good_list)

    test = flare_list + blurry_list + good_list

    # label flare files as 1 and add to training set
    append_sets(flare_list, pp.flare_path, x, labels, 1)
    # label blurry files as 2 and add to training set
    append_sets(blurry_list, pp.blurry_path, x, labels, 2)
    # label good files as 3 and add to training set
    append_sets(good_list, pp.good_path, x, labels, 3)



    return np.float32(x), np.array(labels, dtype=np.int32), test, paths

def append_sets(dataset, path, training, labels, l):
    '''
    appends labels to labels list for an associated image
    appends histograms to training for an associated image
    '''
    h = Hog()
    for file in dataset:
        img = read_img(path, file)
        if img is None:
            return labels
        # appends histogram of oriented gradients
        # to training list
        r_img=cv2.resize(img,(400,300))
        hist = h.hog(r_img)
        training.append(hist)
        # appends appropriate label to labels list
        labels.append(l)

def read_img(path, file):
    '''
    wrapper function that checks if image is broken or not
    '''
    imrgb = None
    with open(os.path.join(path, file), 'rb') as f:
        check_chars = f.read()[-2:]
    if check_chars != b'\xff\xd9':
        #Not complete image
        return imrgb
    else:
        imrgb = cv2.imread(os.path.join(path, file), 1)

    return imrgb

def calculate_accuracy(prediction, correct, accuracy):
    if prediction == 1 and correct == "flare":
        accuracy = accuracy +1
    if prediction == 2 and correct == "blurry":
        accuracy = accuracy +1
    if prediction == 3 and correct == "good":
        accuracy = accuracy +1

    return accuracy



def main():
    training, labels, test, paths = get_data()
    test_arg = []
    h = Hog()

    kfold = 1
    svm_overall = 0
    knn_overall = 0
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(training, labels):
        X_train, X_test = training[train_index], training[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        test_set = []
        svm_sum = 0
        knn_sum = 0

        # SVM model
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setGamma(0.5)
        svm.setC(30)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train) 

        knn = cv2.ml.KNearest_create()
        knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

        path_files = [paths[i] for i in test_index]
        test_files = [test[i] for i in test_index]

#        print X_test
        for path, file in zip(path_files, test_files):
            img = read_img(path, file)
            if img is None:
                continue

            # gets the label of current image for accuracy calculation
            l = path.split('/')[2].split('-')[0]
            r_img=cv2.resize(img,(400,300))
            hist=h.hog(r_img)
            test_set.append(hist)
            test_data = np.float32(test_set)
            result = svm.predict(test_data)
            retval, knnresults, neigh_resp, dists = knn.findNearest(test_data, 3)
            prediction_svm =  int(result[1][-1][0])
            prediction_knn = int(knnresults[-1][0])
            svm_sum = calculate_accuracy(prediction_svm, l, svm_sum)
            knn_sum = calculate_accuracy(prediction_knn, l, knn_sum)

        print "---------------- k=", kfold, " ----------------"
        print "SVM predictions: " , result[1].ravel()
        print "KNN predictions: " , knnresults.ravel()
        print "Actual labels for test set:" , labels[test_index]
        svm_acc = float(svm_sum) / float(len(result[1]))
        knn_acc = float(knn_sum) / float(len(knnresults.ravel()))
        print "Accuracy of svm: ", svm_acc
        print "Accuracy of knn: ", knn_acc

        svm_overall = svm_acc + svm_overall
        knn_overall = knn_acc + knn_overall
        kfold = kfold + 1

    svm_avg = float(svm_overall)/float(kfold-1)
    knn_avg = float(knn_overall)/float(kfold-1)

    print "Avg accuracy of svm after k=",kfold-1, ": ", svm_avg
    print "Avg accuracy of knn after k=",kfold-1, ": ", knn_avg


    if len(sys.argv) > 1:
        if len(sys.argv) == 3 and sys.argv[2] == "--show-results":
            print "TO DO"
        file = sys.argv[1].split('/')[-1]
        path = os.path.dirname(os.path.abspath(sys.argv[1]))
        arg_img = read_img(path, file)
        if arg_img is None:
            print "The image passed in invalid, bad path or corrupt file"

        # gets the label of current image for accuracy calculation
        r_img=cv2.resize(arg_img,(400,300))
        hist=h.hog(r_img)
        test_arg.append(hist)
        test_data = np.float32(test_arg)
        result = svm.predict(test_data)
        retval, knnresults, neigh_resp, dists = knn.findNearest(test_data, 3)
        prediction =  int(result[1][-1][0])
#       prediction_knn = int(knnresults[-1][0])
        print prediction
#        print prediction_knn
    else:
        print "Usage: ./detection.py ./path/to/image.jpg"


if __name__ == "__main__":
    main()
