#!/c/Python27/python

import numpy as np
import cv2
import os
import sys

from preprocessing import Preprocessing
from hog import Hog
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

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
    append_sets(good_list, pp.good_path, x, labels, 0)

    return np.float32(x), np.array(labels, dtype=np.int32), test, paths

def get_split_data():
    y = []
    x = []
    test = []
    paths = []
    pp = Preprocessing()

    # get list of data files
    flare_list = pp.get_training_data(pp.flare_path)
    blurry_list = pp.get_training_data(pp.blurry_path)
    good_list = pp.get_training_data(pp.good_path)

    #seperate into training and test sets 80:20
    # different ratios in the case we are using different data sets
    split_f = int(len(flare_list) * 0.8)    #split ratio for flares
    split_b = int(len(blurry_list) * 0.8)    #split ratio for blurry
    split_g = int(len(good_list) * 0.8) #split ratio for good

    flist_train, flist_test = flare_list[:split_f], flare_list[split_f:]
    blist_train, blist_test = blurry_list[:split_b], blurry_list[split_b:]
    glist_train, glist_test = good_list[:split_g], good_list[split_g:]

    paths = [pp.flare_path]*len(flist_test)
    paths = paths + [pp.blurry_path]*len(blist_test)
    paths = paths + [pp.good_path]*len(glist_test)

    test = flist_test + blist_test + glist_test

    # label flare files as 1 and add to training set
    append_sets(flist_train, pp.flare_path, x, y, 1)
    # label blurry files as 2 and add to training set
    append_sets(blist_train, pp.blurry_path, x, y, 2)
    # label good files as 3 and add to training set
    append_sets(glist_train, pp.good_path, x, y, 0)

    return np.float32(x), np.array(y, dtype=np.int32), test, paths



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
    i.e. corrupt file
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

def print_cv_results(svm,h):
    training, labels, test, paths = get_data()

    kfold = 1
    svm_overall = 0
    knn_overall = 0
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(training, labels):
        X_train, X_test = training[train_index], training[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        test_set = []

        svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train) 

        knn = cv2.ml.KNearest_create()
        knn.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

        path_files = [paths[i] for i in test_index]
        test_files = [test[i] for i in test_index]
        corrupt_index = 0 # tracks index of corrupt files
        for path, file in zip(path_files, test_files):
            img = read_img(path, file)
            # deletes corrupt file
            if img is None:
                y_test = np.delete(y_test, corrupt_index-1)
                corrupt_index = corrupt_index
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
            corrupt_index = corrupt_index+1

        r_svm = map(int,result[1].ravel())
        r_knn = map(int,knnresults.ravel())
        svm_acc = metrics.accuracy_score(y_test, r_svm)
        knn_acc = metrics.accuracy_score(y_test, r_knn)
#        count_svm = np.bincount(r_svm)
 #       count_knn = np.bincount(r_knn)
        unique_svm, num_svm = np.unique(result[1].ravel(), return_counts=True)
        unique_knn, num_knn = np.unique(knnresults.ravel(), return_counts=True)

        print "---------------- k=", kfold, " ----------------"
        print "SVM predictions: " , result[1].ravel()
        print "KNN predictions: " , knnresults.ravel()
        print "Actual labels for test set:" , y_test
        print "Accuracy of svm: ", svm_acc
        print "Accuracy of knn: ", knn_acc
        print "Null accuracy of svm: ", float(max(num_svm))/float(len(result[1]))
        print "Null accuracy of knn: ", float(max(num_knn))/float(len(result[1]))
        #print "Confusion matrix: ", metrics.confusion_matrix(y_test, r_svm)

        svm_overall = svm_acc + svm_overall
        knn_overall = knn_acc + knn_overall
        kfold = kfold + 1

    svm_avg = float(svm_overall)/float(kfold-1)
    knn_avg = float(knn_overall)/float(kfold-1)

    print "Avg accuracy of svm after k=",kfold-1, ": ", svm_avg
    print "Avg accuracy of knn after k=",kfold-1, ": ", knn_avg




def main():
    test_arg = []
    h = Hog()

    # SVM model
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_LINEAR)

    # create flags for arguments
    show_results = False
    raw_prediction = False
    for arg in sys.argv:
        # if true print cross validation results
        if arg == "--show-results":
            show_results = True
        # if true print 0 for good, 1 for flare, 2 for blurry
        if arg == "--raw-prediction":
            raw_prediction = True

    image_file = Path(sys.argv[1])
    # if arguments are passed correctly train
    if not image_file.is_file():
        print "File does not exist please check file path"
    if len(sys.argv) == 1:
        print "Usage: ./detection.py </path/to/img/file>"
        print "Use the flag --show-results to print cross validation results"
    elif len(sys.argv) > 1 and image_file.is_file():
        training, labels, test, paths = get_split_data()
        svm.train(training, cv2.ml.ROW_SAMPLE, labels)

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
        #retval, knnresults, neigh_resp, dists = knn.findNearest(test_data, 3)
        # if blurry data still print 0 for testing purposes to show faulty image
        prediction =  int(result[1][-1][0])
        if raw_prediction is False:
            if prediction == 2:
                print "1"
            else:                       # else if prediction is good or flare print appropriate label
                print prediction
        else:
            print prediction
#       prediction_knn = int(knnresults[-1][0])
#        print prediction_knn

    if show_results is True:
        print_cv_results(svm,h) # prints cross validation results



if __name__ == "__main__":
    main()
