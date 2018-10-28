import numpy as np
import cv2

from preprocessing import Preprocessing
from hog import Hog


def get_data():
    '''
    Returns np array of labelled data
    '''
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
    split_g = int(len(good_list) * 0.8)    #split ratio for good

    flist_train, flist_test = flare_list[:split_f], flare_list[split_f:]
    blist_train, blist_test = blurry_list[:split_b], blurry_list[split_b:]
    glist_train, glist_test = good_list[:split_g], flare_list[split_g:]

    paths = [pp.flare_path]*len(flist_test)
    paths = paths + [pp.blurry_path]*len(blist_test)
    paths = paths + [pp.good_path]*len(glist_test)

    test = flist_test + blist_test + glist_test

    # label flare files as 1 and add to training set
    append_sets(flist_train, pp.flare_path, x, y, 1)
    # label blurry files as 2 and add to training set
    append_sets(blist_train, pp.blurry_path, x, y, 2)
    # label good files as 3 and add to training set
    append_sets(glist_train, pp.good_path, x, y, 3)

    return np.float32(x), np.asarray(y, dtype=int), test, paths

def append_sets(dataset, path, training, labels, l):
    '''
    appends labels to labels list for an associated image
    appends histograms to training for an associated image
    '''
    h = Hog()
    for file in dataset:
        img = cv2.imread(path + file)
        # appends histogram of oriented gradients
        # to training list
        r_img=cv2.resize(img,(400,300))
        hist = h.hog(r_img)
        training.append(hist)
        # appends appropriate label to labels list
        labels.append(l)


def main():
    training, labels, test, paths = get_data()
    test_set = []
    h = Hog()

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm_params = dict( kernel_type = cv2.SVM_LINEAR,
     #               svm_type = cv2.SVM_C_SVC,
     #               C=2.67, gamma=5.383 )

    print type(labels)
    # Train the SVM:
    svm.train(training, cv2.ml.ROW_SAMPLE, labels) # 10-fold validation

    print "OpenCV version :  {0}".format(cv2.__version__)
    # Store it by using OpenCV functions:
    svm.save("./svm_data.dat")

    # Now create a new SVM & load the model:
    predictor = cv2.ml.SVM_create()
    predictor.setType(cv2.ml.SVM_C_SVC)
    predictor.setKernel(cv2.ml.SVM_LINEAR)

    predictor.load("./svm_data.dat")

    for path, file in zip(paths, test):
        img = cv2.imread(path+file)
        r_img=cv2.resize(img,(400,300))
        hist=h.hog(r_img)
        test_set.append(hist)
        test_data = np.float32(test_set)
        result = predictor.predict(testData)
        print result
    # Predict with predictor:
#    predictor.predict(, dtype=np.float32))
#

if __name__ == "__main__":
    main()
