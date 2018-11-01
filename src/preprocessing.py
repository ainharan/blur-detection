import os

class Preprocessing(object):

    def __init__(self):
        self.flare_path = '../training-data/flare-data/'
        self.blurry_path = '../training-data/blurry-data/'
        self.good_path = '../training-data/good-data/'

    def get_training_data(self, path):
        '''
        Returns a list of files given a directory path
        '''
        return [f for f in os.listdir(path) if os.path.isfile(path+f)]



