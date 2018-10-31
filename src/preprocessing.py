import os

class Preprocessing(object):

    def __init__(self):
        self.flare_path = '../data/flare-data/'
        self.blurry_path = '../data/blurry-data/'
        self.good_path = '../data/good-data/'

    def get_training_data(self, path):
        '''
        Returns a list of files given a directory path
        '''
        return [f for f in os.listdir(path) if os.path.isfile(path+f)]



