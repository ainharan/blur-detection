from preprocessing import Preprocessing

def label_data():
    pp = Preprocessing() 
    flare_list = pp.get_training_data(pp.flare_path)

def main():
    label_data()

if __name__ == "__main__":
    main()
