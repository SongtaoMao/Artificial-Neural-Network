import pandas as pd
from sklearn.utils import shuffle

def normalize(file_dir,cols):
    data = pd.read_csv(file_dir)
    data[cols] = (data[cols] - data[cols].min()) / (data[cols].max() - data[cols].min())
    data.to_csv(file_dir.split('.')[0] + '_normalized.csv')

if __name__ == '__main__':
    filename = 'DATA_training.csv'
    cols = ['x', 'y', 'v']
    data = pd.read_csv('Data/'+filename)
    # data.drop_duplicates(cols, inplace=True)
    # data = shuffle(data)
    # data.to_csv('Data/'+filename.split(".")[0] + '_normalized.csv')
    normalize('Data/'+filename, cols)