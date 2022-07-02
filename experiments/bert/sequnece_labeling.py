"""
Saving original data set to flair format
"""

import pandas as pd
import config

def create_corpus():
    raw_data = pd.read_csv(fr"{config.DATA_PATH}\\IOB tagging.csv",
                           index_col=0)

    train_data = raw_data[raw_data['sample'] <= 440]
    test_data = raw_data[raw_data['sample'] >= 500]
    val_data = raw_data[(raw_data['sample'] > 440) &
                          (raw_data['sample'] < 500)]
    txt_train = ''
    txt_val = ''
    txt_test = ''
    # define columns
    columns = {0: 'text', 1: 'ner'}

    for sample in train_data['sample'].unique():

        data = train_data[train_data['sample']==sample]
        for i, row in data.iterrows():
            txt_train += f'{row["entity"]} {row["tag"]}\n'
        txt_train+='\n'


    for sample in val_data['sample'].unique():

        data = val_data[val_data['sample']==sample]
        for i, row in data.iterrows():
            txt_val += f'{row["entity"]} {row["tag"]}\n'
        txt_val+='\n'

    for sample in test_data['sample'].unique():

        data = test_data[test_data['sample']==sample]
        for i, row in data.iterrows():
            txt_test += f'{row["entity"]} {row["tag"]}\n'
        txt_test+='\n'

    for name, file in zip(('train', 'dev', 'test'),(txt_train, txt_val, txt_test)):
        with open(f"{config.DATA_PATH}\\{name}.txt", 'w', encoding='utf-8') as f:
            f.write(file)
            f.close()

if __name__ == '__main__':
    create_corpus()