import pandas as pd
import numpy as np

if __name__ == '__main__':
    filename = 'DATA_training.csv'
    df_train = pd.read_csv('data/'+filename)
    num = df_train.shape[0]  # 样本数
    columns = df_train.shape[1]

    # 数据测点选择
    for i in range(0, num - 1):
        P1 = df_train.loc[i, 'P1']
        T1 = df_train.loc[i, 'T1']
        if not(P1 == 94400 and T1 == 288):
            df_train = df_train.drop(index=i)

    # 空值删除
    for indexs in df_train.index:
        row = df_train.loc[indexs].values
        for i in range(0, len(row)):
            print(type(row[i]))
            if row[i] == 'BAD' or row[i] == 'error':
                df_train = df_train.drop(index=indexs)
                break
    print(df_train)


    df_train = df_train.drop_duplicates(subset=['T2', 'P2', 'T3', 'P3', 'T34', 'P34', 'T4'], keep='first')  # 删除重复
    print(df_train)

    df_train.to_csv('train')


