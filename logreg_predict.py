import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import core as ft

def scal(df):
    for i in range(len(df)):
        df[i] = (df[i] - df.mean()) / df.std()
    return df

def g_pred(df, thetas):
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    pred = []
    for i in range(len(df)):
        pred_tmp = []
        for a in range(0,4):
            pred_tmp.append(df[i].dot(thetas[a][0]))
        max_ = pred_tmp.index(max(pred_tmp))
        pred.append(houses[max_])
    return pred

def prep_data(df):
    df = np.array((df.iloc[:,6:]).ffill())
    np.apply_along_axis(scal, 0, df)
    df = np.insert(df, 0, 1, axis=1)
    return df

def main():
    try:
        thetas = np.load(sys.argv[2])
        arg = sys.argv[1]
        data = pd.read_csv(arg)
    except:
        exit("usage: logreg_predict.py [dataset_test.csv] [weights.npy]")
    df = prep_data(data)
    pred = g_pred(df, thetas)
    index = []
    for i in range(len(pred)):
        index.append(i)
    pred = pd.DataFrame(list(zip(index, pred)), columns =['Index','Hogwarts House'])
    pred.to_csv("houses.csv", index=False)
    print("predictions saved in current directory")

if __name__ == "__main__":
    main()
