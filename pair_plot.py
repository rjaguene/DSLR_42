import core as ft
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def clean_data(data):
    df = pd.DataFrame(data).dropna()
    feat = df['Hogwarts House']
    df = df[data.columns[6:]]
    return df, feat

def main():
    data = ft.get_csv()
    try:
        df, feat = clean_data(data)
        feat = pd.Series(feat).astype('category').cat.codes.values
        sm = scatter_matrix(df, c=feat, alpha=0.2, figsize=(6, 6), diagonal='hist',label=feat)
        #rotate x, y labels
        [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
        [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
        #padding y label
        [s.get_yaxis().set_label_coords(-1,0.5) for s in sm.reshape(-1)]
        #del ticks
        [s.set_xticks(()) for s in sm.reshape(-1)]
        [s.set_yticks(()) for s in sm.reshape(-1)]
        plt.show()
    except:
        exit(print("File error"))
    
if __name__ == '__main__':
    main()