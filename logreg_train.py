import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import core as ft
import logreg_predict as lp
import argparse

class log_reg(object):
    def __init__(self, w=[], epoch=1000):
        self.epoch = epoch
        self.thetas = w
        self.alpha = 0.01
        self.lampda = 10**-2
        self.epsilon = 10**-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.args = 0
    
    def check_result(self, df, y, thetas, good=0, error=0):     
        pred = lp.g_pred(df, thetas)
        for i in range(len(pred)):
            if self.args.v:
                print(pred[i],f'{y[i]:10}', end=" ")
            if pred[i] == y[i]:
                if self.args.v:
                    print(f'{"👍":>5.5}') 
                good += 1
            else:
                if self.args.v:
                    print(f'{"❌":>5.5}') 
                error += 1
        print("Total :\n", ft.colors.green, good, ft.colors.endc, "good prediction")
        print(" ", ft.colors.red, error, ft.colors.endc, "errors")
        print("Succes rate :", (good * 100) / len(y), end="%\n")

    def scall(self,df):
        for i in range(len(df)):
            if self.args.ls:
                df[i] = (df[i] - min(df)) / (max(df) - min(df))
            elif self.args.log:
                df[i] = np.log(abs(df[i]))
            else:
                df[i] = ( df[i] - df.mean())  / df.std()
        return df

    def g_(self, z):
        return 1 / (1 + np.exp(-z))

    def plot(self):
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        if self.args.p:
            plt.show()

    #Regularized Logistic Regression one vs all 
    def gradient(self, df, y):
        for house in np.unique(y):
            cost = []
            y_copy = np.where(y == house, 1, 0)
            thetas = np.zeros(df.shape[1])
            for _ in range(self.epoch):
                z = df.dot(thetas)
                g = self.g_(z)
                gradient = (g - y_copy) / len(y)          
                thetas -= (self.alpha) * np.dot(df.T, gradient) + ((self.lampda / len(y)) * thetas)     
                j = (1 / len(y)) * (np.dot(-y_copy.T, np.log(g)) - np.dot((1 - y_copy).T, np.log(1 - g)))
                cost.append(j)
            plt.plot(cost, label=house)
            self.thetas.append((thetas, house))
        self.plot()
        return self.thetas

    #Stochastic gradient descent regularized
    def sgd(self, df, y):
        for house in np.unique(y):
            cost = []
            y_copy = np.where(y == house, 1, 0)
            thetas = np.zeros(df.shape[1])
            df_copy = df
            for _ in range(self.epoch):
                p = np.random.permutation(len(df))
                df_copy = df_copy[p]
                y_copy = y_copy[p]
                for i in range(len(df)):
                    z = np.dot(df_copy[i], thetas)
                    g = self.g_(z)
                    gradient = (g - y_copy[i]) / len(y)           
                    thetas -= (self.alpha) * np.dot(df_copy[i].T, gradient) + ((self.lampda / len(y)) * thetas)     
                    j = (1 / len(y)) * (np.dot(-y_copy[i].T, np.log(g)) - np.dot((1 - y_copy[i]).T, np.log(1 - g)))
                cost.append(j)
            plt.plot(cost, label=house)
            self.thetas.append((thetas, house))
        self.plot()
        return self.thetas

    #Stochastic gradient descent with adam optimizer
    def sgd_adam(self, df, y):
            for house in np.unique(y):
                cost = []
                y_copy = np.where(y == house, 1, 0)
                thetas = np.zeros(df.shape[1])
                df_copy = df
                for _ in range(self.epoch):
                    p = np.random.permutation(len(df))
                    df_copy = df_copy[p]
                    y_copy = y_copy[p]
                    for i in range(len(df)):
                        z = np.dot(df_copy[i], thetas)
                        g = self.g_(z)
                        j = (1 / len(y)) * (np.dot(-y_copy[i].T, np.log(g)) - np.dot((1 - y_copy[i]).T, np.log(1 - g)))
                        error = (g - y_copy[i]) / len(y)
                        gradient = np.dot(df_copy[i].T, error)
                        m = np.zeros(len(gradient))
                        v = np.zeros(len(gradient))  
                        for t in range(len(gradient)):
                            m[t] = self.beta1 * m[t] + (1. - self.beta1) * gradient[t]
                            v[t] = self.beta2 * v[t] + ((1. - self.beta2) * gradient[t] ** 2)

                            m_hat = m[t] / (1. - self.beta1 ** (t + 1))
                            v_hat = v[t] / (1. - self.beta2 ** (t + 1))         
                            thetas[t] -= self.alpha / (np.sqrt(v_hat) + self.epsilon) * m_hat  
                        cost.append(j)
                plt.plot(cost, label=house)
                self.thetas.append((thetas, house))
            self.plot()
            return self.thetas

    #clean, scall data and drop useless field
    def prep_data(self, data):
        if self.args.f:
            data = data.fillna(method='ffill')
        else:
            data = data.dropna()
        y = data['Hogwarts House']
        df = np.array((data.iloc[:,6:]))
        np.apply_along_axis(self.scall, 0, df)
        y = y.to_numpy()
        df = np.insert(df, 0, 1, axis=1)
        return df, y

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="input dataset path")
    parser.add_argument('-s',action="store_true", help="Stochastic gradient descent")
    parser.add_argument('-a',action="store_true", help="Stochastic gradient descent with adam optimizer")
    parser.add_argument('-c',action="store_true", help="Check results")
    parser.add_argument('-p',action="store_true", help="Plot cost function")
    parser.add_argument('-f',action="store_true", help="Forward fill method for NaN values. Default: DropNaN")
    parser.add_argument('-e', action='store', dest='e', type=int, default=100, help='Epochs Number. Default: 1000')
    parser.add_argument('-ls',action="store_true", help="Linear Scalling. Default : Z_score")
    parser.add_argument('-log',action="store_true", help="Log Scalling. Default : Z_score")
    parser.add_argument('-v',action="store_true", help="Show compare")
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    try:
        data = pd.read_csv(args.file)
        fit = log_reg(epoch=args.e)
        fit.args = args
        df, y = fit.prep_data(data)
        if args.s:
            thetas = fit.sgd(df, y)
        elif args.a:
            thetas = fit.sgd_adam(df, y)
        else:
            thetas = fit.gradient(df, y)
        if args.c:
            fit.check_result(df, y, thetas)
    except:
        exit("File error")
    np.save("weights", thetas)
    print("weights saved in current directory")
    
if __name__ == "__main__":
    main()
