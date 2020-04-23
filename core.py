import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import defaultdict
import math

class colors:
    blue = '\033[94m'
    green = '\033[92m'
    warn = '\033[93m'
    red = '\033[91m'
    endc = '\033[0m'

def get_nan(x):
    nan = 0
    for i in x:
        if np.isnan(i):
            nan += 1
    return nan

def count_(x):
    ct = 0
    try:    
        for i in x:
            if not np.isnan(i):
                ct += 1
    except:
        return len(x)
    return ct

def mean_(x):
    mean = 0
    ct = 0
    for i in x:
        if not np.isnan(i):
            mean += i
            ct += 1
    return mean / ct

def std_(x):
    std = 0
    mean = mean_(x)
    ct = 0
    for i in x:
        if not np.isnan(i):
            std += (i - mean) ** 2
            ct += 1
    return (1 / ct * std) ** 0.5

def min_(x):
    min_ = x[0]
    for i in x:
        if i < min_:
            min_ = i 
    return min_

def max_(x):
    max_ = x[0]
    for i in x:
        if i > max_:
            max_ = i 
    return max_

def  quartile_(x, p):
    x = x.dropna()
    x = x.sort_values()
    x = x.to_numpy() 
    k = (len(x) - 1) * (p / 100)
    f = np.floor(k)
    c = np.ceil(k)
    if f == c:
        return x[int(k)]
    d0 = x[int(f)] * (c - k)
    d1 = x[int(c)] * (k - f)
    return d0 + d1

def get_csv():
    if len(sys.argv) > 2 or len(sys.argv) < 2:
        print(colors.red + "The program should receive one argument.")
        exit()
    arg = sys.argv[1]
    try:
        data = pd.read_csv(arg)
    except:
        print(colors.red + "Enter a valide path")
        exit()
    return data

def g_marks_by_houses(data, feature):
    a = []
    b = []
    c = []
    d = []
    i = 0
    feat = data[feature]
    for v in data['Hogwarts House']:
        if v == "Gryffindor":
            a.append(feat[i])
        elif v == "Hufflepuff":
            b.append(feat[i])
        elif v == "Ravenclaw":
            c.append(feat[i])
        elif v == "Slytherin":
            d.append(feat[i])
        i += 1    
    return [a, b, c, d]

