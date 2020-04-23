import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import core as ft

def main():
    data = ft.get_csv()
    print(f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |{"Max":>12} |{"NaN":>10}')
    for key, value in data.items():
        print(f'{key:15.15}', end=" |")
        print(f'{ft.count_(value):>12.4f}',end=" |")
        try:
            print(f'{ft.mean_(value):>12.5f}',end=" |")
            print(f'{ft.std_(value):>12.5f}', end=" |")
            print(f'{ft.min_(value):>12.5f}', end=" |")
            print(f'{ft.quartile_(value, 25):>12.5f}',end=" |")
            print(f'{ft.quartile_(value, 50):>12.5f}',end=" |")
            print(f'{ft.quartile_(value, 75):>12.5f}',end=" |")
            print(f'{ft.max_(value):>12.5f}',end=" |")
            print(f'{ft.get_nan(value):>10.5f}')
        except:
            print(ft.colors.warn + f'{"Missing Datas":>60}'+ ft.colors.endc)
            continue

if __name__ == '__main__':
    main()