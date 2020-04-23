import core as ft
import matplotlib.pyplot as plt


def main():
    data = ft.get_csv()
   
    plt.xlabel('Defense Against the Dark Arts')
    plt.ylabel('Astronomy')
    
    feat_1 = ft.g_marks_by_houses(data,'Defense Against the Dark Arts')
    feat_2 = ft.g_marks_by_houses(data,'Astronomy')

    plt.scatter(feat_1[0],feat_2[0],c='red')
    plt.scatter(feat_1[1],feat_2[1],c='blue')
    plt.scatter(feat_1[2],feat_2[2],c='green')
    plt.scatter(feat_1[3],feat_2[3],c='yellow')
    plt.show()

if __name__ == "__main__":
    main()