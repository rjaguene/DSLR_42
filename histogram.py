import core as ft
import matplotlib.pyplot as plt

def get_min_std(data):
    min_ = ft.std_(data['Index'])
    for key, value in data.items():
        try:
            tmp = ft.std_(value)
            if min_ > tmp:
                min_ = tmp
                feature = key
        except:
            continue
    return feature

def main():
    data = ft.get_csv()
    min_feat = get_min_std(data)
    plt.title(min_feat, fontsize=10)
    plt.hist(ft.g_marks_by_houses(data, min_feat), stacked=True, color=['orange', 'green','blue','red'])
    plt.xlabel('Marks')
    plt.ylabel('Number of student')
    plt.show()

if __name__ == "__main__":
    main()