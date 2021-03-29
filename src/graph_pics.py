import matplotlib.pyplot as plt


def graph_pics(name, info, pics):
    plt.plot(range(1, info["nb_pics"] + 1), pics[0], 'ko')
    plt.xlabel('numéro de pic')
    plt.ylabel('amplitude du pic')
    plt.title(name)
    plt.ylim(0, 1.5)
    plt.grid(b=True, which='both')
