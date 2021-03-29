# Lib includes
import matplotlib.pyplot as plt
import os

# File includes
from read_pics import get_pics_from_file
from graph_pics import graph_pics

# Global Varibles
DATA_PATH = "../data/"

# Main
if __name__ == "__main__":
    ######### Pics ############

    # Graphing all Data
    ind_figure = 1
    for filename in os.listdir(DATA_PATH):
        print("\n=====\n{0}".format(filename))
        plt.figure(ind_figure)
        pics, info = get_pics_from_file("../data/" + filename)
        graph_pics(filename, info, pics)
        ind_figure += 1

    # Show graphs
    plt.show()
