# Lib includes
import matplotlib.pyplot as plt

# File includes
from read_pics import get_pics_from_file
from graph_pics import graph_pics

# Main
if __name__ == "__main__":
    ######### Pics ############

    # NO KEY
    plt.figure(1)
    pics_nokey, info_nokey = get_pics_from_file("../data/pics_NOKEY.bin")
    graph_pics("pics_NOKEY", info_nokey, pics_nokey)

    # PAD-0
    plt.figure(2)
    pics_pad0, info_pad0 = get_pics_from_file("../data/pics_0.bin")
    graph_pics("pics_pad0", info_pad0, pics_pad0)

    # Show graphs
    plt.show()
