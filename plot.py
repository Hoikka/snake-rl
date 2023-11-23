from utils import plot_logs
import matplotlib.pyplot as plt
import sys

version = "pytorch"

path = "model_logs/{:s}.csv".format(version)

plot_logs(path)
