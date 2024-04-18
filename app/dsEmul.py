import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import pandas as pd
from IPython.display import display, clear_output

class Salt:
    def __init__(self, filename, woa_filename='/home/gvernier/data/woa/woa_ts.nc'):
        self.filename = filename
        ds = xr.open_dataset(filename)
        self.temp = ds["temp"].values
        self.salt = ds["salt"].values
        self.salt_truth = ds["salt_truth"].values

        indices=[0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33,
        36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66]

        dwoa = xr.open_dataset(woa_filename, decode_times=False)
        self.depth = dwoa["depth"].values[indices]
        print(self.depth)

    def plot(self):
#        fig = plt.figure(figsize=(8, 8))
#        gs = GridSpec(1, 2)
#        ax1 = fig.add_subplot(gs[0, 0])
#        ax1.plot(np.mean(np.abs(self.salt-self.salt_truth), 0), -self.depth, 'k')
#        ax1.grid(True)
#
#        ax2 = fig.add_subplot(gs[0, 1])
#        ax2.plot(np.mean(self.salt, 0), -self.depth[39], 'r')
#        ax2.plot(np.mean(self.salt_truth, 0), -self.depth[39], 'ob')
#        ax2.grid(True)
#        plt.show()
#
        #plt.ion()  # Turn on interactive mode

        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(1, 2)
        #ax1 = fig.add_subplot(gs[0, 0])
        #ax2 = fig.add_subplot(gs[0, 1])

        print(self.salt)
        plt.plot(self.salt, self.salt_truth, '.')
        plt.plot(self.salt_truth, self.salt_truth, '.r', alpha=0.1)
        plt.show()

#        while True:
#            #fig = plt.figure(figsize=(8, 8))
#            #gs = GridSpec(1, 2)
#            #ax1 = fig.add_subplot(gs[0, 0])
#            #ax2 = fig.add_subplot(gs[0, 1])
#
#            n = int(input("profile number: "))
#            ax1.plot(self.salt[n], -self.depth[13], 'or', alpha=1.)
#            ax1.plot(self.salt_truth[n], -self.depth[13], 'ob', alpha=1.)
#            ax1.grid(True)
#            ax2.plot(self.temp[n,:], -self.depth, 'k', alpha=1.)
#            ax2.grid(True)
#            #plt.draw()
#
#            display(plt.gcf())  # Display the current figure
#            clear_output(wait=True)
#
#            #plt.show(block=False)
#        plt.ioff()
salt = Salt(filename='salt.cffnn.nc')
salt.plot()
