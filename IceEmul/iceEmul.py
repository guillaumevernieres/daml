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

def scatterplot_ice(ax, var, varname, bounds=None, cmap='viridis'):
    if (bounds is None):
        scatter = ax.scatter(lon, lat, c=var,
                             cmap=cmap, s=.5, alpha=0.8, transform=ccrs.PlateCarree())
    else:
        scatter = ax.scatter(lon, lat, c=var,
                             cmap=cmap, s=.5, alpha=0.8, transform=ccrs.PlateCarree(),
                             vmin=bounds[0], vmax=bounds[1])
    ax.coastlines()
    ax.set_title(varname)
    plt.colorbar(scatter, ax=ax, orientation='horizontal', shrink=0.5, pad=0.02)

# Load the NetCDF file
file_path = "gdas.t00z.icef009.ffnn.nc"
ds = xr.open_dataset(file_path)

# Extract the variables
aice = ds["aice"].values
aice_ffnn = ds["aice_ffnn"].values
dcdt = ds["dcdt"].values
dcds = ds["dcds"].values
dcdsi = ds["dcdsi"].values
dcdhi = ds["dcdhi"].values
dcdhs = ds["dcdhs"].values
dcdtair = ds["dcdtair"].values
dcdtsfc = ds["dcdtsfc"].values
lon = ds["lon"].values
lat = ds["lat"].values

# plot stuff
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])

#projection = ccrs.SouthPolarStereo()
projection = ccrs.NorthPolarStereo()

# Target aice
ax1 = fig.add_subplot(gs[0, 0], projection=projection)
scatterplot_ice(ax1, aice, "$aice$", bounds=[0, 1])

# Target aice
ax2 = fig.add_subplot(gs[0, 1], projection=projection)
scatterplot_ice(ax2, aice_ffnn, "$aice_{ffnn}$", bounds=[0, 1])

# Un-balanced aice
#ax2 = fig.add_subplot(gs[0, 2], projection=projection)
#scatterplot_ice(ax2, aice - aice_ffnn, "$aice_{u}$", bounds=[-0.1, 0.1], cmap='bwr')

# dcdsst
ax3 = fig.add_subplot(gs[1, 0], projection=projection)
scatterplot_ice(ax3, dcdt, "$dc/dT_{ocean}$", cmap='RdBu', bounds=[-0.3, 0.3])

# dcds
ax4 = fig.add_subplot(gs[1, 1], projection=projection)
scatterplot_ice(ax4, dcds, "$dc/dS_{ocean}$", cmap='RdBu', bounds=[-0.02, 0.02])

# dcdsi
ax = fig.add_subplot(gs[0, 2], projection=projection)
scatterplot_ice(ax, dcdsi, "$dc/dS_{ice}$", cmap='RdBu', bounds=[-0.05, 0.05])

# dcdhi
ax5 = fig.add_subplot(gs[1, 2], projection=projection)
scatterplot_ice(ax5, dcdhi, "$dc/dh_{ice}$", cmap='RdBu', bounds=[-3, 3])

# dcdhs
ax6 = fig.add_subplot(gs[2, 0], projection=projection)
scatterplot_ice(ax6, dcdhs, "$dc/dh_{snow}$", cmap='RdBu', bounds=[-3, 3])

# dcdtsfc
ax = fig.add_subplot(gs[2, 1], projection=projection)
scatterplot_ice(ax, dcdtsfc, "$dc/dT_{sfc}$", cmap='RdBu', bounds=[-0.1, 0.1])

# dcdtair
ax = fig.add_subplot(gs[2, 2], projection=projection)
scatterplot_ice(ax, dcdtair, "$dc/dT_{air}$", cmap='RdBu', bounds=[-0.05, 0.05])

plt.tight_layout()
plt.show()

# Calculate RMSE and correlation coefficient
rmse = np.sqrt(mean_squared_error(aice, aice_ffnn))
corr_coef, _ = pearsonr(aice.flatten(), aice_ffnn.flatten())

df = pd.DataFrame({"aice": aice, "aice_ffnn": aice_ffnn})

# Create a density plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 8))
resolution=15
sns.kdeplot(data=df, x="aice", y="aice_ffnn", cmap=cm.get_cmap("jet"), fill=True,
            common_norm=False, n_levels=resolution, common_grid=False)
plt.xlabel("aice")
plt.ylabel("aice_ffnn")
plt.title(f"aice vs aice_ffnn\nRMSE: {rmse:.4f}, Correlation Coefficient: {corr_coef:.4f}")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
