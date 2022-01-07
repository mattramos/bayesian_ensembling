#!/bin/env python

import xarray as xr
import glob
import matplotlib.pyplot as plt
import numpy as np
import gc
import os
import sys

args = sys.argv
files = args[1:]

def gmst(file):
    strs = file.split('/')
    strs[-1] = 'GMST_' + strs[-1]
    new_file_name = '/'.join(strs)
    
    if not os.path.exists(new_file_name):
        print(f'Calculating GMST for {file}')
        da = xr.open_dataset(file).tas
        
        # Identify name of latitude coordinate
        if 'latitude' in da.coords:
            lat = 'latitude'
            lon = 'longitude'
        elif 'lat' in da.coords:
            lat = 'lat'
            lon = 'lon'
        else:
            raise AttributeError
            
        weights = np.cos(np.deg2rad(da[lat]))
        weights.name = "weights"

        da_weighted = da.weighted(weights)
        da_gmst = da_weighted.mean((lon, lat))
        da_gmst.to_netcdf(new_file_name)
        
        del da
        del da_weighted
        gc.collect()
    else:
        print(f'GMST already calculated: {file}')
    
    return

for file in files:
    gmst(file)
