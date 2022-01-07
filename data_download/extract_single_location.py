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

def sing_loc(file):
    strs = file.split('/')
    strs[-1] = 'SingLoc_' + strs[-1]
    new_file_name = '/'.join(strs)
    
    if not os.path.exists(new_file_name):
        print(f'Calculating GMST for {file}')
        da = xr.open_dataset(file).tas
        
        # Identify name of latitude coordinate
        if 'latitude' in da.coords:
            da = da.rename({'latitude' : 'lat', 'longitude': 'lon'})
        elif 'lat' in da.coords:
            pass
        else:
            raise AttributeError
            
        da_singloc = da.sel(lat=52.5, lon=0, method='nearest')
        da_singloc.to_netcdf(new_file_name)
        
        del da
        gc.collect()
    else:
        print(f'Single location already calculated: {file}')
    
    return

for file in files:
    sing_loc(file)