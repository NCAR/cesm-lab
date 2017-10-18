#! /usr/bin/env python
import xarray as xr

ds = xr.open_dataset('/glade/p/work/mclong/regrid/62_level/region_mask.nc',
                  decode_coords = False,
                  decode_times = False)

mask = (ds.REGION_MASK == -14)

ds.REGION_MASK.values[mask] = 0

ds.to_netcdf('/glade/p/work/mclong/regrid/62_level/region_mask_no_caspian.nc')
