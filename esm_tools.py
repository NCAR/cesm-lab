#! /usr/bin/env python
import os
import sys
import xarray as xr
import numpy as np
from netcdftime import utime
import time
import tempfile
import json

debug = True
nmol_to_Pmol = 1e-9 * 1e-15
nmols_to_Tmolyr = 1e-9 * 1e-12 * 86400. * 365.


tmpdir = os.environ['TMPDIR']

time_chunks = 5 * 12 # read data in 5 year chunks
xr_open_dataset = {
    'chunks':{'time' : time_chunks},
    'mask_and_scale' : True,
    'decode_times' : False,
    'decode_coords': False}

#--------------------------------------------------------------------------------
#---- class
#--------------------------------------------------------------------------------
class timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def json_cmd(kwargs_dict):
    return '\'{0}\''.format(json.dumps(kwargs_dict))


#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def pop_calc_zonal_mean(ds):
    '''
    compute zonal mean of POP field
    in lieau of wrapping klindsay's zon_avg program so as to operate on
    an `xarray` dataset: write to file, compute, read back.
    '''

    print('computing zonal mean')
    za = '/glade/u/home/klindsay/bin/za'
    rmask_file = '/glade/p/work/mclong/grids/PacAtlInd_REGION_MASK_gx1v6.nc'

    _,za_file_in = tempfile.mkstemp('.nc','pop_calc_zonal_mean')
    _,za_file_out = tempfile.mkstemp('.nc','pop_calc_zonal_mean')

    #-- write input dataset to file_in
    ds.to_netcdf(za_file_in,unlimited_dims='time')

    #-- run the za program
    stat = call([za,'-rmask_file',rmask_file,'-o',za_file_out,za_file_in])
    if stat != 0:
        print('za failed')
        sys.exit(1)

    #-- read the dataset
    ds = xr.open_dataset(za_file_out,**xr_open_dataset)
    ds.load()

    call(['rm','-f',za_file_out,za_file_in])

    return ds

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def pop_calc_global_mean(ds):
    print('computing global mean')

    need_vol = False
    for variable in ds:
        if ds[variable].dims == ('time','z_t','nlat','nlon'):
            need_vol = True
            break

    area = ds.TAREA.values[None,:,:]
    if need_vol:
        with timer('computing vol array'):
            vol_values = ds.TAREA.values[None,:,:] * ds.dz.values[:,None,None]
            vol = xr.DataArray(vol_values,dims=('z_t','nlat','nlon'))

    for variable in ds:
        if ds[variable].dims == ('time','z_t','nlat','nlon'):
            with timer('computing volume-weighted integral %s'%variable):
                attrs = ds[variable].attrs
                convert = 1.0
                new_units = ds[variable].attrs['units']+' cm^3'

                if variable == 'O2':
                    convert = nmol_to_Pmol
                    new_units = 'Pmol'
                elif any([variable == v for v in ['O2_CONSUMPTION','O2_PRODUCTION']]):
                    convert = nmols_to_Tmolyr
                    new_units = 'Tmol yr$^{-1}$'

                ds[variable] = (ds[variable] * vol * convert).sum(dim=['z_t','nlat','nlon'])
                ds[variable].attrs = attrs
                ds[variable].attrs['units'] = new_units

        elif ds[variable].dims == ('time','nlat','nlon'):
            with timer('computing area-weighted integral %s'%variable):
                attrs = ds[variable].attrs
                convert = 1.0
                new_units = ds[variable].attrs['units']+' cm^2'

                if variable == 'STF_O2':
                    convert = nmols_to_Tmolyr
                    new_units = 'Tmol yr$^{-1}$'

                ds[variable] = (ds[variable] * area * convert).sum(dim=['nlat','nlon'])
                ds[variable].attrs = attrs
                ds[variable].attrs['units'] = new_units

    return ds


#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def calc_ann_mean(ds):
    time_dimname = 'time'

    with timer('computing weights for annual means'):
        tb_name = ''
        if 'bounds' in ds[time_dimname].attrs:
            tb_name = ds[time_dimname].attrs['bounds']
            tb = ds[tb_name].values
            wgt = np.diff(tb[:,:],axis=1)[:,0]
        else:
            print('no time bound attribute found')
            wgt = np.ones(len(ds.time))

        #-- compute weights (assume 'year' is there)
        time_bnd = xr.DataArray(wgt,coords=[ds.time])
        dsw = xr.Dataset({'year':ds.year,'time_bnd':time_bnd})
        wgt = dsw.groupby('year')/dsw.groupby('year').sum()
        nyr = len(dsw.groupby('year').groups)

        #-- test that weights sum to 1.0 for each year
        np.testing.assert_allclose(
            wgt['time_bnd'].groupby('year').sum().values,
            np.ones(nyr))

    print('computing ann mean over %d years'%nyr)

    #-- compute annual mean
    time_vars = [k for k in ds.keys() if time_dimname in ds[k].dims]
    grid_vars = [k for k in ds.keys() if time_dimname not in ds[k].dims]
    dsg = ds.drop(time_vars)
    ds = ds.drop(grid_vars)

    # groupby.sum() does not seem to handle missing values correctly: yeilds 0 not nan
    # the groupby.mean() does return nans, so create a mask of valid values
    with timer('compute mask'):
        valid = ds.groupby('year').mean(dim='time').notnull()

    # drop of "year" required for viable output
    with timer('compute weighted mean'):
        ds = ds.drop(['month','yearfrac','year'])
        dso = (ds * wgt['time_bnd']).groupby('year').sum(dim='time')
        for v in ds:
            if v not in dso: continue
            if 'dtype' in ds[v].encoding:
                dso[v] = dso[v].astype(ds[v].encoding['dtype'])
            if ds[v].attrs:
                dso[v].attrs = ds[v].attrs

    #-- apply mask for valid values
    with timer('apply missing value mask'):
        dso = dso.where(valid)

    #-- fix the time coordindate
    with timer('fix the time coordinate and merge grid vars'):
        #-- make time coordinate
        time_coord = (ds[time_dimname] * wgt['time_bnd']).groupby('year').sum(dim='time')
        time_coord = time_coord.rename({'year':'time'})
        time_coord = time_coord.assign_coords(time=time_coord)
        time_coord.attrs = ds.time.attrs

        #-- detach and reattach 'year'; put time
        year = dso.year.rename({'year':'time'})
        year = year.assign_coords(time=time_coord)
        dso = dso.rename({'year':'time'})
        dso = dso.assign_coords(time=time_coord)
        dso['year'] = year


        #-- put the grid variables back
        dso = xr.merge((dso,dsg))

    return dso

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def pop_total_ocean_volume(ds):
    vol_values = ds.TAREA.values[None,:,:] * ds.dz.values[:,None,None]
    for j in range(len(ds.nlat)):
        for i in range(len(ds.nlon)):
            k = ds.KMT.values[j,i].astype(int)
            vol_values[k:,j,i] = 0.
    return vol_values.sum()


#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------
def apply_drift_correction(variable,ds_ctrl,ds_list,file_out_list=[]):

    if file_out_list:
        if len(file_out_list) != len(ds_list):
            print('File out list does not match ds_list')
            sys.exit(1)

    #-- make time vector
    nt = len(ds_ctrl.time)
    x = np.arange(0,nt,1)

    #-- read data
    y = ds_ctrl[variable].values.reshape((nt,-1))

    # compute regression coeff
    beta = np.polyfit(x,y,1)

    #-- compute drift
    drift = (beta[0,:] * x[:,None]).reshape(ds_ctrl[variable].shape)

    for i,ds in enumerate(ds_list):
        ds_list[i][variable].values = ds[variable].values - drift

    if file_out_list:
        for i,f in enumerate(file_out_list):
            ds_list[i].to_netcdf(f,unlimited_dims='time')

    return ds_list

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------
def tseries_dataset(case_files, yr0,
                    file_out = '',
                    year_range = None,
                    transform_func = None):

    #-- loop over cases and get datasets
    if debug:
        print
        print('-'*80)
        print(case_files)

    #-- make files into xarray datasets
    dsi = []
    for f in case_files:
        if debug:
            print('opening %s'%f)
        dsii = xr.open_dataset(f,**xr_open_dataset)

        cdftime = utime(dsii.time.attrs['units'],
                        calendar=dsii.time.attrs['calendar'])


        if year_range is not None:
            if debug:
                print('apply year_range to %s:'%f)
                print(year_range)

            year = np.array([d.year+yr0
                             for d in cdftime.num2date(dsii.time-1./86400.)])
            if debug:
                print('years = %d - %d'%(year[0],year[-1]))

            nx = np.where((year_range[0] <= year) & (year <= year_range[-1]))[0]
            if len(nx) == 0:
                if debug:
                    print('no dates found in range')
                    print
                continue

            if debug:
                print('time index = %s'%str(slice(nx[0],nx[-1]+1)))
            dsii = dsii.isel(time=slice(nx[0],nx[-1]+1))

            if debug:
                print('subsetted dates:')
                print(cdftime.num2date(dsii.time[0]))
                print(cdftime.num2date(dsii.time[-1]))
                print

        dsii['year'] = dsii.time.copy()
        dsii.year.attrs = {}
        dsii.year.values = np.array([d.year+yr0
                                     for d in cdftime.num2date(dsii.time-1./86400.)])

        dsii['yearfrac'] = dsii.time.copy()
        dsii.yearfrac.attrs = {}
        dsii.yearfrac.values = np.array([d.year+yr0+d.day/365.
                                         for d in cdftime.num2date(dsii.time-1./86400.)])

        dsii['month'] = dsii.time.copy()
        dsii.month.attrs = {}
        dsii.month.values = np.array([d.month
                                      for d in cdftime.num2date(dsii.time-1./86400.)])

        if transform_func is not None:
            if debug:
                print('transforming data')
            if hasattr(transform_func,'__iter__'):
                for tfunc in transform_func:
                    dsii = tfunc(dsii)
            else:
                dsii = transform_func(dsii)

        dsi.append(dsii)

    #-- concatenate
    if len(dsi) > 1:
        dsi = xr.concat(dsi,dim='time',data_vars='minimal')
    elif len(dsi) > 0:
        dsi = dsi[0]

    if file_out and dsi:
        dsi.to_netcdf(file_out,unlimited_dims='time')

    return dsi

#----------------------------------------------------------------
#---- CLASS
#----------------------------------------------------------------
class hfile( object ):
    def __init__(self,
                 dirname = '',
                 prefix  = '',
                 ens     = '',
                 stream  = '',
                 op      = '',
                 varname     = '',
                 datestr = '',
                 ext = 'nc'):

        self.dirname = dirname
        self.prefix  = prefix
        self.ens     = ens
        self.stream  = stream
        self.op      = op
        self.varname = varname
        self.datestr = datestr
        self.ext     = ext

    def __str__(self):
        name_parts = []
        for n in ['prefix','ens','stream','op','varname','datestr','ext']:
            if self.__dict__[n]: name_parts.append(self.__dict__[n])
        return os.path.join(self.dirname,'.'.join(name_parts))

    def copy(self):
        import copy
        return copy.copy(self)

    def append(self,**kwargs):
        new = self.copy()
        for key,value in kwargs.items():
            if key in self.__dict__:
                new.__dict__[key] = '_'.join([self.__dict__[key],value])
            else:
                raise AttributeError('%s has no attribute %s'%
                                     (self.__class__.__name__,key))
        return new

    def prepend(self,**kwargs):
        new = self.copy()
        for key,value in kwargs.items():
            if key in self.__dict__:
                new.__dict__[key] = '_'.join([value,self.__dict__[key]])
            else:
                raise AttributeError('%s has no attribute %s'%
                                     (self.__class__.__name__,key))
        return new

    def update(self,**kwargs):
        new = self.copy()
        for key,value in kwargs.items():
            if key in self.__dict__:
                new.__dict__[key] = value
            else:
                raise AttributeError('%s has no attribute %s'%
                                     (self.__class__.__name__,key))
        return new

if __name__ == '__main__':
    import argparse
    import json

    #-- set defaults
    control_defaults = {
        'task': None,
        'kwargs': {}}

    #-- set up parser
    p = argparse.ArgumentParser(description='perform operations')
    p.add_argument('json_control',
                   default=control_defaults)

    p.add_argument('-f',dest='json_as_file',
                   action='store_true',default=False,
                   help='Interpret input as a file name')

    #-- parse args
    args = p.parse_args()
    if not args.json_as_file:
        control_in = json.loads(args.json_control)
    else:
        with open(args.json_control,'r') as fp:
            control_in = json.load(fp)

    control = control_defaults
    control.update(control_in)

    #-- consider required arguments:
    missing_req = False
    req = ['task','kwargs']
    for k in req:
        if control[k] is None or not control[k]:
            print('ERROR: missing %s'%k)
            missing_req = True
    if missing_req:
        print('stopping')
        sys.exit(1)

    #-- this is ugly
    task_function = eval(control['task'])
    if 'transform_func' in control['kwargs']:
        transform_func = control['kwargs']['transform_func']
        if hasattr(transform_func,'__iter__'):
            for i,func in enumerate(transform_func):
                transform_func[i] = eval(func)
        else:
            transform_func = eval(transform_func)

    print control
    task_function(**control['kwargs'])
