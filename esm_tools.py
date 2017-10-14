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

rmask_file = '/glade/p/work/mclong/grids/PacAtlInd_REGION_MASK_gx1v6.nc'

tmpdir = os.environ['TMPDIR']

#time_chunks = 5 * 12 # read data in 5 year chunks
#    'chunks' : time_chunks,
xr_open_dataset = {
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
        if self.name:
            print '[%s]' % self.name,
    def __exit__(self, type, value, traceback):
        print 'Elapsed: %s' % (time.time() - self.tstart)

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def json_cmd(kwargs_dict):
    return '\'{0}\''.format(json.dumps(kwargs_dict))


#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def pop_calc_zonal_mean(file_in,file_out):
    '''
    compute zonal mean of POP field
    in lieau of wrapping klindsay's zon_avg program so as to operate on
    an `xarray` dataset: write to file, compute, read back.
    '''
    from subprocess import call

    print('computing zonal mean')
    za = '/glade/u/home/klindsay/bin/za'

    #-- run the za program
    stat = call([za,'-rmask_file',rmask_file,'-o',file_out,file_in])
    if stat != 0:
        print('za failed')
        sys.exit(1)

    #-- read the dataset
    ds = xr.open_dataset(file_out,**xr_open_dataset)

    return ds

#------------------------------------------------------------
#-- function
#------------------------------------------------------------

def pop_calc_vertical_integral(ds):
    gridvar = [k for k in ds if 'time' not in ds[k].dims]
    timevar = [k for k in ds if 'time' in ds[k].dims]

    dsg = ds.drop(timevar)
    ds = ds.drop(gridvar)

    for variable in ds.keys():
        if not any(d in ds[variable].dims for d in ['z_t','z_t_150m']):
            continue

        attrs = ds[variable].attrs
        new_units = ds[variable].attrs['units']+' cm'

        if 'z_t' in ds[variable].dims:
            ds[variable] = (ds[variable] * dsg.dz).sum(dim='z_t')

        elif 'z_t_150m' in ds[variable].dims:
            ds[variable] = (ds[variable] * dsg.dz[0:15]).sum(dim='z_t_150m')

        ds[variable].attrs = attrs
        ds[variable].attrs['units'] = new_units

    ds = xr.merge((ds,dsg))

    return ds

#------------------------------------------------------------
#-- function
#------------------------------------------------------------

def pop_calc_global_mean(ds):
    ds = pop_calc_spatial_mean(ds,avg_over_dims=['z_t','nlat','nlon'])
    return ds

#------------------------------------------------------------
#-- function
#------------------------------------------------------------

def pop_calc_spatial_mean(ds,avg_over_dims=['z_t','nlat','nlon']):

    # TODO: add masking capability
    '''
    rmask = xr.open_dataset(rmask_file,**xr_open_dataset)

    if rmask.REGION_MASK.ndim == 2:
        region = rmask.REGION_MASK.where(rmask.REGION_MASK!=0.).pipe(np.unique)
        region = region[~np.isnan(region)]
        nrgn = len(region)
        mask3d = np.zeros((nrgn,)+rmask.REGION_MASK.shape)
    else:
        mask3d = rmask.REGION_MASK.values
        nrgn = mask3d.shape[0]
    '''

    vol_wgt = []
    area_wgt = []
    for variable in ds:
        if not all([d in ds[variable].dims for d in ['time','nlat','nlon']]):
            continue

        attrs = ds[variable].attrs

        avg_over_dims_v = [k for k in avg_over_dims if k in ds[variable].dims]

        #-- 3D vars
        if ds[variable].dims == ('time','z_t','nlat','nlon'):
            if not vol_wgt:
                dsv = pop_ocean_volume(ds)
                ds['vol_sum'] = dsv.VOL.sum(dim=avg_over_dims_v)
                ds.vol_sum.attrs['units'] = 'cm^3'
                vol_wgt = dsv.VOL / ds.vol_sum

            ds[variable] = (ds[variable] * vol_wgt).sum(dim=avg_over_dims_v)

        #-- 2D vars
        elif ds[variable].dims == ('time','nlat','nlon'):
            if not area_wgt:
                ds['area_sum'] = ds.TAREA.where(ds.KMT > 0).sum(avg_over_dims_v)
                ds.area_sum.attrs['units'] = 'cm^2'
                area_wgt = ds['TAREA'] / ds.area_sum
            ds[variable] = (ds[variable] * ds.TAREA).sum(dim=avg_over_dims_v)

        ds[variable].attrs = attrs

    return ds

#------------------------------------------------------------
#-- function
#------------------------------------------------------------

def pop_calc_area_mean(ds):
    ds = pop_calc_spatial_mean(ds,avg_over_dims=['nlat','nlon'])
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

def pop_ocean_volume(ds):
    dso = ds.copy().drop([k for k in ds
                          if 'time' in ds[k].dims])

    dso['VOL'] = ds.dz * ds.TAREA
    for j in range(len(ds.nlat)):
        for i in range(len(ds.nlon)):
            k = ds.KMT.values[j,i].astype(int)
            dso.VOL.values[k:,j,i] = 0.

    return dso

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------
def apply_drift_correction_ann(variable,file_ctrl,file_drift,
                               file_in_list,file_out_list):

    if len(file_out_list) != len(file_in_list):
        print('File out list does not match file_in_list')
        sys.exit(1)

    if all([os.path.exists(f) for f in file_out_list]):
        return

    #-- make time vector
    with timer('computing control drift'):

        if not os.path.exists(file_drift):
            #-- open dataset
            dsc = xr.open_dataset(file_ctrl,**xr_open_dataset)

            #-- generate dummy axis
            nt = len(dsc.time)
            x = np.arange(0,nt,1)

            #-- read data
            y = dsc[variable].values.reshape((nt,-1))

            # compute regression coeff
            beta = np.polyfit(x,y,1)

            #-- compute drift
            drift = (beta[0,:] * x[:,None]).reshape(dsc[variable].shape)
            dsc[variable+'_drift_corr'] = dsc[variable].copy()
            dsc[variable+'_drift_corr'].values = drift
            dsc.to_netcdf(file_drift,unlimited_dims='time')
        else:
            dsc = xr.open_dataset(file_drift,**xr_open_dataset)

        yearc = dsc.year.values

    for file_in,file_out in zip(file_in_list,file_out_list):
        with timer('drift correcting: %s'%file_in):
            if not os.path.exists(file_out):
                ds = xr.open_dataset(file_in,**xr_open_dataset)
                year = ds.year.values
                slc = slice(np.where(year[0]==yearc)[0][0],
                            np.where(year[-1]==yearc)[0][-1]+1)
                print slc
                print yearc[slc]

                #-- subtract the drift
                ds[variable].values = ds[variable].values - \
                    dsc[variable+'_drift_corr'].values[slc,:]
                ds.to_netcdf(file_out,unlimited_dims='time')

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def ensemble_mean_std(file_in_list,file_out_avg,file_out_std):
    dse = xr.open_mfdataset(file_in_list,
                            concat_dim='ens',
                            **xr_open_dataset)

    dseo = dse.mean(dim='ens')
    dseo.to_netcdf(file_out_avg,unlimited_dims='time')

    dseo = dse.std(dim='ens')
    dseo.to_netcdf(file_out_std,unlimited_dims='time')

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def transform_file(file_in,file_out,preprocess):
    ds = xr.open_dataset(file_in,**xr_open_dataset)

    if hasattr(preprocess,'__iter__'):
        for tfunc in preprocess:
            ds = tfunc(ds)
    else:
        dsi = preprocess(ds)

    ds.to_netcdf(file_out,unlimited_dims='time')


#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def debug_print(msg=''):
    if debug:
        print(msg)

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def interpret_time(ds,year_offset):
    cdftime = utime(ds.time.attrs['units'],
                    calendar=ds.time.attrs['calendar'])
    ds['year'] = ds.time.copy()
    ds.year.attrs = {}
    ds.year.values = np.array([d.year+year_offset
                               for d in cdftime.num2date(ds.time-1./86400.)])

    ds['yearfrac'] = ds.time.copy()
    ds.yearfrac.attrs = {}
    ds.yearfrac.values = np.array([d.year+year_offset+d.day/365.
                                   for d in cdftime.num2date(ds.time-1./86400.)])

    ds['month'] = ds.time.copy()
    ds.month.attrs = {}
    ds.month.values = np.array([d.month
                                for d in cdftime.num2date(ds.time-1./86400.)])
    return ds

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def select_by_year(ds,year_range):
    if year_range is None:
        return ds

    year = ds.year.values
    nx = np.where((year_range[0] <= year) & (year <= year_range[-1]))[0]
    if len(nx) == 0:
        return None

    ds = ds.isel(time=slice(nx[0],nx[-1]+1))
    return ds

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def require_variables(ds,req_var):
    missing_var_error = False
    for v in req_var:
        if v not in ds:
            print('Missing required variable: %s'%v)
            missing_var_error = True

    if missing_var_error:
        sys.exit(1)

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def pop_derive_var_OUR(ds):
    require_variables(ds,['AOU','IAGE'])

    ds['IAGE'] = ds.IAGE.where(ds.IAGE>0)
    ds['OUR'] = ds.AOU / ds.IAGE
    ds.OUR.attrs['units'] = ds.AOU.attrs['units']+'/'+ds.IAGE.attrs['units']
    ds.OUR.attrs['long_name'] = 'OUR'

    ds = ds.drop(['AOU','IAGE'])

    return ds

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def pop_derive_var_NPP(ds):

    require_variables(ds,['photoC_sp','photoC_diat','photoC_diaz'])

    ds['NPP'] = ds.photoC_sp + ds.photoC_diat + ds.photoC_diaz
    ds.NPP.attrs['units'] = ds.photoC_sp.attrs['units']
    ds.NPP.attrs['long_name'] = 'NPP'
    ds = ds.drop(['photoC_sp','photoC_diat','photoC_diaz'])

    return ds

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------

def open_tsdataset(paths,
                   year_offset = 0,
                   file_out = '',
                   year_range = None,
                   preprocess = [],
                   preprocess_kwargs = []):
    '''Open multiple timeseries files as a single dataset.

    Read and concatenate multiple files a single dataset.

    Parameters
    ----------
    paths : list or dict
       open and concatenate or, if a list of dictionaries
       open, merge, and then concatenate.
    year_offset : int, optional
       year to add to base year to align calendar.
    year_range : [int,int], optional
       pick only data in this range
    preprocess : callable or list of callable, optional
       transform dataset with this function
    preprocess_kwargs : dict
       arguments to pass to preprocess.

    Returns
    -------
    xarray.Datasets

    '''

    #-- add some default operations
    preprocess_def = [interpret_time,select_by_year]
    preprocess_kwargs_def = [{'year_offset':year_offset},
                             {'year_range':year_range}]
    if preprocess:
        if not isinstance(preprocess,list):
            preprocess = [preprocess]

        if preprocess_kwargs:
            if not isinstance(preprocess_kwargs,list):
                preprocess_kwargs = [preprocess_kwargs]
        else:
            preprocess_kwargs = [{}]*len(preprocess)

        if len(preprocess) != len(preprocess_kwargs):
            print('ERROR: len(preprocess) != len(preprocess_kwargs)')
            sys.exit(1)

        preprocess = preprocess_def+preprocess
        preprocess_kwargs = preprocess_kwargs_def+preprocess_kwargs

    #-- loop over paths
    dsi = []
    if isinstance(paths,dict):
        paths_iter = zip(*paths.values())
    else:
        paths_iter = paths

    for path_i in paths_iter:

        #-- if iterable, do a merge
        if hasattr(path_i,'__iter__'):
            dsii = {}
            for path_ii in path_i:
                dsiii = xr.open_dataset(path_ii,**xr_open_dataset)
                if not dsii:
                    dsii = dsiii
                else:
                    dsii = xr.merge((dsii,dsiii))
        else:
            dsii = xr.open_dataset(path_i,**xr_open_dataset)

        #-- apply preprocess functions
        for func,kwargs in zip(preprocess,preprocess_kwargs):
            dsii = func(dsii,**kwargs)
            if dsii is None: break

        if dsii is None: continue
        dsi.append(dsii)

    #-- concatenate
    if not dsi:
        return None
    elif len(dsi) > 1:
        dsi = xr.concat(dsi,dim='time',data_vars='minimal')
    else:
        dsi = dsi[0]

    if file_out and dsi:
        dsi.to_netcdf(file_out,unlimited_dims='time')

    return dsi

#----------------------------------------------------------------
#---- CLASS
#----------------------------------------------------------------
class hfile(object):
    '''A file name object with an underlying dictionary of parts.
    '''
    def __init__(self,**kwargs):
        '''Build file name from fields in OrderedDict.

        Kwargs:
        fields in the dictionary
        '''
        from collections import OrderedDict
        self._parts = OrderedDict([('dirname',''),
                                   ('prefix',''),
                                   ('ens',''),
                                   ('stream',''),
                                   ('op',''),
                                   ('varname',''),
                                   ('datestr',''),
                                   ('ext','nc')])
        self.update(**kwargs)

    def __str__(self):
        '''Return the file name as a string.
        '''
        return os.path.join(self._parts['dirname'],
                            '.'.join([s for k,s in self._parts.items()
                                      if s and k != 'dirname']))

    def __call__(self):
        '''Call __str__ method.
        '''
        return self.__str__()

    def _check_args(self,**kwargs):
        '''Private method: check that kwargs are defined fields.
        '''
        valid = [kw in self._parts.keys() for kw in kwargs.keys()]
        if not all(valid):
            raise AttributeError('%s has no attribute(s): %s'%
                                 (self.__class__.__name__,
                                 str([kwargs.keys()[i]
                                      for i,k in enumerate(valid) if not k])))

    def copy(self):
        '''Return a copy.
        '''
        import copy
        return copy.deepcopy(self)

    def append(self,**kwargs):
        '''Append a filename part with string.
        '''
        self._check_args(**kwargs)
        for key,value in kwargs.items():
            self._parts[key] = '_'.join([self._parts[key],value])
        return self

    def prepend(self,**kwargs):
        '''Prepend a filename part with string.
        '''
        self._check_args(**kwargs)
        for key,value in kwargs.items():
            self._parts[key] = '_'.join([value,self._parts[key]])
        return self

    def update(self,**kwargs):
        '''Change file name parts.
        '''
        self._check_args(**kwargs)
        self._parts.update(**kwargs)
        return self

    def exists(self):
        '''Check if file exists on disk.
        '''
        return os.path.exists(self())


#-------------------------------------------------------------------------------
#--- main
#-------------------------------------------------------------------------------
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
    if 'preprocess' in control['kwargs']:
        preprocess = control['kwargs']['preprocess']
        if hasattr(preprocess,'__iter__'):
            for i,func in enumerate(preprocess):
                preprocess[i] = eval(func)
        else:
            preprocess = eval(preprocess)

    print control
    task_function(**control['kwargs'])
