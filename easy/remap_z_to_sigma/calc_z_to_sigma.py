#! /usr/bin/env python
import xarray as xr
import numpy as np
from netCDF4 import default_fillvals
import remap_z
import remap_z_dbl

verbose = True

#------------------------------------------------------------------------
#-- MODULE VARIABLES
#------------------------------------------------------------------------
xr_open_dataset_args = {'decode_times':False,'decode_coords':False}

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def remap_z_type(klev_out,KMT,z_edges,VAR_IN,NEW_Z,new_z_edge):

    #-- make sure that type is the same
    NEW_Z = NEW_Z.astype(VAR_IN.dtype)
    new_z_edge = new_z_edge.astype(VAR_IN.dtype)
    z_edges = z_edges.astype(VAR_IN.dtype)
    
    if VAR_IN.dtype == 'float32':
        status('calling remap_z: single precision')
        THICKNESS,VAR_OUT = remap_z.remap_z(klev_out = klev_out,
                                            kmt = KMT.T,
                                            z_edge = z_edges, 
                                            var_in = VAR_IN.T, 
                                            new_z = NEW_Z.T,
                                            new_z_edge = new_z_edge,
                                            msv = default_fillvals['f4'])
    elif VAR_IN.dtype == 'float64':
        status('calling remap_z: double precision')
        THICKNESS,VAR_OUT = remap_z_dbl.remap_z_dbl(klev_out = klev_out,
                                                    kmt = KMT.T,
                                                    z_edge = z_edges, 
                                                    var_in = VAR_IN.T, 
                                                    new_z = NEW_Z.T,
                                                    new_z_edge = new_z_edge,
                                                    msv = default_fillvals['f8'])
    else: 
        status('ERROR: unable to determine type',error=True)

    return THICKNESS.T,VAR_OUT.T

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def _dim_index_to_slice(index_list):
    '''
    .. function:: _dim_index_to_slice(index_list)

    Convert string formatted as: dimname,start[,stop[,stride]]
    to index (for the case where only 'start' is provided)
    or indexing object (slice).

    :param index_list: index list as passed in from
                       -d dimname,start,stop,stride

    :returns: dict -- {dimname: indexing object}
    '''

    if len(index_list) == 1:
        return index_list[0]
    elif len(index_list) == 2:
        return slice(index_list[0],index_list[1])
    elif len(index_list) == 3:
        return slice(index_list[0],index_list[1],index_list[2])
    else:
        status('ERROR: illformed dimension subset',error=True)
        exit(1)

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def sigma_coord_edges(sigma_start=24.475,
                      sigma_stop =26.975,
                      dsigma = 0.05):
    sigma_edges = np.arange(sigma_start,sigma_stop+dsigma,dsigma)
    return sigma_edges

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def compute_kmt(ds,varname):
    nk = ds[varname].shape[1]
    #-- init KMT array
    KMT = np.zeros(ds[varname].shape[-2:]).astype(int)
    
    #-- where surface is missing, KMT = 0, else full depth
    KMT = np.where(np.isnan(ds[varname].values[0,0,:,:]),0,nk)

    #-- loop over k
    #   where level k is missing: KMT = k, i.e. the level above in 1-based indexing
    for k in range(1,nk):
        KMT = np.where(np.isnan(ds[varname].values[0,k,:,:]) & (KMT > k),k,KMT)

    return KMT

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def compute(file_in,file_out,
            file_in_sigma,
            sigma_varname,
            convert_from_pd,
            sigma_start,
            sigma_stop,
            sigma_delta,
            zname,
            dzname,
            kmtname = '',
            dimsub = {}):

    #-- read sigma file
    SIGMA = read_sigma(file_in_sigma,sigma_varname,convert_from_pd,dimsub)
    
    #-- compute sigma coordinate
    sigma_edges = sigma_coord_edges(sigma_start,sigma_stop,sigma_delta)
    
    #-- open dataset
    ds = xr.open_dataset(file_in,**xr_open_dataset_args)
    if dimsub:
        ds = ds.isel(**dimsub)

    #-- define output dataset
    dso = ds.copy()

    #-- list vars to remap, drop those with "z" dimension
    #   find dimesion lengths
    tlev = len(ds.time)
    varname_list = []
    for v in ds:
        if zname in ds[v].dims:
            if ds[v].ndim == 4: 
                varname_list.append(v)
            dso = dso.drop(v)
    
    if not varname_list:
        status('ERROR: no 4D variables found',error=True)
    else:
        jmt = ds[varname_list[0]].shape[-2]
        imt = ds[varname_list[0]].shape[-1]
        jdim = ds[varname_list[0]].dims[-2]
        idim = ds[varname_list[0]].dims[-1]

    #-- the new coordinate
    sigma = np.average(np.vstack((sigma_edges[0:-1],sigma_edges[1:])),axis=0)
    klev_out = len(sigma_edges)-1        
    dso = dso.assign_coords(sigma=sigma)
    dso.sigma.attrs = {'long_name':'sigma',
                       'units' : 'kg/m^3'}

    #-- get old coordinate
    status('reading z')
    status(ds[zname])
    z = get_values(ds,zname)
    klev = len(z)
    status('')

    status('reading dz')
    status(ds[dzname])
    dz = get_values(ds,dzname)
    z_edges = np.concatenate(([0.],np.cumsum(dz)))
    status('')

    #-- read 1-based index of bottom level
    if kmtname:
        status('reading kmt')
        status(ds[kmtname])
        KMT = get_values(ds,kmtname)
        status('')
    else:
        status('constructing kmt')
        KMT = compute_kmt(ds,varname_list[0])
        dso['KMT_c'] = xr.DataArray(KMT,
                                    dims=(jdim,idim),
                                    attrs={'long_name':'Index of bottom cell'})
        
        status('')
   
    if SIGMA.shape != (tlev,klev,jmt,imt):
        status('ERROR: SIGMA has shape: %s; expected shape %s'%(
                str(SIGMA.shape),str((tlev,klev,jmt,imt))),error=True)


    #-- remap z to get a thickness and depth field
    Z = np.broadcast_to(z[np.newaxis,:,np.newaxis,np.newaxis],(tlev,klev,jmt,imt))

    VAR_OUT = np.empty((imt,jmt,klev_out,tlev)).astype(Z.dtype)
    THICKNESS = VAR_OUT.copy()

    status('remapping z')
    THICKNESS,VAR_OUT = remap_z_type(klev_out,KMT,z_edges,Z,SIGMA,sigma_edges)   

    dso['Z'] = xr.DataArray(VAR_OUT,
                            dims=('time','sigma',jdim,idim),
                            attrs = {'long_name':'Depth',
                                     'units':ds[zname].attrs['units']})
    
    dso['THICKNESS'] = xr.DataArray(THICKNESS,
                                    dims=('time','sigma',jdim,idim),
                                    attrs={'long_name':'Thickness',
                                           'units':ds[zname].attrs['units']})
    status('')

    #-- remap variables
    for v in varname_list:
        status('remapping %s'%v)
        status(ds[v])
        VAR = get_values(ds,v)
        VAR_OUT = np.empty((imt,jmt,klev_out,tlev)).astype(VAR.dtype)

        _,VAR_OUT = remap_z_type(klev_out,KMT,z_edges,VAR,SIGMA,sigma_edges)   

        dso[v] = xr.DataArray(VAR_OUT,
                              dims=('time','sigma',jdim,idim),
                              attrs = ds[v].attrs)
        status('')

    #-- write output (or not)
    if file_out:
        dso.to_netcdf(file_out,unlimited_dims='time')
    else: 
        return dso


#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def read_sigma(file_in,sigma_varname,
               convert_from_pd = False,
               dimsub = {}):
    
    ds = xr.open_dataset(file_in,**xr_open_dataset_args)
    if dimsub:
        ds = ds.isel(**dimsub)

    status('reading %s'%sigma_varname)
    status(ds[sigma_varname])    
    sigma = get_values(ds,sigma_varname)

    if convert_from_pd:
        sigma_order = np.floor(np.log10(np.nanmean(sigma)))

        if sigma_order == 0.:
            status('%s assumed to be cgs units'%(sigma_varname))
            sigma = (sigma - 1.)*1000.
        elif sigma_order == 3.:
            status('%s assumed to be mks units'%(sigma_varname))
            sigma = (sigma - 1000.)
        else:
            status('ERROR: the units of %s could not be determined'%sigma_varname,error=True)

    status('')
    return sigma

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def get_values(ds,v):
    return ds[v].values.astype(ds[v].encoding['dtype'])

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def status(msg,error=False):
    if verbose:
        print(msg)
    if error:
        exit(1)
   
#------------------------------------------------------------------------
#-- MAIN
#------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    import sys
    import json
    
    #-- set defaults
    control_defaults = {
        'file_in':None,
        'file_out':None,
        'file_in_sigma':None,
        'zname':None,
        'dzname':None,
        'kmtname':'',
        'sigma_varname':None,
        'convert_from_pd':False,
        'dimsub': {},
        'sigma_start':24.475,
        'sigma_stop':26.975,
        'sigma_delta':0.05}
    
    help_str = []
    for k,v in control_defaults.items():
        if v is None:
            help_str.append('%s : REQUIRED'%k)
        elif not v:
            help_str.append('%s : \'\''%k)
        else:
            help_str.append('%s : %s'%(k,v))

    p = argparse.ArgumentParser(description='Regrid to sigma coords')
    p.add_argument('json_control',
                   default=control_defaults,
                   help = '{'+', '.join(help_str)+'}')
    p.add_argument('-f',dest='json_as_file',
                   action='store_true',default=False,
                   help='Interpret input as a file name')

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
    req = ['file_in','file_out','zname','dzname','kmtname','sigma_varname']
    for k in req:
        if control[k] is None:
            status('ERROR: missing %s'%k)
            missing_req = True
    if missing_req: status('stopping',error=True)

    #-- if no sigma file, assume same as input file
    if control['file_in_sigma'] is None:
        control['file_in_sigma'] = control['file_in']

    #-- convert dimsub from list to slice format 
    if control['dimsub']:
        for k,v in control['dimsub'].items():
            control['dimsub'][k] = _dim_index_to_slice(v)

    status('running remap_z')
    for k,v in control.items():
        status('%s = %s'%(k,str(v)))
    status('')

    #-- compute
    compute(**control)

