#! /usr/bin/env python
import os
from glob import glob
from datetime import datetime
from ncl import ncl
from config import ncl_regrid_tools,diro

nowstr = datetime.now().strftime('%Y%m%d')

#--------------------------------------------------------
#-- regrid var
#--------------------------------------------------------
def regrid_var(**kwargs):

    req = ['wgtFile','fname_in','varname_in','time_coordname','depth_coordname',
           'vert_grid_file','fname_out','varname_out','src_grid','dst_grid',
           'postfill_opt','prefill_opt','outfile_opt']

    for k in req:
        if k not in kwargs:
            print('missing %s'%k)
            exit(1)

    kwargs['task'] = 'regrid_var'
    ok = ncl(ncl_regrid_tools,kwargs)

    return ok

#--------------------------------------------------------
#-- regrid var
#--------------------------------------------------------
def gen_weight_file(**kwargs):

    req = ['wgtFile','srcGridFile','dstGridFile','wgtFile','InterpMethod']
    for k in req:
        if k not in kwargs:
            print('missing %s'%k)
            exit(1)
    kwargs['task'] = 'gen_weight_file'
    ok = ncl(ncl_regrid_tools,kwargs)

    return ok

#--------------------------------------------------------
#-- regrid var
#--------------------------------------------------------
def gen_latlon_grid_file(**kwargs):

    req = ['grid_out_fname','grid_type','left_lon_corner','dlat','dlon']
    for k in req:
        if k not in kwargs:
            print('missing %s'%k)
            exit(1)
    kwargs['task'] = 'gen_latlon_grid_file'
    ok = ncl(ncl_regrid_tools,kwargs)

    return ok

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def get_file_unknown_datestr(file_str_pattern,datestr):
    '''
    return lastest file or, if none exist, a file with the present date
    '''

    filels = sorted(glob(file_str_pattern('????????')))
    if not filels:
        return file_str_pattern(nowstr)
    else:
        return filels[-1]

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def wgt_file(src_grid,dst_grid,interp_method,datestr=''):
    file_str_pattern = lambda d: os.path.join(diro['weights'],
                                              '_'.join([src_grid,'to',dst_grid,
                                                        interp_method,d])+'.nc')
    if datestr:
        return file_str_pattern(datestr)
    else:
        return get_file_unknown_datestr(file_str_pattern,datestr)

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def grid_file(grid,datestr=''):
    file_str_pattern = lambda d: os.path.join(diro['grids'],
                                              '_'.join([grid,'SCRIP',d])+'.nc')
    if datestr:
        return file_str_pattern(datestr)
    else:
        return get_file_unknown_datestr(file_str_pattern,datestr)

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def vert_grid_file(grid,datestr=''):
    file_str_pattern = lambda d: os.path.join(diro['grids'],
                                              '_'.join([grid,'vert',d])+'.nc')
    if datestr:
        return file_str_pattern(datestr)
    else:
        return get_file_unknown_datestr(file_str_pattern,datestr)

#--------------------------------------------------------
#--- MAIN
#--------------------------------------------------------
if __name__ == '__main__':
    pass
