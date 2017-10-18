#! /usr/bin/env python
import os
from subprocess import call,Popen,PIPE
import tempfile
from glob import glob
from datetime import datetime

diro = {}
diro['root'] = os.path.join('/glade/p/work/',os.environ['USER'],'regrid')
diro['grids'] = os.path.join(diro['root'],'grid_files')
diro['weights'] = os.path.join(diro['root'],'weight_files')
for pth in diro.values():
    if not os.path.exists(pth):
        call(['mkdir','-p',pth])


pop_grid_file_datestr = '20170710'

#--------------------------------------------------------
#--- FUNCTION
#--------------------------------------------------------
def ncl_vardef(kwargs):
    cmd = []
    for k,v in kwargs.items():
        if isinstance(v, (str)):
            cmd.append('%s="%s"'%(k,v))
        elif isinstance(v, (int,long)):
            cmd.append('%s=%d'%(k,v))
        elif isinstance(v, (float)):
            cmd.append('%s=%e'%(k,v))
        else:
            print('type: of var: '+k+' not supported')
            exit(1)
    return cmd

#--------------------------------------------------------
#-- interface to NCL
#--------------------------------------------------------
def ncl(script_ncl,var_dict):

    #-- make temporary "exit" file (NCL must delete)
    (f,exitfile) = tempfile.mkstemp('','ncl-abnormal-exit.')
    var_dict.update({'ABNORMAL_EXIT': exitfile})
    
    #-- call NCL
    cmd = ['ncl','-n','-Q']+ncl_vardef(var_dict)+[script_ncl]
    p = Popen(cmd,stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()

    #-- error?  if return code not zero or exitfile exists
    ok = (p.returncode == 0) and (not os.path.exists(exitfile))
    
    #-- print stdout and stderr
    if out: print(out)
    if err: print(err)

    if not ok:
        print('NCL ERROR')
        if os.path.exists(exitfile): call(['rm','-f',exitfile])
        return False
    
    return True    

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
    ok = ncl('py_regrid_tools.ncl',kwargs)

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
    ok = ncl('py_regrid_tools.ncl',kwargs)

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
    ok = ncl('py_regrid_tools.ncl',kwargs)

    return ok    

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def wgt_file(src_grid,dst_grid,interp_method,datestr):                     
    return os.path.join(diro['weights'],
                        '_'.join([src_grid,'to',dst_grid,
                                  interp_method,datestr])+'.nc')

#--------------------------------------------------------
#-- function
#--------------------------------------------------------
def grid_file(grid):
    return os.path.join(diro['grids'],
                         '_'.join([grid,'SCRIP',datestr])+'.nc')
     
#--------------------------------------------------------
#--- MAIN
#--------------------------------------------------------
if __name__ == '__main__':
    now = datetime.now()
    nowstr = now.strftime('%Y%m%d')
    grid_file_datestr = '20170710'

    grid_out_fname = grid_file('latlon_1x1_180W',nowstr)
    grid_type = '1x1'
    left_lon_corner = -180.
    dlat = 1.
    dlon = 1.
    ok = gen_latlon_grid_file(grid_out_fname = grid_out_fname,
                              grid_type = grid_type,
                              left_lon_corner = left_lon_corner,
                              dlat = dlat,
                              dlon = dlon)


    src_grid = 'latlon_1x1_180W'
    dst_grid = 'POP_gx1v6'
    interp_method = 'conserve'
    
    wgtFile = wgt_file(src_grid,dst_grid,interp_method,nowstr)
    srcGridFile = grid_file(src_grid,nowstr)
    dstGridFile = grid_file(dst_grid,grid_file_datestr)
    ok = gen_weight_file(wgtFile = wgtFile,
                         srcGridFile = srcGridFile,
                         dstGridFile = dstGridFile,
                         InterpMethod = interp_method)
    print ok
    exit()

    
    diro['out'] = '/glade/scratch/'+os.environ['USER']+'/calcs/regrid'


    
    dst_grid = 'POP_gx3v7'
    interp_method = 'conserve'

    dst_grid = 'POP_gx1v6'
    interp_method = 'bilinear'

    dst_grid = 'POP_gx1v7'
    interp_method = 'bilinear'

    postfill_opt = 'fill_ocean_vals_smooth'
    outfile_opt = 'create'
    prefill_opt = 'zeros'    
   
    vert_grid_file = os.path.join(diro['grids'],
                                  dst_grid+'_vert_'+grid_file_datestr+'.nc')
    
    src_vars = {'NO3' : 
                {'varname_in' : 'n_an',
                 'fname_in' : '/glade/p/work/mclong/woa2013/1x1d/woa13_all_n00_01.nc',
                 'src_grid':'latlon_1x1_180W', 
                 'time_coordname' : 'time',
                 'depth_coordname' : 'depth'}}

    tmp_fname_out = diro['out']+'/regrid_tmp_'+dst_grid+'_'+nowstr+'.nc'
    fname_out = diro['out']+'/regrid_'+dst_grid+'_'+nowstr+'.nc'
    
    for varname_out,src in src_vars.items():
        wgtFile = wgt_file(diro['weights'],
                           src['src_grid'],
                           dst_grid,
                           interp_method,
                           grid_file_datestr)

        if not os.path.exists(wgtFile):
            srcGridFile = os.path.join(diro['grids'],
                                       src['src_grid']+'_SCRIP_'+grid_file_datestr+'.nc')
            dstGridFile = os.path.join(diro['grids'],
                                       dst_grid+'_SCRIP_'+grid_file_datestr+'.nc')
            
            ok = gen_weight_file(wgtFile = wgtFile,
                                 srcGridFile = srcGridFile,
                                 dstGridFile = dstGridFile,
                                 InterpMethod = interp_method)



        ok = regrid_var(wgtFile = wgtFile, 
                        fname_in = src['fname_in'], 
                        varname_in = src['varname_in'], 
                        time_coordname = src['time_coordname'],
                        depth_coordname = src['depth_coordname'],
                        vert_grid_file = vert_grid_file, 
                        fname_out = tmp_fname_out, 
                        varname_out = varname_out, 
                        src_grid = src['src_grid'],
                        dst_grid = dst_grid,
                        postfill_opt = postfill_opt,
                        prefill_opt = prefill_opt,
                        outfile_opt = outfile_opt)
        if not ok:
            exit(1)

        outfile_opt = 'append'
        exit()
