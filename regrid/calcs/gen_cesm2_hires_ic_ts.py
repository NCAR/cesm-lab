#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 12:00
#BSUB -n 1
#BSUB -J gen_cesm2_hires_ic_ts
#BSUB -o logs/regrid.%J
#BSUB -e logs/regrid.%J
#BSUB -q geyser
#BSUB -N
from config_calc import *

nowstr = datetime.now().strftime('%Y%m%d')
clobber = False

dst_grid = 'POP_tx0.1v3_62lev'
vert_grid_file = regrid.vert_grid_file('POP_tx0.1_km62')
interp_method_default = 'bilinear'

#dst_grid = 'POP_gx1v7'
#vert_grid_file = regrid.vert_grid_file('POP_gx1v7')
#interp_method_default = 'conserve'

postfill_opt = 'fill_ocean_vals_smooth'
prefill_opt = 'zeros'

#----------------------------------------------------
#--- blend Jan and Jan-Mar WOA data
#----------------------------------------------------

odir = '/glade/p/ncgd0033/inputdata/ic/work'
woa_pth = '/glade/p/ncgd0033/obs/woa2013v2'
if not os.path.exists(odir):
    call(['mkdir','-p',odir])

for ts in ['t','s']:
    #-- output file
    file_out = os.path.join(odir,'woa13_decav_%s01_blend_13_04v2.nc'%ts)
    if os.path.exists(file_out) and not clobber:
        continue

    #-- read Jan data
    ds01 = xr.open_dataset(woa_pth+'/0.25x0.25d/woa13_decav_%s01_04v2.nc'%ts,
                           decode_times = False,
                           decode_coords = False)
    nk = len(ds01.depth)

    #-- read Jan-Mar data
    ds13 = xr.open_dataset(woa_pth+'/0.25x0.25d/woa13_decav_%s13_04v2.nc'%ts,
                           decode_times = False,
                           decode_coords = False)

    #-- overwrite upper ocean with Jan data
    dso = ds13.copy()
    for v in ds13:
        if ds13[v].ndim == 4:
            print('blending %s'%v)
            dso[v].values[:,0:nk,:,:] = ds01[v].values[:]

    #-- produce netcdf file
    dso.to_netcdf(file_out)
    print('wrote %s'%file_out)

#----------------------------------------------------
#--- perform regridding
#----------------------------------------------------

odir = '/glade/p/ncgd0033/inputdata/ic'
woa_pth = '/glade/p/ncgd0033/inputdata/ic/work'
if not os.path.exists(odir):
    call(['mkdir','-p',odir])

src_var_groups = {
    'ts.woa2013v2_0.25.ic' : {
        #----------------------------------------------
        'TEMPERATURE' :
        {'varname_in' : 't_an',
         'fname_in' : woa_pth+'/woa13_decav_t01_blend_13_04v2.nc',
         'src_grid':'latlon_0.25x0.25_180W',
         'time_coordname' : 'time',
         'depth_coordname' : 'depth'},
        #----------------------------------------------
        'SALINITY' :
        {'varname_in' : 's_an',
         'fname_in' : woa_pth+'/woa13_decav_s01_blend_13_04v2.nc',
         'src_grid':'latlon_0.25x0.25_180W',
         'time_coordname' : 'time',
         'depth_coordname' : 'depth'}}}

#-- loop over variable groups
for data_type,var_defs_dict in src_var_groups.items():

    #-- create new file
    outfile_opt = 'create'
    file_out = os.path.join(odir,'.'.join([data_type,dst_grid,nowstr,'nc']))
    if os.path.exists(file_out) and not clobber:
        continue

    #-- ensure that output path exists
    if not os.path.exists(os.path.dirname(file_out)):
        call(['mkdir','-pv',os.path.dirname(file_out)])

    #-- loop over variables in group
    for varname_out,src in var_defs_dict.items():

        #-- source and destination grids
        src_grid = src['src_grid']
        srcGridFile = regrid.grid_file(src_grid)
        dstGridFile = regrid.grid_file(dst_grid)
        if not os.path.exists(srcGridFile):
            print('missing src grid file %s'%srcGridFile)
            exit(1)
        if not os.path.exists(dstGridFile):
            print('missing dst grid file %s'%dstGridFile)
            exit(1)

        #-- interp method: default or specified?
        if 'interp_method' in src:
            interp_method = src['interp_method']
        else:
            interp_method = interp_method_default

        #-- regrid weights file
        wgtFile = regrid.wgt_file(src_grid,dst_grid,interp_method)
        if not os.path.exists(wgtFile):
            print('missing weight file: %s'%wgtFile)
            print('generating: %s'%wgtFile)
            ok = regrid.gen_weight_file(wgtFile = wgtFile,
                                        srcGridFile = srcGridFile,
                                        dstGridFile = dstGridFile,
                                        InterpMethod = interp_method)
        #-- regrid variable
        print('-'*40)
        print('regridding %s on %s --> %s on %s'%(src['varname_in'],src_grid,
                                                   varname_out,dst_grid))
        ok = regrid.regrid_var(wgtFile = wgtFile,
                               fname_in = src['fname_in'],
                               varname_in = src['varname_in'],
                               time_coordname = src['time_coordname'],
                               depth_coordname = src['depth_coordname'],
                               vert_grid_file = vert_grid_file,
                               fname_out = file_out,
                               varname_out = varname_out,
                               src_grid = src_grid,
                               dst_grid = dst_grid,
                               postfill_opt = postfill_opt,
                               prefill_opt = prefill_opt,
                               outfile_opt = outfile_opt)
        if not ok: exit(1)

        #-- set to append for rest of group
        outfile_opt = 'append'
