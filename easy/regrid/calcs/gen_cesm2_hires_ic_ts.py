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
from regrid_var_groups import regrid_var_groups
import seawater as sw

nowstr = datetime.now().strftime('%Y%m%d')
clobber = False

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
#--- convert to potential temperature
#----------------------------------------------------

woa_pth = '/glade/p/ncgd0033/inputdata/ic/work'

file_out = woa_pth+'/woa13_decav_ptmp01_blend_13_04v2.nc'
if not os.path.exists(file_out):

    #-- read temperature and salinity
    dst = xr.open_dataset(woa_pth+'/woa13_decav_t01_blend_13_04v2.nc',
                          decode_times = False,
                          decode_coords = False)
    dss = xr.open_dataset(woa_pth+'/woa13_decav_s01_blend_13_04v2.nc',
                          decode_times = False,
                          decode_coords = False)

    #-- compute pressure
    na = np.newaxis
    pressure = sw.eos80.pres(dst.depth.values[na,:,na,na],dst.lat.values[na,na,:,na])

    dso = dst.copy()
    dso.t_an.values = sw.eos80.ptmp(dss.s_an.values, dst.t_an.values, pressure, pr=0.)
    dso.t_an.attrs['note'] = 'Coverted to potential temperature using EOS-80: http://pythonhosted.org/seawater/eos80.html'
    dso.to_netcdf(file_out)
    print('wrote %s'%file_out)

#----------------------------------------------------
#--- perform regridding
#----------------------------------------------------

odir = '/glade/p/ncgd0033/inputdata/ic'
if not os.path.exists(odir):
    call(['mkdir','-p',odir])

src_var_groups = {
    'ts.woa2013v2_0.25.ic' : {
        #----------------------------------------------
        'TEMPERATURE' :
        {'varname_in' : 't_an',
         'fname_in' : woa_pth+'/woa13_decav_ptmp01_blend_13_04v2.nc',
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

dst_grid = 'POP_tx0.1v3_62lev'
vert_grid_file = regrid.vert_grid_file('POP_tx0.1_km62')
interp_method_default = 'bilinear'

regrid_var_groups(src_var_groups = src_var_groups,
                  output_directory = odir,
                  dst_grid = dst_grid,
                  vert_grid_file = vert_grid_file,
                  interp_method_default = interp_method_default,
                  postfill_opt = postfill_opt,
                  prefill_opt = prefill_opt,
                  clobber = clobber)

dst_grid = 'POP_gx1v7'
vert_grid_file = regrid.vert_grid_file('POP_gx1v7')
interp_method_default = 'conserve'

regrid_var_groups(src_var_groups = src_var_groups,
                  output_directory = odir,
                  dst_grid = dst_grid,
                  vert_grid_file = vert_grid_file,
                  interp_method_default = interp_method_default,
                  postfill_opt = postfill_opt,
                  prefill_opt = prefill_opt,
                  clobber = clobber)
