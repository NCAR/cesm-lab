#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J gen_POP_grid_files
#BSUB -o logs/regrid.%J
#BSUB -e logs/regrid.%J
#BSUB -q geyser
#BSUB -N

from config_calc import *
from regrid_var_groups import regrid_var_groups

nowstr = datetime.now().strftime('%Y%m%d')
clobber = False

dst_grid = 'POP_tx0.1v3_62lev'
vert_grid_file = regrid.vert_grid_file('POP_tx0.1_km62')

interp_method_default = 'bilinear'
postfill_opt = 'fill_ocean_vals_smooth'
prefill_opt = 'zeros'

odir = '/glade/p/ncgd0033/inputdata/forcing'
if not os.path.exists(odir): call(['mkdir','-p',odir])

#----------------------------------------------------
#--- prepare FESEDFLUX dataset
#----------------------------------------------------

src_var_groups = {
    #----------------------------------------------
    'feventflux_gx1v6_4gmol_cesm2.0_2017' : {
    #----------------------------------------------
        'FESEDFLUXIN' :
        {'varname_in' : 'FESEDFLUXIN',
         'fname_in' : os.path.join(odir,'work','feventflux_gx1v6_4gmol_cesm2.0_2017.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'depth',
         'time_coordname' : 'none',
         'interp_method' : 'conserve'}},
    #----------------------------------------------
    'fesedflux_gx1v6_cesm2_2017_tmp' :  {
    #----------------------------------------------
        'FESEDFLUXIN' :
        {'varname_in' : 'FESEDFLUXIN',
         'fname_in' : os.path.join(odir,'work','fesedflux_gx1v6_cesm2_2017_tmp.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'depth',
         'time_coordname' : 'none',
         'interp_method' : 'conserve'}},
    #----------------------------------------------
    'solFe_scenario4_current_gx1v6' : {
    #----------------------------------------------
        'DSTSF' :
        {'varname_in' : 'DSTSF',
         'fname_in' : os.path.join(odir,'work','solFe_scenario4_current_gx1v6.nc'),
        'src_grid' : 'POP_gx1v6',
        'depth_coordname' : 'none',
        'time_coordname' : 'time',
        'interp_method' : 'conserve'}},
    #----------------------------------------------
    'ndep_ocn_1850-2005_w_nhx_emis_gx1v6_c160717' : {
    #----------------------------------------------
        'NHx_deposition' :
        {'varname_in' : 'NHx_deposition',
         'fname_in' : os.path.join(odir,'work','ndep_ocn_1850-2005_w_nhx_emis_gx1v6_c160717.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'},
        'NOy_deposition' :
        {'varname_in' : 'NOy_deposition',
         'fname_in' : os.path.join(odir,'work','ndep_ocn_1850-2005_w_nhx_emis_gx1v6_c160717.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'}}
        }

#-- make some modifications to coordinates
clean_up = []
for data_type,var_defs_dict in src_var_groups.items():

    flux_file_in = os.path.join(odir,data_type+'.nc')
    if 'FESEDFLUXIN' in var_defs_dict:
        flux_file_out = var_defs_dict['FESEDFLUXIN']['fname_in']
    elif 'DSTSF' in var_defs_dict:
        flux_file_out = var_defs_dict['DSTSF']['fname_in']
    elif 'NHx_deposition' in var_defs_dict:
        flux_file_out = var_defs_dict['NHx_deposition']['fname_in']
    else:
        continue

    if not os.path.exists(flux_file_out) or clobber:
        clean_up.append(flux_file_out)
        call(['cp','-v',flux_file_in,flux_file_out])

        if 'DSTSF' in var_defs_dict:
            call(['ncrename','-d','X,nlon','-d','Y,nlat',flux_file_out])

        call(['ncks','-A','-v','depth',regrid.vert_grid_file('POP_gx1v7'),
              flux_file_out])
        call(['ncks','-A','-v','TLATd,TLONd',regrid.grid_file('POP_gx1v7'),
              flux_file_out])
        call(['ncrename','-v','TLATd,TLAT','-v','TLONd,TLONG',flux_file_out])

#----------------------------------------------------
#--- perform regridding
#----------------------------------------------------

regrid_var_groups(src_var_groups = src_var_groups,
                  output_directory = odir,
                  dst_grid = dst_grid,
                  vert_grid_file = vert_grid_file,
                  interp_method_default = interp_method_default,
                  postfill_opt = postfill_opt,
                  prefill_opt = prefill_opt,
                  clobber = clobber)

call(['rm','-f',' '.join(clean_up)])
