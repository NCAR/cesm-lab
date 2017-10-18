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
    'feventflux_gx1v6_5gmol_cesm1_97_2017' : {
    #----------------------------------------------
        'FESEDFLUXIN' :
        {'varname_in' : 'FESEDFLUXIN',
         'fname_in' : os.path.join(odir,'work','feventflux_gx1v6_5gmol_cesm1_97_2017.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'depth',
         'time_coordname' : 'none',
         'interp_method' : 'conserve'}},
    #----------------------------------------------
    'fesedflux_Oxic_plus_Reduc_gx1v6_cesm1_97_2017' :  {
    #----------------------------------------------
        'FESEDFLUXIN' :
        {'varname_in' : 'FESEDFLUXIN',
         'fname_in' : os.path.join(odir,'work','fesedflux_Oxic_plus_Reduc_gx1v6_cesm1_97_2017.nc'),
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
    'dst79gnx_gx1v6_090416' : {
    #----------------------------------------------
        'DSTSF' :
        {'varname_in' : 'DSTSF',
         'fname_in' : os.path.join(odir,'work','dst79gnx_gx1v6_090416.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'}},
    #----------------------------------------------
    'ndep_ocn_1850_w_nhx_emis_gx1v6_c160716' : {
    #----------------------------------------------
        'NHx_deposition' :
        {'varname_in' : 'NHx_deposition',
         'fname_in' : os.path.join(odir,'work','ndep_ocn_1850_w_nhx_emis_gx1v6_c160716.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'},
        'NOy_deposition' :
        {'varname_in' : 'NOy_deposition',
         'fname_in' : os.path.join(odir,'work','ndep_ocn_1850_w_nhx_emis_gx1v6_c160716.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'}},
    #----------------------------------------------
    'b.e20.B1850.f09_g17.pi_control.all.179.cplhist.cpl.presaero.clim.0037-0056.gx1v6.c20170823' : {
    #----------------------------------------------
        'DSTSF' :
        {'varname_in' : 'DSTSF',
         'fname_in' : os.path.join(odir,'work','b.e20.B1850.f09_g17.pi_control.all.179.cplhist.cpl.presaero.clim.0037-0056.gx1v6.c20170823.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'},
        'BC' :
        {'varname_in' : 'BC',
         'fname_in' : os.path.join(odir,'work','b.e20.B1850.f09_g17.pi_control.all.179.cplhist.cpl.presaero.clim.0037-0056.gx1v6.c20170823.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'},
        'solFe' :
        {'varname_in' : 'solFe',
         'fname_in' : os.path.join(odir,'work','b.e20.B1850.f09_g17.pi_control.all.179.cplhist.cpl.presaero.clim.0037-0056.gx1v6.c20170823.nc'),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'none',
         'time_coordname' : 'time',
         'interp_method' : 'conserve'}
        }
}


#-- make some modifications to coordinates
clean_up = []
for data_type,var_defs_dict in src_var_groups.items():

    flux_file_in = os.path.join(odir,data_type+'.nc')
    if 'feventflux' in data_type or 'fesedflux' in data_type:
        v = 'FESEDFLUXIN'
    elif any([data_type == d for d in ['solFe_scenario4_current_gx1v6','dst79gnx_gx1v6_090416','b.e20.B1850.f09_g17.pi_control.all.179.cplhist.cpl.presaero.clim.0037-0056.gx1v6.c20170823']]):
        v = 'DSTSF'
    elif 'ndep_ocn_1850_w_nhx_emis_gx1v6_c160716' == data_type:
        v = 'NHx_deposition'
    else:
        continue

    flux_file_out = var_defs_dict[v]['fname_in']

    if not os.path.exists(flux_file_out) or clobber:
        clean_up.append(flux_file_out)
        call(['cp','-v',flux_file_in,flux_file_out])

        #-- add depth coord to 3D fields
        if 'FESEDFLUXIN' in var_defs_dict:
            call(['ncks','-A','-v','depth',regrid.vert_grid_file('POP_gx1v7'),
                  flux_file_out])

        #-- pull coords from mapping file
        call(['ncks','-O','-v','TLATd,TLONd',regrid.grid_file('POP_gx1v7'),
              flux_file_out+'.tmp'])
        call(['ncrename','-v','TLATd,TLAT','-v','TLONd,TLONG',flux_file_out+'.tmp'])

        #-- remove old coords
        call(['ncks','-O','-v','TLONG,TLAT','-x',flux_file_out,flux_file_out])

        #-- append mapping file coords
        call(['ncks','-A',flux_file_out+'.tmp',flux_file_out])
        call(['rm','-f',flux_file_out+'.tmp'])

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
