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

odir = '/glade/p/ncgd0033/inputdata/ic'
if not os.path.exists(odir): call(['mkdir','-p',odir+'/work'])

#----------------------------------------------------
#--- prepare FESEDFLUX dataset
#----------------------------------------------------
def avg_cur_old(rest_file,grid_file,vert_grid_file,file_out,varlist):

    #-- read dataset
    ds = xr.open_dataset(rest_file,decode_times=False,decode_coords=False)
    varlist_cur = [v+'_CUR' for v in varlist]
    #-- drop unwanted vars, rename
    dso = ds.drop([v for v in ds if v not in varlist_cur])
    dso = dso.rename({k+'_CUR' : k for k in varlist})

    #-- loop over vars, average CUR and OLD
    for v in varlist:
        dso[v].values = (ds[v+'_CUR'] + ds[v+'_OLD'])/2.
    ds.close()

    ds = xr.open_dataset(grid_file,decode_times=False,decode_coords=False)
    dso['TLONG'] = ds.TLONd
    dso['TLAT'] = ds.TLATd

    ds = xr.open_dataset(vert_grid_file,decode_times=False,decode_coords=False)
    dso['depth'] = ds.depth

    dso.to_netcdf(file_out)

#-- specify lo-res file
#rest_file = '/glade/scratch/mclong/archive/g.e20a07c.GECO.T62_g17.test.004/rest/0002-01-01-00000/g.e20a07c.GECO.T62_g17.test.004.pop.r.0002-01-01-00000.nc'
rest_file = '/glade/scratch/mclong/archive/g.e20a07c.GECO.T62_g17.test.003/rest/0011-01-01-00000/g.e20a07c.GECO.T62_g17.test.003.pop.r.0011-01-01-00000.nc'

#-- BGC vars to regrid
varlist = [
    'PO4','NO3','SiO3','NH4','Fe','Lig','O2','DIC','DIC_ALT_CO2','ALK',
    'ALK_ALT_CO2','DOC','DON','DOP','DOPr','DONr','DOCr','zooC',
    'spChl','spC','spP','spFe','spCaCO3','diatChl','diatC','diatP','diatFe',
    'diatSi','diazChl','diazC','diazP','diazFe']

#-- add coords to rest_file
if not os.path.exists(os.path.join(odir,'work',os.path.basename(rest_file))):
    avg_cur_old(rest_file = rest_file,
                grid_file = regrid.grid_file('POP_gx1v7'),
                vert_grid_file = regrid.vert_grid_file('POP_gx1v7'),
                file_out = os.path.join(odir,'work',os.path.basename(rest_file)),
                varlist = varlist)

#-- assemble variable groups
var_groups = {}
for v in varlist:
    var_groups.update({
        v :
        {'varname_in' : v,
         'fname_in' : os.path.join(odir,'work',os.path.basename(rest_file)),
         'src_grid' : 'POP_gx1v6',
         'depth_coordname' : 'depth',
         'time_coordname' : 'none',
         'interp_method' : 'bilinear'}})

src_var_groups = {os.path.basename(rest_file).replace('.nc','') : var_groups}

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
