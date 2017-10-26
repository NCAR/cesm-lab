#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J o2dist
#BSUB -o logs/marbl_diag.%J
#BSUB -e logs/marbl_diag.%J
#BSUB -q caldera
#BSUB -N
from config_calc import *

clobber = False

#-- user input
case = 'g.e20a07c.GECO.T62_g17.test.004'
casepath = '/glade/scratch/mclong/archive/%s/ocn/hist'%case
year_range = [1,1]
datestr_out = '%04d-%04d'%tuple(year_range)

#-- configure the processing operations
toolbelt = {
    'aavgsurf' : [
        (et.calc_mean , {'dim':['nlat','nlon'],'isel':{'z_t':0,'z_t_150m':0}})
        ],
    'tavgsurf' : [
        (et.calc_ann_mean , {'isel':{'z_t':0,'z_t_150m':0}})
        ],
}


#-- model specific information
# TODO: enable multiple streams for particular temporal resolution:
#       pick right one based on variable name?
# place holder for getting files
def get_case_files(stream,varname):
    stream_definitions = {'monthly': {'name':'pop.h','datepatt':'????-??'},
                          'daily' :  {'name':'pop.h.ecosys.nday1','datepatt':'????-??-??'}}

    stream_name = stream_definitions[stream]['name']
    stream_datepatt = stream_definitions[stream]['datepatt']
    file_pattern = '.'.join([case,stream_name,stream_datepatt,'nc'])
    return sorted(glob(os.path.join(casepath,file_pattern)))


def open_transformed_dataset(process_name,processes,variables,stream):

    dir_output = os.path.join(diro['out'],case,
                              datestr_out,
                              stream)

    if not os.path.exists(dir_output):
        call(['mkdir','-p',dir_output])

    preprocess = [p[0] for plist in processes for p in plist]
    preprocess_kwargs = [p[1] for plist in processes for p in plist]

    ds = {}
    for v in variables:
        varname = v
        varsubset_varname = v

        if ':' in varname:
            varname = v.split(':')[0]
            varsubset_varname = v.split(':')[1].split(',')
            preprocess = [et.pop_derive_var]+preprocess
            preprocess_kwargs = [{'varname':varname}]+preprocess_kwargs
            case_files = {vi:get_case_files(stream,vi) for vi in varsubset_varname}
        else:
            case_files = get_case_files(stream,varname)

        file_out = '.'.join([case,stream,process_name,varname,datestr_out,'nc'])
        file_out = os.path.join(dir_output,file_out)

        if not os.path.exists(file_out) or clobber:
            dsi = et.open_tsdataset(file_out =  file_out,
                                   paths =  case_files,
                                   year_range =  year_range,
                                   varname = varsubset_varname,
                                   preprocess =  preprocess,
                                   preprocess_kwargs = preprocess_kwargs)
        else:
            dsi = xr.open_dataset(file_out,decode_times=False,decode_coords=False)
        ds = xr.merge((ds,dsi),compat='equals')

    return ds

if __name__ == '__main__':
    pass
