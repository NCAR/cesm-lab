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

#-- configure the processing
tool_chain = {
    'aavgsurf' : [
        (et.calc_mean , {'dim':['nlat','nlon'],'isel':{'z_t':0,'z_t_150m':0}})
        ],
    'tavgsurf' : [
        (et.calc_ann_mean , {'isel':{'z_t':0,'z_t_150m':0}})
        ],
}

collections = {
    'timeseries_surface' : {
        'process_name' : 'aavgsurf',
        'processes' : [tool_chain['aavgsurf']],
        'variables' : ['pCO2SURF','NPP:photoC_sp,photoC_diat,photoC_diaz'],
        'stream' : 'monthly'
    },
    'timemean_surface' : {
        'process_name' : 'tavgsurf',
        'processes' : [tool_chain['tavgsurf']],
        'variables' : ['pCO2SURF','NPP:photoC_sp,photoC_diat,photoC_diaz'],
        'stream' : 'monthly'
    }
}

#-- model specific information
# TODO: enable multiple streams for particular temporal resolution:
#       pick right one based on variable name?
# place holder for getting files
def get_case_files(stream,varname):
    stream_definitions = {'monthly': {'name':'pop.h','datepatt':'????-??'},
                          'daily' :  {'name':'pop.h.nday1','datepatt':'????-??-??'}}

    stream_name = stream_definitions[stream]['name']
    stream_datepatt = stream_definitions[stream]['datepatt']
    file_pattern = '.'.join([case,stream_name,stream_datepatt,'nc'])
    return sorted(glob(os.path.join(casepath,file_pattern)))

#-- do the processing
datestr_out = '%04d-%04d'%tuple(year_range)
for category,category_spec in collections.items():

    stream = category_spec['stream']

    dir_output = os.path.join(diro['out'],case,
                              datestr_out,
                              category,stream)

    if not os.path.exists(dir_output):
        call(['mkdir','-p',dir_output])

    op = category_spec['process_name']
    preprocess = [p[0] for plist in category_spec['processes']
                  for p in plist]
    preprocess_kwargs = [p[1] for plist in category_spec['processes']
                         for p in plist]

    for v in category_spec['variables']:
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


        file_out = '.'.join([case,stream,op,varname,datestr_out,'nc'])
        file_out = os.path.join(dir_output,file_out)

        if not os.path.exists(file_out) or clobber:
            print(file_out)
            control = {'task' : 'open_tsdataset',
                       'kwargs': {'file_out': file_out,
                                  'paths': case_files,
                                  'year_range': year_range,
                                  'varname' : varsubset_varname,
                                  'preprocess' : preprocess,
                                  'preprocess_kwargs':preprocess_kwargs}}
            et.open_tsdataset(**control['kwargs'])
            #jid = tm.submit([easy,et.json_cmd(control)])
