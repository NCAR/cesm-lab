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

xr_open_dataset = {'decode_times':False,
                   'decode_coords':False}

#-- user input
case = 'g.e20a07c.GECO.T62_g17.test.004'
casepath = '/glade/scratch/mclong/archive/%s/ocn/hist'%case
year_range = [1,1]
year_offset = 0
datestr_out = '%04d-%04d'%tuple(year_range)

#-- configure the processing operations
toolbelt = {
    'aavgsurf' : [
        (ez.calc_mean , {'dim':['nlat','nlon'],'isel':{'z_t':0,'z_t_150m':0}})
        ],
    'tavgsurf' : [
        (ez.calc_ann_mean , {'isel':{'z_t':0,'z_t_150m':0}}),
        (ez.calc_mean , {'dim':['time']})
        ],
}


#-- model specific information
# TODO: enable multiple streams for particular temporal resolution:
#       pick right one based on variable name?
# place holder for getting files
def get_case_files(stream):
    stream_definitions = {'monthly': {'name':'pop.h','datepatt':'????-??'},
                          'daily' :  {'name':'pop.h.ecosys.nday1','datepatt':'????-??-??'}}

    stream_name = stream_definitions[stream]['name']
    stream_datepatt = stream_definitions[stream]['datepatt']
    file_pattern = '.'.join([case,stream_name,stream_datepatt,'nc'])
    return sorted(glob(os.path.join(casepath,file_pattern)))

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def open_transformed_hist_dataset(process_name,processes,variables,collection_name,stream):

    dir_output = os.path.join(diro['out'],case,'proc',datestr_out,stream)

    if not os.path.exists(dir_output):
        call(['mkdir','-p',dir_output])


    preprocess = [p[0] for plist in processes for p in plist]
    preprocess_kwargs = [p[1] for plist in processes for p in plist]

    file_out = '.'.join([case,stream,process_name,collection_name,datestr_out,'nc'])
    file_out = os.path.join(dir_output,file_out)

    if os.path.exists(file_out) and not clobber:
        return xr.open_dataset(file_out,**xr_open_dataset)

    else:
        case_files = get_case_files(stream)

        #-- identify the grid vars
        dsg = xr.open_dataset(case_files[0],**xr_open_dataset)
        grid_vars = [k for k in dsg if 'time' not in dsg[k].dims]
        dsg = dsg.drop([k for k in dsg if k not in grid_vars])

        #-- read all the datasets
        dsa = xr.open_mfdataset(case_files,concat_dim='time',**xr_open_dataset)

        #-- replace concatenated grid vars with grid_vars
        dsa = dsa.drop(grid_vars)
        dsa = xr.merge((dsg,dsa))

        #-- time processing
        dsa = ez.interpret_time(dsa,year_offset=year_offset)
        dsa = ez.select_by_year(dsa,year_range=year_range)

        #-- compute derived vars
        varsubset_list = []
        for v in variables:
            varname = v
            varsubset_varname = v

            if ':' in varname:
                varname = v.split(':')[0]
                varsubset_varname = v.split(':')[1].split(',')
                dsa = ez.pop_derive_var(dsa,varname=varname)

            varsubset_list.append(varname)

        dsa = ez.variable_subset(dsa,
                                 varname = varsubset_list,
                                 keep_grids_vars = True)

        for func,kwargs in zip(preprocess,preprocess_kwargs):
            dsa = func(dsa,**kwargs)

        dsa.to_netcdf(file_out,unlimited_dims='time')

        return dsa

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def exec_nb(notebook_filename,output_path='./',kernel_name='python2'):
    '''execute a notebook
    see http://nbconvert.readthedocs.io/en/latest/execute_api.html
    '''
    import io
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors import CellExecutionError

    #-- open notebook
    with io.open(notebook_filename, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    #-- config for execution
    ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)

    #-- run with error handling
    try:
        out = ep.preprocess(nb, {'metadata': {'path': output_path}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
        msg += 'See notebook "%s" for the traceback.' % notebook_filename
        print(msg)
        raise
    finally:
        with io.open(os.path.join(output_path,notebook_filename),mode='wt',encoding='utf-8') as f:
            nbformat.write(nb, f)

    return out

if __name__ == '__main__':

    start_index = 0

    dir_output = os.path.join(diro['out'],case,'notebooks')
    if not os.path.exists(dir_output):
        call(['mkdir','-p',dir_output])

    call(['cp','config_calc.py',dir_output])
    notebooks = ['timemean_surface.ipynb','timeseries_surface.ipynb']



    for i,nb in enumerate(notebooks):
        if i < start_index: continue
        print('\n'.join(['-'*80,'[%d] running %s'%(i,nb),'-'*80]))
        exec_nb(nb,output_path=dir_output)
        print
