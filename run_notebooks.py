#! /usr/bin/env python
import os
import sys
import yaml

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def exec_nb(notebook_filename,kernel_name='python2'):
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
        out = ep.preprocess(nb, {'metadata': {'path': './'}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
        msg += 'See notebook "%s" for the traceback.' % notebook_filename
        print(msg)
        raise
    finally:
        with io.open(os.path.basename(notebook_filename),mode='wt',encoding='utf-8') as f:
            nbformat.write(nb, f)

    return out
#-------------------------------------------------------------------------------
#-- main
#-------------------------------------------------------------------------------

if __name__ == '__main__':

    input_arg = sys.argv[1]
    start_index = 0
    if len(sys.argv) > 2:
        start_index = int(sys.argv[2])

    extension = os.path.splitext(input_arg)[1]
    if extension == '.yaml':
        with open(input_arg, 'r') as fid:
            notebook = yaml.load(fid)
    else:
        notebook = [input_arg]

    for i,nb in enumerate(notebook):
        if i < start_index: continue
        print('\n'.join(['-'*80,'[%d] running %s'%(i,nb),'-'*80]))
        out = exec_nb(nb)
        print
