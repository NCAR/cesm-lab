import os
import numpy as np
import xarray as xr
import json
import tempfile
import copy

import task_manager as tm

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def ncrcat(input,output,depjob=[]):
    '''
    call ncrcat
    '''

    (fid,tmpfile) = tempfile.mkstemp('.filelist')
    with open(tmpfile,'w') as fid:
        for f in input:
            fid.write('%s\n'%f)
        jid = tm.submit(['cat',tmpfile,'|','ncrcat','-o',output],depjob=depjob)
    return jid

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def gen_time_chunks(start,stop,chunk_size):
    '''
    generate a list of index pairs
    '''

    time_level_count = stop - start
    nchunk =  time_level_count / chunk_size
    if time_level_count%chunk_size != 0:
        nchunk += 1
    time_ndx = [(start+i*chunk_size,start+i*chunk_size+chunk_size)
                for i in range(nchunk-1)] + \
                [(start+(nchunk-1)*chunk_size,stop)]

    return time_ndx

#------------------------------------------------------------
#-- function
#------------------------------------------------------------
def ncop_chunktime(script,kwargs,chunk_size,
                   start=0,stop=None,
                   clobber=False,cleanup=True):
    '''
    run script on time segments within a file and concatenate results
    '''
    jid_list = []
    def op_one_chunk(tnx):
        #-- intermediate output file
        file_out_i = file_out+'.tnx.%d-%d'%(tnx)

        #-- update input arguments
        kwargs.update({'dimsub' : {'time' : tnx},
                       'file_out': file_out_i})

        #-- submit
        print '\'{0}\''.format(json.dumps(kwargs))
        if not os.path.exists(file_out_i) or clobber:
            jid = tm.submit([script,'\'{0}\''.format(json.dumps(kwargs))])
            jid_list.extend(jid)
        return file_out_i

    file_out = copy.copy(kwargs['file_out'])

    if not os.path.exists(file_out) or clobber:
        #-- get time chunks
        if stop is None:
            ds = xr.open_dataset(kwargs['file_in'],
                                 decode_times=False,
                                 decode_coords=False)
            stop = len(ds.time)
        time_chunks = gen_time_chunks(start,stop,chunk_size)

        #-- operate on each chunk
        file_cat = [op_one_chunk(tnx) for tnx in time_chunks]

        #-- concatenate files
        jid = ncrcat(file_cat,file_out,depjob=jid_list)
        if cleanup:
            tm.submit(['rm','-f',' '.join(file_cat)],depjob=jid)

    return jid_list
