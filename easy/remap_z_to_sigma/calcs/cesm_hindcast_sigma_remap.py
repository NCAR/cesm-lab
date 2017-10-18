#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 12:00
#BSUB -n 1
#BSUB -J z_to_sigma
#BSUB -o logs/remap_z.%J
#BSUB -e logs/remap_z.%J
#BSUB -q geyser
#BSUB -N
from config_calc import *

clobber = False

#-- que parameters
tm.QUEUE_MAX_HOURS = 20. # stop script (graceful exit) and wait on jobs after
tm.MAXJOBS = 400 # max number of jobs in queue
tm.ACCOUNT = 'NCGD0011'

#-- list cases
ddir = '/glade/p/decpred/cesm1_LE_dp_POPCICEhindcast'
case = 'g.e11_LENS.GECOIAF.T62_g16.009'

#-- variable to remap
varname = ['O2','AOU','TEMP','SALT']

#-- specify number of time levels to operate on
chunk_size = 5*12

#-- loop over cases
file_sig = sorted(glob(os.path.join(ddir,'.'.join([case,'pop.h','PD','*','nc']))))
for v in varname:

    #-- get list of all files
    file_var = sorted(glob(os.path.join(ddir,'.'.join([case,'pop.h',v,'*','nc']))))

    #-- loop over files and perform operation
    for f,s in zip(file_var,file_sig):
        print(f)
        print(s)
        file_out = os.path.join(diro['work'],
                                os.path.basename(f).replace(v,v+'.sigma'))

        control = {'file_in_sigma':s,
                   'file_in':f,
                   'file_out':file_out,
                   'dzname':'dz',
                   'kmtname': 'KMT', 
                   'zname': 'z_t', 
                   'convert_from_pd': True, 
                   'sigma_varname' : 'PD',
                   'sigma_start':24.475,
                   'sigma_stop':27.975,
                   'sigma_delta':0.05}


        jid = ncops.ncop_chunktime(script='.././calc_z_to_sigma.py',
                                   kwargs = control,
                                   chunk_size = chunk_size,
                                   clobber=clobber,
                                   cleanup=True)
tm.wait()
