#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 12:00
#BSUB -n 1
#BSUB -J z_to_sigma
#BSUB -o logs/remap_z.%J
#BSUB -e logs/remap_z.%J
#BSUB -q geyser
#BSUB -N
import os
from subprocess import call,Popen,PIPE
from glob import glob

import numpy as np
import xarray as xr

import task_manager as tm
import ops

clobber = False

#-- que parameters
tm.QUEUE_MAX_HOURS = 20. # stop script (graceful exit) and wait on jobs after
tm.MAXJOBS = 400 # max number of jobs in queue
tm.ACCOUNT = 'NCGD0011'

#-- output directory
odir = '/glade/scratch/'+os.environ['USER']+'/calcs/remap_z'
if not os.path.exists(odir):
    call(['mkdir','-p',odir])

#-- list cases
ddir = '/glade/p/cesmLE/CESM-CAM5-BGC-LE/ocn/proc/tseries/monthly'
CTRL = 'b.e11.B1850C5CN.f09_g16.005'
T20C = 'b.e11.B20TRC5CNBDRD.f09_g16'
TR85 = 'b.e11.BRCP85C5CNBDRD.f09_g16'
TR45 = 'b.e11.BRCP45C5CNBDRD.f09_g16'

CASELIST = [CTRL]
ENS      = ['005']
ens = ['%03d'%i for i in range(1,36) if not any([i == ii for ii in range(3,9)])]
ens.extend(['%03d'%i for i in range(101,106)])
ENS.extend(ens)
CASELIST.extend([(T20C+'.'+e,TR85+'.'+e) for e in ens])

#-- variable to remap
varname = 'O2'

#-- specify number of time levels to operate on
chunk_size = 5*12

#-- loop over cases
for casename in CASELIST:

    #-- get list of all files
    file_var = []
    file_sig = []
    for case in casename:
        file_var.extend(sorted(glob(os.path.join(ddir,varname,case+'*.nc'))))
        file_sig.extend(sorted(glob(os.path.join(ddir,'PD',case+'*.nc'))))

    #-- loop over files and perform operation
    for f,s in zip(file_var,file_sig):
        print(f)
        print(s)
        file_out = os.path.join(odir,
                                os.path.basename(f).replace(varname,varname+'.sigma'))

        control = {'file_in_sigma':s,
                   'file_in':f,
                   'file_out':file_out,
                   'dzname':'dz',
                   'kmtname': 'KMT', 
                   'zname': 'z_t', 
                   'convert_from_pd': True, 
                   'sigma_varname' : 'PD'}

        jid = ops.ncop_chunktime(script='calc_z_to_sigma.py',
                                 kwargs = control,
                                 chunk_size = chunk_size,
                                 clobber=clobber,
                                 cleanup=True)
tm.wait()
