#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J remap_z
#BSUB -o logs/remap_z.%J
#BSUB -e logs/remap_z.%J
#BSUB -q geyser
#BSUB -N
import os
from subprocess import call
from datetime import datetime,timedelta

script = 'sigma_remap_cesm_le.py'

#----------------------------------------
# SCRIPT
#----------------------------------------
os.putenv('PYTHONUNBUFFERED','no')
if not os.path.isdir('logs'):
    call(['mkdir','-p','logs'])

#----------------------------------------
#-- submit
time = datetime.now()
print('='*40)
print(time.strftime("%Y %m %d %H:%M:%S")+': Beginning '+script)
print('='*40)

stat = call([script])

print('='*40)
print(time.strftime("%Y %m %d %H:%M:%S")+': End '+script)
print('='*40)

#-- what to do next?
if stat == 1:
    print(script+' FAILED.')
    exit(1)
elif stat == 43:
    print('RESUBMIT')
    call('bsub < run.py',shell=True)
elif stat == 0:
    print(script+' finished.')
    exit(0)
else:
    print('unknown status code returned: '+stat)
    exit(1)

