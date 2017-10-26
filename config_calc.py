import os
import sys
from glob import glob
from subprocess import call

from datetime import datetime
import xarray as xr
import numpy as np

cesm_lab_path = '/glade/u/home/mclong/p/cesm-lab'
path_tools = [cesm_lab_path,os.path.join(cesm_lab_path,'easy')]
for p in path_tools:
    sys.path.insert(0,os.path.abspath(os.path.expanduser(p)))

import task_manager as tm
from regrid import regrid
import esm_tools as ez
easy = ez.__file__

calc_name = 'cesm-lab'
diro = {}
diro['out'] = '/glade/scratch/'+os.environ['USER']+'/calcs/'+calc_name
diro['notebooks'] = os.path.join(cesm_lab_path,'case_notebooks')
diro['fig'] =  './fig'
diro['logs'] = './logs'

for pth in diro.values():
    if not os.path.exists(pth):
        call(['mkdir','-p',pth])
