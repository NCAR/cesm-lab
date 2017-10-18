import os
import sys
from subprocess import call

path_tools = '~mclong/p/easy-analysis'
sys.path.insert(0,os.path.abspath(os.path.expanduser(path_tools)))

from datetime import datetime
from glob import glob
import xarray as xr
import numpy as np


import task_manager as tm
import ncops

diro = {}
diro['out'] = '/glade/scratch/mclong/calcs/oxygen'
diro['work'] = '/glade/scratch/mclong/calcs/oxygen/work'
diro['logs'] = './logs'

for pth in diro.values():
    if not os.path.exists(pth):
        call(['mkdir','-p',pth])
