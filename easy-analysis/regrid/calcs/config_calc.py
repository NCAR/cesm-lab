import os
import sys
from datetime import datetime
from subprocess import call,PIPE,Popen
import xarray as xr
import numpy as np


path_tools = '~mclong/p/easy-analysis'
sys.path.insert(0,os.path.abspath(os.path.expanduser(path_tools)))

from regrid import regrid
