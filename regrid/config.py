import os
from subprocess import call

diro = {}
diro['root'] = os.path.join('/glade/p/work/',os.environ['USER'],'regrid')
diro['grids'] = os.path.join(diro['root'],'grid_files')
diro['weights'] = os.path.join(diro['root'],'weight_files')
for pth in diro.values():
    if not os.path.exists(pth):
        call(['mkdir','-p',pth])

_path_to_here = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ncl_regrid_tools = os.path.join(_path_to_here,'py_regrid_tools.ncl')

os.environ['NCL_DEF_LIB_DIR'] = _path_to_here
