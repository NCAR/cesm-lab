#!/bin/bash

#----------------------------------------------------------------
#---- environment
date=`date +%Y%m%d-%H%M%S`

f2py -c -m remap_z remap_z.f # -DF2PY_REPORT_ON_ARRAY_COPY=1.
f2py -c -m remap_z_dbl remap_z_dbl.f # -DF2PY_REPORT_ON_ARRAY_COPY=1.

mv *.so ../
