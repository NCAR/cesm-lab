#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J o2dist
#BSUB -o logs/o2dist.%J
#BSUB -e logs/o2dist.%J
#BSUB -q caldera
#BSUB -N
from config_calc import *

clobber = False
