#! /usr/bin/env python
#BSUB -P NCGD0011
#BSUB -W 24:00
#BSUB -n 1
#BSUB -J virtual-drive
#BSUB -o log-virtual-drive.%J
#BSUB -e log-virtual-drive.%J
#BSUB -q caldera
#BSUB -N


import os
from subprocess import call
import sys
from glob import glob

vroot = os.path.join('/glade/scratch',os.environ['USER'],'CESM-CAM5-BGC-LE+ME')
if not os.path.exists(vroot):
    call(['mkdir','-pv',vroot])

CTRL = 'b.e11.B1850C5CN.f09_g16.005'
T20C = 'b.e11.B20TRC5CNBDRD.f09_g16'
TR85 = 'b.e11.BRCP85C5CNBDRD.f09_g16'
TR45 = 'b.e11.BRCP45C5CNBDRD.f09_g16'

#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------
def case_info(include_control=True,
              include_rcp85=True,
              include_rcp45=True):

    if include_control:
        caselist = [(CTRL,)]
        ens      = ['005']
        yr0      = [(1850 - 402)]
        scenario = ['ctrl']
    else:
        caselist = []
        ens      = []
        yr0      = []
        scenario = []

    #-- rcp85
    if include_rcp85:
        ensi = ['%03d'%i for i in range(1,36)
            if not any([i == ii for ii in range(3,9)]) and i != 33] # 33 has zeros for O2 in March 1938
        ensi.extend(['%03d'%i for i in range(101,106)])
        ens.extend(ensi)
        caselist.extend([(T20C+'.'+e,TR85+'.'+e) for e in ensi])
        yr0.extend([0 for e in ensi])
        scenario.extend(['tr85' for e in ensi])

    #-- rcp45
    if include_rcp45:
        ensi = ['%03d'%i for i in range(1,16) if not any([i == ii for ii in range(3,9)])]
        ens.extend(ensi)
        caselist.extend([(TR45+'.'+e,) for e in ensi])
        yr0.extend([0 for e in ensi])
        scenario.extend(['tr45' for e in ensi])

    return {'caselist' : caselist,
            'ens' : ens,
            'yr0' : yr0,
            'scenario' : scenario}


#----------------------------------------------------------------
#-- function
#----------------------------------------------------------------
def tseries_files(caselist,component,freq,variable,rootdir=vroot):

    #-- check that component is valid
    if not any([component == c for c in ['atm','ice','ocn','rof','lnd']]):
        print('unknown component')
        return None

    #-- construct path, check that it's there
    pth = os.path.join(rootdir,component,'proc/tseries',freq,variable)
    if not os.path.exists(pth):
        print('path not found: %s'%pth)
        return None

    file_list = []
    for i,case in enumerate(caselist):

        #-- make into iterable
        case_iter = case
        if not hasattr(case,'__iter__'):
            case_iter = [case]

        #-- find case/variable files
        case_files = []
        for c in case_iter:
            glob_res = sorted(glob(pth+'/'+c+'*'))
            if not glob_res:
                print('no files for case: '+str(case))
                return None
            case_files.extend(glob_res)

        file_list.append(case_files)

    return file_list


if __name__ == '__main__':

    #-- specifiy variables variables
    component = {'atm' : {'refvar' : 'FLDS',
                          'varlist' : ['FLDS','FSDS','FSNS',
                                       'PRECC','PRECL','TS','U','V','PSL','Q']},
                 'ocn' : {'refvar' : 'TEMP',
                          'varlist' : ['O2','AOU','O2SAT','STF_O2','TEMP','O2_CONSUMPTION',
                                       'O2_PRODUCTION','PD','IAGE','FG_CO2','SHF','QFLUX',
                                       'XMXL','HMXL','IAGE','SALT','DIC',
                                       'PO4','TEMP','SALT','PO4','NO3','SiO3','NH4',
                                       'Fe','O2','DIC','DIC_ALT_CO2','ALK',
                                       'ALK_ALT_CO2','DOC','DON','DOP','zooC',
                                       'spChl','spC','spP','spFe','spCaCO3','diatChl',
                                       'diatC','diatP','diatFe',
                                       'diatSi','diazChl','diazC','diazP','diazFe']}
                 }


    freq = 'monthly'

    for sim in ['LE','ME']:
        if sim == 'LE':
            info = case_info(include_control = True,
                             include_rcp85 = True,
                             include_rcp45 = False)
            droot = '/glade/p/cesmLE/CESM-CAM5-BGC-LE'
            hroot = '/CCSM/csm/CESM-CAM5-BGC-LE'
        elif sim == 'ME':
            info = case_info(include_control = False,
                             include_rcp85 = False,
                             include_rcp45 = True)
            droot = '/glade/p/ncgd0014/CESM-CAM5-BGC-ME'
            hroot = '/CCSM/csm/CESM-CAM5-BGC-LE'

        #-- find the files and either link or stage for transfer
        files_to_get = []
        for cmpnt,vardef in component.items():

            #-- set the variable list and reference variable
            varlist = vardef['varlist']
            refvar = vardef['refvar']

            #-- get expected files for reference variable
            files_expected = tseries_files(info['caselist'],cmpnt,freq,refvar,rootdir=droot)
            if files_expected is None:
                print('need some files')
                sys.exit(1)
            nfiles_expected = len([f for ff in files_expected for f in ff])

            #-- loop over variables and find missing files
            for variable in varlist:
                files = tseries_files(info['caselist'],cmpnt,freq,variable,rootdir=droot)
                try:
                    nfiles = len([f for ff in files for f in ff])
                except:
                    nfiles = 0

                if nfiles != nfiles_expected:
                    files_to_get.extend(
                        [f.replace('.'+refvar+'.','.'+variable+'.').replace('/'+refvar+'/','/'+variable+'/')
                         for ff in files_expected
                         for f in ff])
                else:
                    print('---soft linking---')
                    for ff in files:
                        for f in ff:
                            b = os.path.basename(f)
                            odir = os.path.dirname(f).replace(droot,vroot)
                            fo = os.path.join(odir,b)
                            if not os.path.exists(odir):
                                call(['mkdir','-pv',odir])
                            stat = call(['ln','-sfv',f,fo])
                            if stat != 0:
                                print('link failed')
                                sys.exit(1)

            #-- transfer missing files from tape
            for f in files_to_get:

                odir = os.path.dirname(f).replace(droot,vroot)
                if not os.path.exists(odir):
                    call(['mkdir','-pv',odir])

                lfile = f.replace(droot,vroot)
                hfile = f.replace(droot,hroot)

                cmd = ['hsi','cget '+lfile+' : '+hfile]
                print cmd
                stat = call(cmd)
                if stat != 0:
                    print('file transfer failed')
                    print('')
                    #sys.exit(1)
