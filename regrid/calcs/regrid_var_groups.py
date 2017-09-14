from config_calc import *

nowstr = datetime.now().strftime('%Y%m%d')

def regrid_var_groups(src_var_groups,
                      output_directory,
                      dst_grid,
                      vert_grid_file,
                      postfill_opt = 'fill_ocean_vals_smooth',
                      prefill_opt = 'zeros',
                      interp_method_default = 'bilinear',
                      clobber = False):



    #-- loop over variable groups
    for data_type,var_defs_dict in src_var_groups.items():

        outfile_opt = 'create'

        #-- create new file
        file_out_patt = lambda d: os.path.join(output_directory,'.'.join([data_type,dst_grid,d,'nc']))
        file_out = regrid.get_file_unknown_datestr(file_out_patt,nowstr)

        if os.path.exists(file_out) and not clobber:
            print('File exists: '+file_out)
            continue
        elif os.path.exists(file_out) and clobber:
            file_out = file_out_patt(nowstr)

        #-- ensure that output path exists
        if not os.path.exists(os.path.dirname(file_out)):
            call(['mkdir','-pv',os.path.dirname(file_out)])

        #-- loop over variables in group
        for varname_out,src in var_defs_dict.items():

            #-- source and destination grids
            src_grid = src['src_grid']
            srcGridFile = regrid.grid_file(src_grid)
            dstGridFile = regrid.grid_file(dst_grid)
            if not os.path.exists(srcGridFile):
                print('missing src grid file %s'%srcGridFile)
                exit(1)
            if not os.path.exists(dstGridFile):
                print('missing dst grid file %s'%dstGridFile)
                exit(1)

            #-- interp method: default or specified?
            if 'interp_method' in src:
                interp_method = src['interp_method']
            else:
                interp_method = interp_method_default

            #-- regrid weights file
            wgtFile = regrid.wgt_file(src_grid,dst_grid,interp_method)
            if not os.path.exists(wgtFile):
                print('missing weight file: %s'%wgtFile)
                print('generating: %s'%wgtFile)
                ok = regrid.gen_weight_file(wgtFile = wgtFile,
                                            srcGridFile = srcGridFile,
                                            dstGridFile = dstGridFile,
                                            InterpMethod = interp_method)
            #-- regrid variable
            print('-'*40)
            print('regridding %s on %s --> %s on %s'%(src['varname_in'],src_grid,
                                                       varname_out,dst_grid))
            ok = regrid.regrid_var(wgtFile = wgtFile,
                                   fname_in = src['fname_in'],
                                   varname_in = src['varname_in'],
                                   time_coordname = src['time_coordname'],
                                   depth_coordname = src['depth_coordname'],
                                   vert_grid_file = vert_grid_file,
                                   fname_out = file_out,
                                   varname_out = varname_out,
                                   src_grid = src_grid,
                                   dst_grid = dst_grid,
                                   postfill_opt = postfill_opt,
                                   prefill_opt = prefill_opt,
                                   outfile_opt = outfile_opt)
            if not ok: exit(1)

            outfile_opt = 'append'
