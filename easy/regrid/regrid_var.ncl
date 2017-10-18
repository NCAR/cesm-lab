;
; regrid_var.ncl
;
loadpath = systemfunc("dirname "+get_script_name())
loadscript(loadpath+"/regrid_utils.ncl")
loadscript(loadpath+"/regrid_var_lateral.ncl")
loadscript(loadpath+"/regrid_var_depth.ncl")
loadscript(loadpath+"/land_mask.ncl")
loadscript(loadpath+"/fill.ncl")

setfileoption("nc","Format","LargeFile")

undef("regrid_var")
procedure regrid_var(wgtFile:string, fname_in:string, varname_in:string, time_coordname:string, depth_coordname:string, \
                     vert_grid_file:string, fname_out:string, varname_out:string, opt:logical)
local f_wgtFile, f_in, var_in, var_in_dimsizes, var_in_lateral_rank, non_lateral_rank, \
      dimind, time_dimind, depth_dimind, \
      var_out_lateral_rank, var_out_rank, var_lateral_regrid_dimsizes, var_lateral_regrid, \
      dst_grid_type, regrid_opt, var_out, f_vert_grid, depth, km, var_out_dimsizes, f_out
begin
   if (opt@outfile_opt .eq. "append" .and. .not. fileexists(fname_out)) then
      print("outfile_opt = append, but fname_out = "+fname_out+" does not exist")
      status_exit(1)
   end if

   debug_print("wgtFile = ", wgtFile, opt)
   if (.not. fileexists(wgtFile)) then
      print("wgtFile = "+wgtFile+" does not exist")
      status_exit(1)
   end if

   f_wgtFile = addfile(wgtFile, "r")

   debug_print("fname_in = ", fname_in, opt)
   if (.not. fileexists(fname_in)) then
      print("fname_in = "+fname_in+" does not exist")
      status_exit(1)
   end if

   f_in = addfile(fname_in, "r")

   debug_print("varname_in = ", varname_in, opt)
   if (.not. isfilevar(f_in, varname_in)) then
      print("varname_in = "+varname_in+" not a filevar in fname_in = "+fname_in)
      status_exit(1)
   end if

   ;
   ; ensure that varname_in is on src_grid from wgtFile
   ;
   verify_var_src_grid(f_in, varname_in, f_wgtFile, opt)

   ;
   ; read incoming variable into memory
   ;
   var_in = f_in->$varname_in$

   ;
   ; decompose var_in_rank into non_lateral and lateral ranks
   ;
   var_in_dimsizes = dimsizes(var_in)
   var_in_lateral_rank = dimsizes(f_wgtFile->src_grid_dims)
   non_lateral_rank = dimsizes(var_in_dimsizes) - var_in_lateral_rank

   ;
   ; verify that number of specified non-lateral dimensions matches that of var_in's rank
   ;
   dimind = 0
   if (time_coordname .ne. "none") then
      time_dimind = dimind
      dimind = dimind + 1
   else
      time_dimind = -1
   end if
   if (depth_coordname .ne. "none") then
      depth_dimind = dimind
      dimind = dimind + 1
   else
      depth_dimind = -1
   end if
   if (dimind .ne. non_lateral_rank) then
      print("mismatch between specified non-lateral coordinates and var rank")
      status_exit(1)
   end if

   ;
   ; generate rank of outgoing var
   ;
   var_out_lateral_rank = dimsizes(f_wgtFile->dst_grid_dims)
   var_out_rank = non_lateral_rank + var_out_lateral_rank

   ;
   ; generate dimension sizes of var with lateral regridding
   ; rank of var with lateral regridding is same as rank of outgoing var
   ;
   var_lateral_regrid_dimsizes = new(var_out_rank, integer)
   var_lateral_regrid_dimsizes(0:non_lateral_rank-1) = var_in_dimsizes(0:non_lateral_rank-1)
   var_lateral_regrid_dimsizes(non_lateral_rank:var_out_rank-1) = f_wgtFile->dst_grid_dims(::-1)
   debug_print("var_lateral_regrid_dimsizes = ", var_lateral_regrid_dimsizes, opt)

   ;
   ; verify that ranks and fill options are supported
   ;
   if (var_in_lateral_rank .lt. 1 .or. var_in_lateral_rank .gt. 2) then
      print("var_in_lateral_rank = "+var_in_lateral_rank+" not implemented")
      status_exit(1)
   end if
   if (var_in_lateral_rank .eq. 1) then
      if (.not. any(opt@prefill_opt .ne. (/ "none" /))) then
         print("prefill_opt = "+opt@prefill_opt+" not implemented for var_in_lateral_rank = "+var_in_lateral_rank)
         status_exit(1)
      end if
   end if
   if (non_lateral_rank .lt. 0 .or. non_lateral_rank .gt. 2) then
      print("non_lateral_rank = "+non_lateral_rank+" not implemented")
      status_exit(1)
   end if
   if (var_out_lateral_rank .lt. 1 .or. var_out_lateral_rank .gt. 2) then
      print("var_out_lateral_rank = "+var_out_lateral_rank+" not implemented")
      status_exit(1)
   end if
   if (var_out_lateral_rank .eq. 1) then
      if (.not. any(opt@postfill_opt .ne. (/ "none" /))) then
         print("postfill_opt = "+opt@postfill_opt+" not implemented for var_out_lateral_rank = "+var_out_lateral_rank)
         status_exit(1)
      end if
   end if

   ;
   ; allocate variable for lateral regridding, and generate metadata
   ;
   var_lateral_regrid = new(var_lateral_regrid_dimsizes, float)
   copy_VarAtts(var_in, var_lateral_regrid)
   if (isatt(var_lateral_regrid, "_FillValue_original")) then
      delete(var_lateral_regrid@_FillValue_original)
   end if

   ;
   ; copy non-lateral dimensions
   ;
   if (time_coordname .ne. "none") then
      var_lateral_regrid!time_dimind = time_coordname
      var_lateral_regrid&$time_coordname$ = f_in->$time_coordname$
   end if
   if (depth_coordname .ne. "none") then
      var_lateral_regrid!depth_dimind = depth_coordname
      var_lateral_regrid&$depth_coordname$ = f_in->$depth_coordname$
   end if

   ;
   ; generate lateral dimensions
   ;
   dst_grid_type = str_get_field(opt@dst_grid, 1, "_")
   debug_print("dst_grid_type = ", dst_grid_type, opt)
   if (dst_grid_type .eq. "POP") then
      var_lateral_regrid!(var_out_rank-2) = "nlat"
      var_lateral_regrid!(var_out_rank-1) = "nlon"
      var_lateral_regrid@coordinates = "TLONG TLAT"
   end if

   ;
   ; perform lateral regridding
   ;
   regrid_opt = True
;  regrid_opt@Debug = opt@Debug

   if (non_lateral_rank .eq. 0) then
      var_lateral_regrid = (/ tofloat(regrid_var_lateral_single_layer(wgtFile, regrid_opt, var_in, opt)) /)
   else
      regrid_var_lateral_multi_layers(wgtFile, regrid_opt, non_lateral_rank, var_in, var_lateral_regrid, opt)
   end if

   ;
   ; if var_in has a depth_coordname, remap vertically
   ;    otherwise copy from lateral remapping
   ;
   if (depth_coordname .eq. "none") then
      var_out = var_lateral_regrid
   else
      f_vert_grid = addfile(vert_grid_file, "r")
      depth = f_vert_grid->depth
      km = dimsizes(depth)

      var_out_dimsizes = var_lateral_regrid_dimsizes
      var_out_dimsizes(depth_dimind) = km

      ;
      ; allocate output variable, and generate metadata
      ;
      var_out = new(var_out_dimsizes, float)
      copy_VarAtts(var_lateral_regrid, var_out)

      ;
      ; copy time dimension from var_lateral_regrid, if present
      ; generate depth dimension
      ;
      if (time_coordname .ne. "none") then
         var_out!time_dimind = time_coordname
         var_out&$time_coordname$ = var_lateral_regrid&$time_coordname$
      end if
      var_out!depth_dimind = "depth"
      var_out&depth = depth

      ;
      ; generate lateral dimensions
      ; coordinate attribute already copied from copy_VarAtts call above
      ;
      if (dst_grid_type .eq. "POP") then
         var_out!(var_out_rank-2) = "nlat"
         var_out!(var_out_rank-1) = "nlon"
      end if
      if (time_coordname .eq. "none") then
         debug_print("calling regrid_var_depth_single_time, time_ind = ", -1, opt)
         regrid_var_depth_single_time(var_lateral_regrid, var_out, opt)
      else
         regrid_var_depth_multi_time(var_lateral_regrid, var_out, opt)
      end if
   end if


   ;
   ; apply land mask
   ;
   if (dst_grid_type .eq. "POP") then
      apply_land_mask_POP(var_out, f_wgtFile@grid_file_dst, opt)
   end if

   ;
   ; perform lateral fills on var_out
   ;
   if (opt@postfill_opt .eq. "fill_ocean_vals_smooth") then
      if (dst_grid_type .eq. "POP") then
         fill_ocean_vals_smooth_POP(var_out, f_wgtFile@grid_file_dst, opt)
      else
         print("postfill_opt = "+opt@postfill_opt+" only supported for dst_grid_type=POP")
         status_exit(1)
      end if
   end if

   if (dst_grid_type .eq. "rectilinear") then
     var_out!(var_out_rank-2) = "lat"
     var_out!(var_out_rank-1) = "lon"
   end if

   ;
   ; create output file
   ;
   if (opt@outfile_opt .eq. "create") then
      system("rm -f "+fname_out)
      f_out = addfile(fname_out, "c")

      ; make time dim unlimited (help with large vars?)
      if (time_coordname .ne. "none") then
        filedimdef(f_out,time_coordname,-1,True)
      end if

      ; add dst_grid specific grid info
      if (dst_grid_type .eq. "POP") then
         append_POP_grid_vars(f_out, f_wgtFile@grid_file_dst)
      end if

   end if

   if (opt@outfile_opt .eq. "append") then
      f_out = addfile(fname_out, "w")
   end if

   var_out@src_file = fname_in
   var_out@src_file_varname = varname_in
   var_out@wgtFile = wgtFile
   f_out->$varname_out$ = var_out

end
