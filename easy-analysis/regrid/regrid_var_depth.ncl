;
; regrid_var_depth.ncl
;
loadscript(loadpath+"/regrid_utils.ncl")

undef("regrid_var_depth_single_time")
procedure regrid_var_depth_single_time(var_in:numeric, var_out:numeric, opt:logical)
local var_in_dimsizes, var_out_dimsizes, non_lateral_rank, lateral_rank, \
      dimind, depth_in, depth_out, fiCyclic, foOption, km_in, km_out, k
begin
   var_in_dimsizes = dimsizes(var_in)
   var_out_dimsizes = dimsizes(var_out)

   ;
   ; ensure that var_in and var_out have same rank
   ;
   if (dimsizes(var_in_dimsizes) .ne. dimsizes(var_out_dimsizes)) then
      debug_print("var_in_rank", dimsizes(var_in_dimsizes), opt)
      debug_print("var_out_rank", dimsizes(var_out_dimsizes), opt)
      print("argument rank mismatch")
      exit
   end if

   non_lateral_rank = 1
   lateral_rank = dimsizes(var_in_dimsizes) - non_lateral_rank

   ;
   ; ensure that lateral first dimension(s) of var_in and var_out have same length
   ;
   do dimind = 1, lateral_rank
      if (var_in_dimsizes(dimind) .ne. var_out_dimsizes(dimind)) then
         debug_print("dimind", dimind, opt)
         debug_print("var_in_dimsizes(dimind)", var_in_dimsizes(dimind), opt)
         debug_print("var_out_dimsizes(dimind)", var_out_dimsizes(dimind), opt)
         print("dimension length mismatch")
         exit
      end if
   end do

   ;
   ; perform linear interpolation in depth
   ;
   depth_in = var_in&$var_in!0$
   depth_out = var_out&$var_out!0$

   print(depth_in)
   print(depth_out)

   fiCyclic = False
   foOption = 0
   dimind = 0
   var_out = (/ linint1_n(depth_in, var_in, fiCyclic, depth_out, foOption, dimind) /)

   ;
   ; if depth_out is out of range of depth_in, copy values from nearest depth
   ;
   do k = 0, num(depth_out .lt. depth_in(0))-1
      debug_print("copy to level ", k, opt)
      if (lateral_rank .eq. 1) then
         var_out(k,:) = (/ var_in(0,:) /)
      end if
      if (lateral_rank .eq. 2) then
         var_out(k,:,:) = (/ var_in(0,:,:) /)
      end if
   end do
   km_in  = var_in_dimsizes(0)
   km_out = var_out_dimsizes(0)
   do k = km_out-num(depth_out .gt. depth_in(km_in-1)), km_out-1
      debug_print("copy to level ", k, opt)
      if (lateral_rank .eq. 1) then
         var_out(k,:) = (/ var_in(km_in-1,:) /)
      end if
      if (lateral_rank .eq. 2) then
         var_out(k,:,:) = (/ var_in(km_in-1,:,:) /)
      end if
   end do
end


undef("regrid_var_depth_multi_time")
procedure regrid_var_depth_multi_time(var_in:numeric, var_out:numeric, opt:logical)
local var_in_dimsizes, var_out_dimsizes, non_lateral_rank, lateral_rank, var_dims
begin
   var_in_dimsizes = dimsizes(var_in)
   var_out_dimsizes = dimsizes(var_out)

   ;
   ; ensure that var_in and var_out have same rank
   ; ensure that first dimension (time) of var_in and var_out have same length
   ;
   if (dimsizes(var_in_dimsizes) .ne. dimsizes(var_out_dimsizes)) then
      debug_print("var_in_rank", dimsizes(var_in_dimsizes), opt)
      debug_print("var_out_rank", dimsizes(var_out_dimsizes), opt)
      print("argument rank mismatch")
      status_exit(1)
   end if
   if (var_in_dimsizes(0) .ne. var_out_dimsizes(0)) then
      debug_print("var_in_dimsizes(0)", var_in_dimsizes(0), opt)
      debug_print("var_out_dimsizes(0)", var_out_dimsizes(0), opt)
      print("time length mismatch")
      status_exit(1)
   end if

   non_lateral_rank = 2
   lateral_rank = dimsizes(var_in_dimsizes) - non_lateral_rank

   if (lateral_rank .eq. 1) then
      do time_ind = 0, var_in_dimsizes(0)-1
         debug_print("calling regrid_var_depth_single_time, time_ind = ", time_ind, opt)
         regrid_var_depth_single_time(var_in(time_ind,:,:), var_out(time_ind,:,:), opt)
      end do
      var_dims = getvardims(var_out)
      if (isatt(var_out, var_dims(0))) then
         delete(var_out@$var_dims(0)$)
      end if
   end if

   if (lateral_rank .eq. 2) then
      do time_ind = 0, var_in_dimsizes(0)-1
         debug_print("calling regrid_var_depth_single_time, time_ind = ", time_ind, opt)
         regrid_var_depth_single_time(var_in(time_ind,:,:,:), var_out(time_ind,:,:,:), opt)
      end do
      var_dims = getvardims(var_out)
      if (isatt(var_out, var_dims(0))) then
         delete(var_out@$var_dims(0)$)
      end if
   end if
end
