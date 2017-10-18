;
; land_mask.ncl
;
loadscript(loadpath+"/regrid_utils.ncl")

undef("apply_land_mask_POP")
procedure apply_land_mask_POP(var:numeric, grid_file:string, opt:logical)
local f_grid_file, KMT, var_dimsizes, var_rank, mask, time_ind, depth_ind
begin
   f_grid_file = addfile(grid_file, "r")
   KMT = f_grid_file->KMT

   var_dimsizes = dimsizes(var)
   var_rank = dimsizes(var_dimsizes)

   if (var_rank .eq. 2) then
      ; assume this is a surface field
      mask = KMT .gt. 0
      var = (/ where(mask, var, var@_FillValue) /)
   end if

   if (var_rank .eq. 3) then
      if (var!0 .ne. "depth") then
         ; assume this is a time-varying surface field
         mask = KMT .gt. 0
         do time_ind = 0, var_dimsizes(0)-1
            debug_print("applying land_mask, time_ind = ", time_ind, opt)
            var(time_ind,:,:) = (/ where(mask, var(time_ind,:,:), var@_FillValue) /)
         end do
      else
         ; assume this is a time-independent 3D field
         do depth_ind = 0, var_dimsizes(0)-1
            mask = depth_ind .lt. KMT
            debug_print("applying land_mask, depth_ind = ", depth_ind, opt)
            var(depth_ind,:,:) = (/ where(mask, var(depth_ind,:,:), var@_FillValue) /)
         end do
      end if
   end if

   if (var_rank .eq. 4) then
      ; assume this is a time-varying 3D field
      do time_ind = 0, var_dimsizes(0)-1
         debug_print("applying land_mask, time_ind = ", time_ind, opt)
         do depth_ind = 0, var_dimsizes(1)-1
            mask = depth_ind .lt. KMT
            debug_print("applying land_mask, depth_ind = ", depth_ind, opt)
            var(time_ind,depth_ind,:,:) = (/ where(mask, var(time_ind,depth_ind,:,:), var@_FillValue) /)
         end do
      end do
   end if
end
