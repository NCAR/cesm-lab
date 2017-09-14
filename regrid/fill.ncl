;
; fill.ncl
;
loadscript(loadpath+"/regrid_utils.ncl")

;external fill_POP_ext "./fill_POP_core.so"

undef("fill_ocean_vals_smooth_single_layer_POP")
procedure fill_ocean_vals_smooth_single_layer_POP(var:numeric, mask:logical, opt:logical)
local fill_mask, var_dimsizes, nlat, nlon, msv, tol
begin
   fill_mask = ismissing(var) .and. mask
   if (.not. any(fill_mask)) then
      return
   end if
   var_dimsizes = dimsizes(var)
   nlat = var_dimsizes(0)
   nlon = var_dimsizes(1)
   msv = var@_FillValue
   tol = 1.0e-4
   ;fill_POP_ext::FILL_POP_CORE(nlon, nlat, var, fill_mask, msv, tol, opt@ltripole)
   fill_pop_core(nlon, nlat, var, fill_mask, msv, tol, opt@ltripole)
end


undef("fill_ocean_vals_smooth_single_time_POP")
procedure fill_ocean_vals_smooth_single_time_POP(var:numeric, KMT:integer, opt:logical)
local var_dimsizes, mask, unfilled_vals_mask
begin
   var_dimsizes = dimsizes(var)
   do k = 0, var_dimsizes(0)-1
      debug_print("filling level k = ", k, opt)
      mask = k .lt. KMT
      fill_ocean_vals_smooth_single_layer_POP(var(k,:,:), mask, opt)

      ; if values were not filled, copy from the previous level
      if (k .gt. 0) then
         unfilled_vals_mask = ismissing(var(k,:,:)) .and. mask
         if (any(unfilled_vals_mask)) then
            var(k,:,:) = (/ where(unfilled_vals_mask, var(k-1,:,:), var(k,:,:)) /)
         end if
      end if
   end do
   if (isatt(var, "depth")) then
      delete(var@depth)
   end if
end


undef("fill_ocean_vals_smooth_POP")
procedure fill_ocean_vals_smooth_POP(var:numeric, grid_file:string, opt:logical)
local f_grid_file, KMT, var_dimsizes, var_rank, mask, time_ind, var_dims
begin
   f_grid_file = addfile(grid_file, "r")
   KMT = f_grid_file->KMT
   opt@ltripole = f_grid_file@ltripole .eq. 1

   var_dimsizes = dimsizes(var)
   var_rank = dimsizes(var_dimsizes)

   if (var_rank .eq. 2) then
      ; assume this is a surface field
      mask = KMT .gt. 0
      fill_ocean_vals_smooth_single_layer_POP(var, mask, opt)
   end if

   if (var_rank .eq. 3) then
      if (var!0 .ne. "depth") then
         ; assume this is a time-varying surface field
         mask = KMT .gt. 0
         do time_ind = 0, var_dimsizes(0)-1
            fill_ocean_vals_smooth_single_layer_POP(var(time_ind,:,:), mask, opt)
         end do
         var_dims = getvardims(var)
         if (isatt(var, var_dims(0))) then
            delete(var@$var_dims(0)$)
         end if
      else
         ; assume this is a time-independent 3D field
         fill_ocean_vals_smooth_single_time_POP(var, KMT, opt)
      end if
   end if

   if (var_rank .eq. 4) then
      ; assume this is a time-varying 3D field
      do time_ind = 0, var_dimsizes(0)-1
         fill_ocean_vals_smooth_single_time_POP(var(time_ind,:,:,:), KMT, opt)
      end do
      var_dims = getvardims(var)
      if (isatt(var, var_dims(0))) then
         delete(var@$var_dims(0)$)
      end if
   end if
end
