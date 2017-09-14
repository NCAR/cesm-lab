;
; regrid_var_lateral.ncl
;
loadscript(loadpath+"/regrid_utils.ncl")
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

undef("regrid_var_lateral_single_layer")
function regrid_var_lateral_single_layer(wgtFile:string, regrid_opt:logical, layer_in:numeric, opt:logical)
local layer_in_ones, layer_out, layer_out_ones
begin
   if (opt@prefill_opt .eq. "zeros") then
      layer_in_ones = where(ismissing(layer_in), 0.0, 1.0)
   end if

   fill_layer(layer_in, opt@prefill_opt)

   ;
   ; regrid single lateral layer
   ;
   layer_out = ESMF_regrid_with_weights(layer_in, wgtFile, regrid_opt)

   ;
   ; normalize regridded layer
   ;
   if (opt@prefill_opt .eq. "zeros") then
      layer_out_ones = ESMF_regrid_with_weights(layer_in_ones, wgtFile, regrid_opt)
      layer_out_ones = where(layer_out_ones .gt. 0.0, layer_out_ones, layer_out_ones@_FillValue)
      layer_out = layer_out / layer_out_ones
   end if

   return(layer_out)
end


undef("regrid_var_lateral_multi_layers")
procedure regrid_var_lateral_multi_layers(wgtFile:string, regrid_opt:logical, non_lateral_rank:integer, var_in:numeric, \
                                          var_lateral_regrid:numeric, opt:logical)
local var_in_dimsizes, var_in_lateral_rank, var_out_lateral_rank, non_lateral_dim_inds, non_lateral_loops_done, layer_in, layer_out
begin
   if (non_lateral_rank .eq. 0) then
      print("regrid_var_lateral_multi_layers only handles regrid_var_lateral_multi_layers > 0")
      exit
   end if

   var_in_dimsizes = dimsizes(var_in)
   var_in_lateral_rank = dimsizes(var_in_dimsizes) - non_lateral_rank
   var_out_lateral_rank = dimsizes(dimsizes(var_lateral_regrid)) - non_lateral_rank

   ;
   ; process each layer, looping over non-lateral dimensions
   ;
   non_lateral_dim_inds = new(non_lateral_rank, integer)
   non_lateral_dim_inds = 0
   non_lateral_loops_done = False
   do while (.not. non_lateral_loops_done)

      debug_print("non_lateral_dim_inds = ", non_lateral_dim_inds, opt)

      ;
      ; extract single lateral layer from var_in
      ;
      if (non_lateral_rank .eq. 1) then
         if (var_in_lateral_rank .eq. 1) then
            layer_in = var_in(non_lateral_dim_inds(0),:)
         else
            layer_in = var_in(non_lateral_dim_inds(0),:,:)
         end if
      end if
      if (non_lateral_rank .eq. 2) then
         if (var_in_lateral_rank .eq. 1) then
            layer_in = var_in(non_lateral_dim_inds(0),non_lateral_dim_inds(1),:)
         else
            layer_in = var_in(non_lateral_dim_inds(0),non_lateral_dim_inds(1),:,:)
         end if
      end if

      ;
      ; regrid single lateral layer
      ;
      layer_out = regrid_var_lateral_single_layer(wgtFile, regrid_opt, layer_in, opt)

      ;
      ; store regridded layer to var_lateral_regrid
      ;
      if (non_lateral_rank .eq. 1) then
         if (var_out_lateral_rank .eq. 1) then
            var_lateral_regrid(non_lateral_dim_inds(0),:) = (/ tofloat(layer_out) /)
         else
            var_lateral_regrid(non_lateral_dim_inds(0),:,:) = (/ tofloat(layer_out) /)
         end if
      end if
      if (non_lateral_rank .eq. 2) then
         if (var_out_lateral_rank .eq. 1) then
            var_lateral_regrid(non_lateral_dim_inds(0),non_lateral_dim_inds(1),:) = (/ tofloat(layer_out) /)
         else
            var_lateral_regrid(non_lateral_dim_inds(0),non_lateral_dim_inds(1),:,:) = (/ tofloat(layer_out) /)
         end if
      end if

      increment_dim_inds(non_lateral_dim_inds, var_in_dimsizes(0:non_lateral_rank-1), non_lateral_loops_done)
   end do
end
