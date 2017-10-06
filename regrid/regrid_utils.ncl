;
; regrid_utils.ncl
;

undef("debug_print")
procedure debug_print(info:string, val, opt:logical)
begin
   if (opt@Debug) then
      print(info+val)
   end if
end


;
; ensure that varname in f is on the src_grid specified by opt@wgtFile
;
undef("verify_var_src_grid")
procedure verify_var_src_grid(f:file, varname:string, f_wgtFile:file, opt:logical)
local f_wgtFile, yc_a, xc_a, src_grid_type, vardims, rank, y, x, yc_var, xc_var
begin
   yc_a = f_wgtFile->yc_a
   xc_a = f_wgtFile->xc_a
   xc_a = where(xc_a .ge. 180., xc_a - 360., xc_a)

   src_grid_type = str_get_field(opt@src_grid, 1, "_")
   debug_print("src_grid_type = ", src_grid_type, opt)

   ;
   ; construct yc and xc arrays for varname
   ;
   if (src_grid_type .eq. "latlon" .or. src_grid_type .eq. "rectilinear") then
       vardims = getfilevardims(f, varname)
       rank = dimsizes(vardims)
       y = f->$vardims(rank-2)$
       x = f->$vardims(rank-1)$
       x = where(x .ge. 180., x - 360., x)
       yc_var = ndtooned(conform_dims((/ dimsizes(y), dimsizes(x) /), y, 0))
       xc_var = ndtooned(conform_dims((/ dimsizes(y), dimsizes(x) /), x, 1))
   else if (src_grid_type .eq. "POP") then
       yc_var = ndtooned(f->TLAT)
       xc_var = ndtooned(f->TLONG)
       xc_var = where(xc_var .ge. 180., xc_var - 360., xc_var)
   end if
   end if

   ;
   ; ensure that yc_var and xc_var were constructed
   ;
   if (.not. isvar("yc_var") .or. .not. isvar("xc_var")) then
      print("src_grid_type = "+src_grid_type+" not supported in verify_var_src_grid")
      exit
   end if

   ;
   ; verify that yc_var and xc_var agree with yc_a and xc_a from wgtFile
   ;
   mismatch = False
   print("dims yc_var: "+dimsizes(yc_var))
   print("dims yc_a: "+dimsizes(yc_a)) 
   if (any(abs(yc_var - yc_a) .gt. 1e-5)) then
      print("yc_a mismatch for "+varname)
      print("src_grid = "+opt@src_grid)
      mismatch = True
   end if
   if (any(abs(xc_var - xc_a) .gt. 1e-5)) then
      print("xc_a mismatch for "+varname)
      print("max mismatch = "+max(abs(xc_var - xc_a)))
      print("src_grid = "+opt@src_grid)
      mismatch = True
   end if

   if (mismatch) then
       status_exit(1)
   end if
end


;
; increment_non_lateral_dim_inds
;
undef("increment_dim_inds")
procedure increment_dim_inds(dim_inds[*]:integer, dimextents[*]:integer, done:logical)
local rank, dimptr
begin
   rank = dimsizes(dim_inds)
   dimptr = rank-1
   do while (dimptr .ge. 0)
      dim_inds(dimptr) = dim_inds(dimptr) + 1
      if (dim_inds(dimptr) .lt. dimextents(dimptr)) then
         done = False
         return
      else
         dim_inds(dimptr) = 0
         dimptr = dimptr - 1
      end if
   end do
   done = True
end


;
; fill_layer
;
undef("fill_layer")
procedure fill_layer(layer_vals:numeric, fillopt:string)
begin
   if (fillopt .eq. "none") then
      return
   end if
   if (fillopt .eq. "zeros") then
      layer_vals = where(ismissing(layer_vals), 0.0, layer_vals)
      return
   end if
   print("unknown fillopt = "+fillopt)
   status_exit(1)
end


;
; append POP specific grid vars to a file
;
undef("append_POP_grid_vars")
procedure append_POP_grid_vars(f_out:file, grid_file:string)
local f_grid_file, KMT
begin
   f_grid_file = addfile(grid_file, "r")
   f_out->TLAT = f_grid_file->TLATd
   f_out->TLONG = f_grid_file->TLONd

   TAREA = f_grid_file->TAREA
   TAREA@coordinates = "TLONG TLAT"
   TAREA@long_name = "area of T cells"
   f_out->TAREA = TAREA

   KMT = f_grid_file->KMT
   KMT@coordinates = "TLONG TLAT"
   KMT@long_name = "k Index of Deepest Grid Cell on T Grid"
   f_out->KMT = KMT

   REGION_MASK = f_grid_file->REGION_MASK
   REGION_MASK@coordinates = "TLONG TLAT"
   REGION_MASK@long_name = "basin index number"
   f_out->REGION_MASK = REGION_MASK
end
