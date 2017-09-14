;
; gen_grids_and_weights.ncl
;
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

pi = 3.14159265358979323846d0

begin
;----------------------------------------------------------------------
;-- setup
;----------------------------------------------------------------------
   datestr = systemfunc("date +%Y%m%d")
   if .not.ismissing(getenv("GRID_FILE_DATESTR"))
       datestr = getenv("GRID_FILE_DATESTR")
   end if
   
   user = getenv("USER")
   grid_files_outdir = "/glade/p/work/"+user+"/regrid/grid_files"
   if .not.ismissing(getenv("GRID_FILES_OUTDIR"))
       grid_files_outdir = getenv("GRID_FILES_OUTDIR")
   end if
   
   inputdata = "/glade/p/cesmdata/cseg/inputdata/ocn/pop"
   input_templates_dir = "vert_grids_input"

;----------------------------------------------------------------------
;-- gen POP grids
;----------------------------------------------------------------------
   print("-------------------------------------")
   print("generating POP grids")

   grid_cnt = 6

   grid_info := new((/ grid_cnt, 5 /), string)
   grid_info(0,:) = (/ "gx3v7",   "dipole",  "horiz_grid_20030806.ieeer8", "topography_20100105.ieeei4", "region_mask_20090831.ieeei4" /)
   grid_info(1,:) = (/ "gx1v6",   "dipole",  "horiz_grid_20010402.ieeer8", "topography_20090204.ieeei4", "region_mask_20090205.ieeei4" /)
   grid_info(2,:) = (/ "gx1v7",   "dipole",  "horiz_grid_20010402.ieeer8", "topography_20151008.ieeei4", "region_mask_20151008.ieeei4" /)
   grid_info(3,:) = (/ "tx1v1",   "tripole", "horiz_grid_20050510.ieeer8", "topography_20050510.ieeei4", "region_mask_20050510.ieeei4" /)
   grid_info(4,:) = (/ "tx0.1v2", "tripole", "horiz_grid_200709.ieeer8",   "topography_200709.ieeei4",   "region_mask_20080702.ieeei4" /)
   grid_info(5,:) = (/ "tx0.1v3_62lev", "tripole", "/glade/p/work/mclong/regrid/62_level/horiz_grid_200709.ieeer8",   "/glade/p/work/mclong/regrid/62_level/tx0.1v3_62lev_grid.nc",  "/glade/p/work/mclong/regrid/62_level/region_mask_no_caspian.nc" /)

   dims := new((/ grid_cnt, 2 /), integer)
   dims(0,:) = (/  116,  100 /)
   dims(1,:) = (/  384,  320 /)
   dims(2,:) = (/  384,  320 /)
   dims(3,:) = (/  240,  360 /)
   dims(4,:) = (/ 2400, 3600 /)
   dims(5,:) = (/ 2400, 3600 /)

   setfileoption("bin", "ReadByteOrder", "BigEndian")

   dst_grids = new((/ grid_cnt /), string)
   do grid_ind = 0, grid_cnt-1

      grid_name = grid_info(grid_ind,0)
      print("grid_name = "+grid_name)

      grid_out_fname = grid_files_outdir+"/POP_"+grid_name+"_SCRIP_"+datestr+".nc"
      dst_grids(grid_ind) = "POP_"+grid_name

      ;
      ; get file names (use local or full-path-specified files if present)
      ;
      grid_dir = ""
      if .not. fileexists(grid_info(grid_ind,2)) then
          grid_dir = inputdata+"/"+grid_name+"/grid/"
      end if
      horiz_grid_fname = grid_dir+grid_info(grid_ind,2)
      topography_fname = grid_dir+grid_info(grid_ind,3)
      region_mask_fname = grid_dir+grid_info(grid_ind,4)

      nlat = dims(grid_ind, 0)
      nlon = dims(grid_ind, 1)

      ltripole = grid_info(grid_ind,1) .eq. "tripole"
      print("ltripole = "+ltripole)

      ;
      ; read fields from horiz_grid_fname
      ;
      print("reading horiz_grid from "+horiz_grid_fname)
      if isfilepresent(horiz_grid_fname) then
          grid_file = addfile(horiz_grid_fname,"r")
          ULAT := grid_file->ULAT * (pi/180.)
          ULON := grid_file->ULON * (pi/180.)
          HTN  := grid_file->HTN
          HTE  := grid_file->HTE
          delete(grid_file)
      else
          grid_file_contents := cbinread(horiz_grid_fname, (/ 7, nlat, nlon /), "double")
          ULAT := grid_file_contents(0,:,:)
          ULON := grid_file_contents(1,:,:)
          HTN  := grid_file_contents(2,:,:)
          HTE  := grid_file_contents(3,:,:)
      end if

      ;
      ; construct coordinates of grid centers
      ;
      print("constructing coordinates of grid centers")

      TLAT := new((/ nlat, nlon /), double)
      TLON := new((/ nlat, nlon /), double)

      do j = 1, nlat-1
         jm1 = j-1
         do i = 0, nlon-1
            im1 = mod(i-1+nlon, nlon)

            tmp = cos(ULAT(jm1,im1))
            xsw = cos(ULON(jm1,im1))*tmp
            ysw = sin(ULON(jm1,im1))*tmp
            zsw = sin(ULAT(jm1,im1))

            tmp = cos(ULAT(jm1,i))
            xse = cos(ULON(jm1,i))*tmp
            yse = sin(ULON(jm1,i))*tmp
            zse = sin(ULAT(jm1,i))

            tmp = cos(ULAT(j,im1))
            xnw = cos(ULON(j,im1))*tmp
            ynw = sin(ULON(j,im1))*tmp
            znw = sin(ULAT(j,im1))

            tmp = cos(ULAT(j,i))
            xne = cos(ULON(j,i))*tmp
            yne = sin(ULON(j,i))*tmp
            zne = sin(ULAT(j,i))

            xc = 0.25 * (xsw + xse + xnw + xne)
            yc = 0.25 * (ysw + yse + ynw + yne)
            zc = 0.25 * (zsw + zse + znw + zne)

            r = sqrt(xc*xc + yc*yc + zc*zc)

            TLAT(j,i) = asin(zc/r)
            TLON(j,i) = atan2(yc, xc)
         end do
      end do

      ; generate bottom row vals by extrapolating from 2 rows above
      TLAT(0,:) = TLAT(1,:) - (TLAT(2,:) - TLAT(1,:))
      TLON(0,:) = TLON(1,:) - (TLON(2,:) - TLON(1,:))

      ;
      ; construct coordinates of grid centers
      ;
      print("constructing DXT, DYT, TAREA")

      DXT := new((/ nlat, nlon /), double)
      DXT(1:nlat-1,:) = 0.5 * (HTN(0:nlat-2,:) + HTN(1:nlat-1,:))
      ; generate bottom row vals by extrapolating HTN from 2 rows above
      DXT(0,:) = 0.5 * (2*HTN(0,:)-HTN(1,:) + HTN(0,:))

      DYT := new((/ nlat, nlon /), double)
      DYT(:,1:nlon-1) = 0.5 * (HTE(:,0:nlon-2) + HTE(:,1:nlon-1))
      ; assume zonal periodic bc
      DYT(:,0) = 0.5 * (HTE(:,nlon-1) + HTE(:,0))

      TAREA := DXT * DYT

      ;
      ; construct degree versions of coordinates
      ;

      ULATd := (180.0 / pi) * ULAT
      ULONd := (180.0 / pi) * ULON
      TLATd := (180.0 / pi) * TLAT
      TLONd := (180.0 / pi) * TLON

      ;
      ; construct coordinates of grid corners
      ; storage order of corners is counter-clockwise from NE: NE, NW, SW, SE
      ;
      print("constructing coordinates of grid corners")

      GridCornerLat2d := new((/ nlat, nlon, 4 /), double)
      ; NE corner
      GridCornerLat2d(:,:,0) = ULATd
      ; NW corner (copy from NE corner of column to the left, assume zonal periodic bc)
      GridCornerLat2d(:,1:nlon-1,1) = GridCornerLat2d(:,0:nlon-2,0)
      GridCornerLat2d(:,0,1) = GridCornerLat2d(:,nlon-1,0)
      ; SW corner (copy from NW corner of row below, bottom row is extrapolated from 2 rows above)
      GridCornerLat2d(1:nlat-1,:,2) = GridCornerLat2d(0:nlat-2,:,1)
      GridCornerLat2d(0,:,2) = GridCornerLat2d(1,:,2) - (GridCornerLat2d(2,:,2) - GridCornerLat2d(1,:,2))
      ; SE corner (copy from NE corner of row below, bottom row is extrapolated from 2 rows above)
      GridCornerLat2d(1:nlat-1,:,3) = GridCornerLat2d(0:nlat-2,:,0)
      GridCornerLat2d(0,:,3) = GridCornerLat2d(1,:,3) - (GridCornerLat2d(2,:,3) - GridCornerLat2d(1,:,3))

      GridCornerLon2d := new((/ nlat, nlon, 4 /), double)
      ; NE corner
      GridCornerLon2d(:,:,0) = ULONd
      ; NW corner (copy from NE corner of column to the left, assume zonal periodic bc)
      GridCornerLon2d(:,1:nlon-1,1) = GridCornerLon2d(:,0:nlon-2,0)
      GridCornerLon2d(:,0,1) = GridCornerLon2d(:,nlon-1,0)
      ; SW corner (copy from NW corner of row below, bottom row is extrapolated from 2 rows above
      GridCornerLon2d(1:nlat-1,:,2) = GridCornerLon2d(0:nlat-2,:,1)
      GridCornerLon2d(0,:,2) = GridCornerLon2d(1,:,2) - (GridCornerLon2d(2,:,2) - GridCornerLon2d(1,:,2))
      ; SE corner (copy from NE corner of row below, bottom row is extrapolated from 2 rows above
      GridCornerLon2d(1:nlat-1,:,3) = GridCornerLon2d(0:nlat-2,:,0)
      GridCornerLon2d(0,:,3) = GridCornerLon2d(1,:,3) - (GridCornerLon2d(2,:,3) - GridCornerLon2d(1,:,3))

      ;
      ; flatten corner coordinate arrays
      ;
      GridCornerLat := reshape(GridCornerLat2d, (/ nlat*nlon, 4 /))
      GridCornerLon := reshape(GridCornerLon2d, (/ nlat*nlon, 4 /))

      ;
      ; generate mask from topography file
      ;
      print("reading KMT from "+topography_fname)
      if isfilepresent(topography_fname) then
          file_topo = addfile(topography_fname,"r")
          KMT = file_topo->KMT
          delete(file_topo)
      else
          KMT := cbinread(topography_fname, (/ nlat, nlon /), "integer")
      end if
      GridMask := where(KMT .gt. 0, 1, 0)

      ;
      ; read REGION_MASK, for inclusion in generated grid file
      ;
      print("reading REGION_MASK from "+region_mask_fname)
      if isfilepresent(region_mask_fname) then
          file_rmask = addfile(region_mask_fname,"r")
          REGION_MASK = file_rmask->REGION_MASK
          delete(file_rmask)
      else
          REGION_MASK := cbinread(region_mask_fname, (/ nlat, nlon /), "integer")
      end if

      ;
      ; call curvilinear_to_SCRIP function
      ;

      opt = True
      opt@ForceOverwrite = True
      opt@Debug = True
      opt@GridCornerLat = GridCornerLat
      opt@GridCornerLon = GridCornerLon
      opt@GridMask = GridMask
      print("creating grid file "+grid_out_fname)
      curvilinear_to_SCRIP(grid_out_fname, TLATd, TLONd, opt)
      delete(opt)

      ;
      ; add two-dimensional grid variables to generated file
      ;
      ULATd!0 = "nlat"
      ULATd!1 = "nlon"
      ULATd@units = "degrees_north"

      ULONd!0 = "nlat"
      ULONd!1 = "nlon"
      ULONd@units = "degrees_east"

      TLATd!0 = "nlat"
      TLATd!1 = "nlon"
      TLATd@units = "degrees_north"

      TLONd!0 = "nlat"
      TLONd!1 = "nlon"
      TLONd@units = "degrees_east"

      DXT!0 = "nlat"
      DXT!1 = "nlon"
      DXT@units = "cm"

      DYT!0 = "nlat"
      DYT!1 = "nlon"
      DYT@units = "cm"

      TAREA!0 = "nlat"
      TAREA!1 = "nlon"
      TAREA@units = "cm^2"

      KMT!0 = "nlat"
      KMT!1 = "nlon"

      REGION_MASK!0 = "nlat"
      REGION_MASK!1 = "nlon"

      print("appending 2D grid variables to grid file")
      f = addfile(grid_out_fname, "w")
      f@horiz_grid_fname  = horiz_grid_fname
      f@topography_fname  = topography_fname
      f@region_mask_fname = region_mask_fname
      f->ULATd = ULATd
      f->ULONd = ULONd
      f->TLATd = TLATd
      f->TLONd = TLONd
      f->DXT = DXT
      f->DYT = DYT
      f->TAREA = TAREA
      f->KMT = KMT
      f->REGION_MASK = REGION_MASK
      if (ltripole) then
         f@ltripole = 1
      else
         f@ltripole = 0
      end if
      delete(f)
   end do
   print("")

;----------------------------------------------------------------------
;-- gen latlon grids
;----------------------------------------------------------------------
   print("-------------------------------------")
   print("generating vertical grids")

   grid_cnt = 6

   grid_info := new((/ grid_cnt, 2 /), string)
   grid_info(0,:) = (/ "gx3v7",   "gx3v7_vert_grid"   /)
   grid_info(1,:) = (/ "gx1v6",   "gx1v6_vert_grid"   /)
   grid_info(2,:) = (/ "gx1v7",   "gx1v6_vert_grid"   /)
   grid_info(3,:) = (/ "tx1v1",   "tx1v1_vert_grid"   /)
   grid_info(4,:) = (/ "tx0.1v2", "tx0.1v2_vert_grid" /)
   grid_info(5,:) = (/ "tx0.1_km62", "tx0.1_km62_vert_grid" /)

   km_vals := new((/ grid_cnt /), integer)
   km_vals(0) = (/ 60 /)
   km_vals(1) = (/ 60 /)
   km_vals(2) = (/ 60 /)
   km_vals(3) = (/ 40 /)
   km_vals(4) = (/ 42 /)
   km_vals(5) = (/ 62 /)

   do grid_ind = 0, grid_cnt-1
      grid_name = grid_info(grid_ind,0)
      print("grid_name = "+grid_name)

      vert_grid_file = input_templates_dir+"/"+grid_info(grid_ind,1)

      vert_grid_file_out = grid_files_outdir+"/POP_"+grid_name+"_vert_"+datestr+".nc"

      row_cnt = numAsciiCol(vert_grid_file)
      km = km_vals(grid_ind)

      tmp := asciiread(vert_grid_file, (/ km, row_cnt /), "double")
      dz := tmp(:,0) / 100.0
      delete(dz@_FillValue)
      dz!0         = "depth"
      dz@long_name = "layer thickness"
      dz@units     = "m"

      depth_edges := new(km+1, double)
      delete(depth_edges@_FillValue)
      depth_edges(0) = 0.0
      depth_edges(1:km) = cumsum(dz, 0)

      depth := depth_edges(0:km-1) + 0.5 * dz

      depth!0         = "depth"
      depth&depth     = depth
      depth@long_name = "depth from surface to midpoint of layer"
      depth@units     = "m"
      depth@positive  = "down"

      dz&depth = depth

      f = addfile(vert_grid_file_out, "c")
      f->depth = depth
      f->dz = dz
      f@vert_grid_file = vert_grid_file
      delete(f)
      
      print("created "+vert_grid_file_out)
   end do
   print("")

   exit
end
status_exit(1)