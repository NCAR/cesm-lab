C23456789012345678901234567890123456789012345678901234567890123456789012
C NCLFORTSTART
      subroutine remap_z(imt, jmt, klev_in, klev_out, tlev, KMT, z_edge,
     +                   VAR_IN, NEW_Z, new_z_edge, msv,
     +                   THICKNESS, VAR_OUT)

      integer imt, jmt, klev_in, klev_out, tlev
      integer KMT(imt,jmt)
      real z_edge(*)
      real VAR_IN(imt,jmt,klev_in,tlev)
      real NEW_Z(imt,jmt,klev_in,tlev)
      real new_z_edge(*)
      real THICKNESS(imt,jmt,klev_out,tlev)
      real VAR_OUT(imt,jmt,klev_out,tlev)
C NCLEND
Cf2py integer depend(KMT) :: imt=shape(KMT,0),jmt=shape(KMT,1)
Cf2py integer depend(VAR_IN) :: klev_in=shape(VAR_IN,2), tlev=shape(VAR_IN,3)
Cf2py intent(in) klev_out
Cf2py intent(in) KMT
Cf2py intent(in) z_edge
Cf2py intent(in) VAR_IN
Cf2py intent(in) NEW_Z
Cf2py intent(in) new_z_edge
Cf2py intent(in) msv
Cf2py intent(out) THICKNESS
Cf2py intent(out) VAR_OUT

      integer i, j, kin, kout, l
      real msv
      real, allocatable :: dz(:), dzw(:), zt(:), dNEW_Z(:), dVAR_IN(:)
      real h, hlo, hhi, hmid


      allocate(dz(klev_in), dzw(klev_in), zt(klev_in))
      allocate(dNEW_Z(klev_in), dVAR_IN(klev_in))

      do kin = 1, klev_in
        dz(kin) = z_edge(kin+1) - z_edge(kin)
        if (kin .gt. 1) dzw(kin) = 0.5*(dz(kin) + dz(kin-1))
        zt(kin) = 0.5*(z_edge(kin+1) + z_edge(kin))
      end do

      do l = 1, tlev
      do i = 1, imt
      do j = 1, jmt
C       write(6,*) 'i=', i,', j=', j,', KMT=', KMT(i,j)
        if (KMT(i,j) .eq. 0) then
          THICKNESS(i,j,:,l) = 0.0
          VAR_OUT(i,j,:,l) = msv
        else
C
C         construct dVAR_IN and dNEW_Z at z_edge coordinates
C
          do kin = 2, KMT(i,j)
            dVAR_IN(kin) = VAR_IN(i,j,kin,l) - VAR_IN(i,j,kin-1,l)
            dNEW_Z(kin)  = NEW_Z(i,j,kin,l)  - NEW_Z(i,j,kin-1,l)
          end do

          do kout = 1, klev_out
            THICKNESS(i,j,kout,l) = 0.0
            VAR_OUT(i,j,kout,l) = 0.0
C
C           shallow half of shallowest incoming layer
C
            kin = 1
            if ((new_z_edge(kout) .le. NEW_Z(i,j,kin,l)) .and.
     +          (NEW_Z(i,j,kin,l) .lt. new_z_edge(kout+1))) then
              h = 0.5 * dz(kin)
              THICKNESS(i,j,kout,l) = THICKNESS(i,j,kout,l) + h
              VAR_OUT(i,j,kout,l) = VAR_OUT(i,j,kout,l) +
     +                                h * VAR_IN(i,j,kin,l)
            end if

            do kin = 1, KMT(i,j)
C
C           shallow half of general case incoming layers
C
              if (kin .gt. 1) then
C               NEW_Z values over shallower half of cell are (h=0..dz(kin)/2)
C               NEW_Z(zt(kin)-h) = NEW_Z(i,j,kin,l) - h / dzw(kin) * dNEW_Z(kin)
                hlo = (NEW_Z(i,j,kin,l) - new_z_edge(kout)) *
     +                   dzw(kin) / dNEW_Z(kin)
                hlo = min(max(hlo,0.0),0.5*dz(kin))
                hhi = (NEW_Z(i,j,kin,l) - new_z_edge(kout+1)) *
     +                   dzw(kin) / dNEW_Z(kin)
                hhi = min(max(hhi,0.0),0.5*dz(kin))
                h = abs(hhi-hlo)
                THICKNESS(i,j,kout,l) = THICKNESS(i,j,kout,l) + h
C               VAR_IN values over shallower half of cell are (h=0..dz(kin)/2)
C               VAR_IN(zt(kin)-h) = VAR_IN(i,j,kin,l) - h / dzw(kin) * dVAR_IN(kin)
C               evaluate this at hmid
                hmid = 0.5*(hlo+hhi)
                VAR_OUT(i,j,kout,l) = VAR_OUT(i,j,kout,l) + h *
     +              (VAR_IN(i,j,kin,l) - hmid / dzw(kin) * dVAR_IN(kin))
              end if
C
C           deeper half of general case incoming layers
C
              if (kin .lt. KMT(i,j)) then
C               NEW_Z values over deeper half of cell are (h=0..dz(kin)/2)
C               NEW_Z(zt(kin)+h) = NEW_Z(i,j,kin,l) + h / dzw(kin+1) * dNEW_Z(kin+1)
                hlo = (new_z_edge(kout) - NEW_Z(i,j,kin,l)) *
     +                   dzw(kin+1) / dNEW_Z(kin+1)
                hlo = min(max(hlo,0.0),0.5*dz(kin))
                hhi = (new_z_edge(kout+1) - NEW_Z(i,j,kin,l)) *
     +                   dzw(kin+1) / dNEW_Z(kin+1)
                hhi = min(max(hhi,0.0),0.5*dz(kin))
                h = abs(hhi-hlo)
                THICKNESS(i,j,kout,l) = THICKNESS(i,j,kout,l) + h
C               VAR_IN values over deeper half of cell are (h=0..dz(kin)/2)
C               VAR_IN(zt(kin)+h) = VAR_IN(i,j,kin,l) + h / dzw(kin+1) * dVAR_IN(kin+1)
C               evaluate this at hmid
                hmid = 0.5*(hlo+hhi)
                VAR_OUT(i,j,kout,l) = VAR_OUT(i,j,kout,l) + h *
     +          (VAR_IN(i,j,kin,l) + hmid / dzw(kin+1) * dVAR_IN(kin+1))
              end if
            end do

            kin = KMT(i,j)
            if ((new_z_edge(kout) .le. NEW_Z(i,j,kin,l)) .and.
     +          (NEW_Z(i,j,kin,l) .lt. new_z_edge(kout+1))) then
              h = 0.5 * dz(kin)
              THICKNESS(i,j,kout,l) = THICKNESS(i,j,kout,l) + h
              VAR_OUT(i,j,kout,l) = VAR_OUT(i,j,kout,l) +
     +                                h * VAR_IN(i,j,kin,l)
            end if

            if (THICKNESS(i,j,kout,l) .gt. 0.0) then
              VAR_OUT(i,j,kout,l) = VAR_OUT(i,j,kout,l) /
     +                              THICKNESS(i,j,kout,l)
            else
              VAR_OUT(i,j,kout,l) = msv
            end if

          end do
        end if
      end do
      end do
      end do

      deallocate(dNEW_Z, dVAR_IN)
      deallocate(dz, dzw, zt)

      return
      end
