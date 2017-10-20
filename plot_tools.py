#! /usr/bin/env python
import numpy as np
import xarray as xr
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
import matplotlib.gridspec as gridspec
import cartopy
import cartopy.crs as ccrs

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif']=['DejaVu Sans','Fira Sans OT','Bitstream Vera Sans']
plt.rcParams['font.size'] = 9
plt.rcParams['mathtext.default']='regular'

#-------------------------------------------------------------------------------
#-- class
#-------------------------------------------------------------------------------

class MidPointNorm(colors.Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        colors.Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = np.ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result


    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = np.ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def nice_levels(cmin,cmax,max_steps,outside=False):
    '''
    Return nice contour levels
    outside indicates whether the contour should be inside or outside the bounds
    '''
    import sys

    table = [ 1.0,2.0,2.5,4.0,5.0,
              10.0,20.0,25.0,40.0,50.0,
              100.0,200.0,250.0,400.0,500.0]

    am2 = 0.
    ax2 = 0.
    npts = 15

    d = 10.**(np.floor(np.log10(cmax-cmin))-2.)

    u = sys.float_info.max
    step_size = sys.float_info.max

    if outside:
        for i in range(npts):
            t = table[i] * d
            am1 = np.floor(cmin/t) * t
            ax1 = np.ceil(cmax/t) * t

            if (i == npts-1 and step_size == u) or ( (t <= step_size) and ((ax1-am1)/t <= (max_steps-1)) ):
                step_size = t
                ax2 = ax1
                am2 = am1
    else:
        for i in range(npts):
            t = table[i] * d
            am1 = np.ceil(cmin/t) * t
            ax1 = np.floor(cmax/t) * t

            if (i == npts-1 and step_size == u) or ( (t <= step_size) and ((ax1-am1)/t <= (max_steps-1)) ):
                step_size = t
                ax2 = ax1
                am2 = am1

    min_out = am2
    max_out = ax2

    return np.arange(min_out,max_out+step_size,step_size)

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def plotgrid(fig,gs,ax,plot_dim,gridspec_kwargs_in={}):

    gridspec_kwargs = {'hspace' : 0.05,
                       'wspace' : 0.05,
                       'left' : 0.,
                       'right' : 0.87,
                       'bottom' : 0.,
                       'top' : 1.}

    if gridspec_kwargs_in:
        gridspec_kwargs.update(gridspec_kwargs_in)

    hspace = gridspec_kwargs['hspace']
    wspace = gridspec_kwargs['wspace']
    right = gridspec_kwargs['right']
    left = gridspec_kwargs['left']
    top = gridspec_kwargs['top']
    bottom = gridspec_kwargs['bottom']

    nrow,ncol = plot_dim

    #-- get starting figure size
    fgsz = fig.get_size_inches()

    #-- adjust plots
    gs.update(**gridspec_kwargs)
    figW_f,figH_f = (right-left), (top-bottom)

    #-- determine axis size in real display units
    bbox = ax[-1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    axW_d,axH_d = bbox.width, bbox.height

    #-- with the Robinson projection, we trust the width, not the height
    aspect = np.diff(ax[0].projection.x_limits)/np.diff(ax[0].projection.y_limits)[0]
    axH_d = axW_d/aspect

    print('initial plot size (display units): %.2f,%.2f'%(axW_d,axH_d))

    total_axW_d = (ncol + wspace * (ncol-1)) * axW_d
    total_axH_d = (nrow + hspace * (nrow-1)) * axH_d

    figW_in = total_axW_d/figW_f
    figH_in = total_axH_d/figH_f

    #-- no change? go home
    if all([old == new for old,new in zip(fgsz,[figW_in,figH_in])]):
        return fgsz

    print 'Adjusting fig size:'
    print '\t(W,H in): %.2f,%.2f --> %.2f,%.2f'%(fgsz[0],fgsz[1],
                                                 figW_in,figH_in)
    #-- reset figure size
    fig.set_size_inches(figW_in,figH_in)

    gs.update(**gridspec_kwargs)

    bbox = ax[-1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axW_d, axH_d = bbox.width, bbox.height
    print('new plot size: %.2f,%.2f'%(axW_d,axH_d))

    return [figW_in,figH_in]

#---------------------------------------------------------
#--- function
#---------------------------------------------------------

def rasterize_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                       savefig_kw={}):
    """Save a figure with raster and vector components

    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.

    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig

    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized

    Note: does not work correctly with round=True in Basemap

    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line

    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> rasterize_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """
    from inspect import getmembers, isclass

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        rasterize_list = list(rasterize_list)

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = isinstance(item, matplotlib.contour.QuadContourSet)

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


#---------------------------------------------------------
#--- function
#---------------------------------------------------------

def adjust_pop_grid(tlon,tlat,field):
    nj = tlon.shape[0]
    ni = tlon.shape[1]
    xL = ni/2 - 1
    xR = xL + ni

    tlon = np.where(np.greater_equal(tlon,min(tlon[:,0])),tlon-360.,tlon)
    lon  = np.concatenate((tlon,tlon+360.),1)
    lon = lon[:,xL:xR]

    if ni == 320:
        lon[367:-3,0] = lon[367:-3,0]+360.
    lon = lon - 360.
    lon = np.hstack((lon,lon[:,0:1]+360.))
    if ni == 320:
        lon[367:,-1] = lon[367:,-1] - 360.

    #-- periodicity
    lat  = np.concatenate((tlat,tlat),1)
    lat = lat[:,xL:xR]
    lat = np.hstack((lat,lat[:,0:1]))

    field = np.ma.concatenate((field,field),1)
    field = field[:,xL:xR]
    field = np.ma.hstack((field,field[:,0:1]))
    return lon,lat,field


#-------------------------------------------------------------------------------
#-- class
#-------------------------------------------------------------------------------

class contour_label_format(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def canvas_map_contour_overlay(lon,lat,z,
                               contour_specs,
                               units,
                               fig,
                               gridspec,
                               row,col):


    #-- make canvas
    ax = fig.add_subplot(gridspec[row,col],projection=ccrs.Robinson(central_longitude=305.0))
    ax.set_global()

    #-- make filled contours
    cf = ax.contourf(lon,lat,z,
                     transform=ccrs.PlateCarree(),
                     **contour_specs)
    #-- rasterize
    zorder = 0
    for contour_level in cf.collections:
        contour_level.set_zorder(zorder)
        contour_level.set_rasterized(True)

    #-- add contour lines
    cs = ax.contour(lon,lat,z,
                    colors='k',
                    levels = contour_specs['levels'],
                    linewidths = 0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=len(cf.collections)+10)
    cs.levels = [contour_label_format(val) for val in cs.levels]
    fmt = '%r'
    #-- add contour labels
    lb = plt.clabel(cs, fontsize=6,
                   inline = True,
                   fmt=fmt)

    #-- add land mask
    land = ax.add_feature(
        cartopy.feature.NaturalEarthFeature('physical','land','110m',
                                            edgecolor='face',
                                            facecolor='black'))
    #land = ax.add_feature(cartopy.feature.LAND, zorder=100,
    #                       edgecolor='black', facecolor='black')

    #-- add colorbar
    i = 0
    while True:
        i += 1
        try:
            gridspec[i]
        except:
            break
    len_gs = i
    if len_gs == 1:
        shrink_factor = 0.75
    else:
        shrink_factor = 0.75

    cb = fig.colorbar(cf,ax = ax,
                      ticks = contour_specs['levels'],
                      orientation = 'vertical',
                      shrink = shrink_factor)
    cb.ax.set_title(units)

    return {'ax':ax,'cf':cf,'cs':cs,'lb':lb,'cb':cb}

#-------------------------------------------------------------------------------
#-- function
#-------------------------------------------------------------------------------

def canvas_full_depth_section(x,y,field,
                              gridspec_spec,
                              gridspec_index,
                              fig = None,
                              contour_specs = {},
                              colorbar_specs = {},
                              xlim = [-78,70]):
    if fig is None:
        fig = plt.gcf()

    #-- some default resources for plotting
    contour_specs_def = {}

    #-- update intent(in) specs with defaults, but don't overwrite
    contour_specs.update({k:v for k,v in contour_specs_def.items()
                          if k not in contour_specs})

    #-- some default resources for colorbar
    colorbar_specs_def = {'drawedges' :True}

    #-- update intent(in) specs with defaults, but don't overwrite
    colorbar_specs.update({k:v for k,v in colorbar_specs_def.items()
                          if k not in colorbar_specs})

    ax = [None]*2
    gs = gridspec.GridSpecFromSubplotSpec(100, 1,
                                          subplot_spec=gridspec_spec[gridspec_index])

    ax[0] = plt.Subplot(fig,gs[:45,0])
    fig.add_subplot(ax[0]) #add_axes([0.1,0.51,0.7,0.35])
    ax[1] = plt.Subplot(fig,gs[46:,0])
    fig.add_subplot(ax[1]) #fig.add_axes([0.1,0.1,0.7,0.4])

    cf = [None]*2
    cf[0] = ax[0].contourf(x, y, field, **contour_specs)
    cf[1] = ax[1].contourf(x, y, field, **contour_specs)

    ax[0].set_ylim([1000.,0.])
    ax[0].set_yticklabels(np.arange(0,800,200))
    ax[0].xaxis.set_ticks_position('top')

    ax[1].set_ylim([5000.,1000.])
    ax[1].set_xlabel('Latitude [$^\circ$N]')
    ax[1].xaxis.set_ticks_position('bottom')
    ax[1].set_ylabel('Depth [m]')
    ax[1].yaxis.set_label_coords(-0.18, 1.05)

    for axi in ax:
        axi.set_xlim(xlim)
        axi.set_facecolor((0, 0, 0))
        axi.set_xticklabels([])
        axi.minorticks_on()
        axi.tick_params(which='major',direction='out',width=1,  length=6)
        axi.tick_params(which='minor',direction='out',width=0.5,  length=4)


    return {'gs':gs,'ax':ax,'cf':cf}

# TODO: add regrid capability?
#https://github.com/j08lue/pygeo/blob/master/pygeo/regrid.py
