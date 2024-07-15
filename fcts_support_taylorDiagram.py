#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
# EDITED FROM https://gist.github.com/ycopin/3342888 -------> HAS TO ACKNOWLEDGE IT!

Taylor diagram (Taylor, 2001) implementation.
Note: If you have found these software useful for your research, I would appreciate an acknowledgment.
"""

__version__ = "Time-stamp: <2018-12-06 11:43:41 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as fa
import mpl_toolkits.axisartist.grid_finder as gf
import numpy as np
import regionmask as regionmask
import seaborn as sns  # # for colors
from matplotlib import cm
from matplotlib.projections import PolarAxes

CB_color_cycle = sns.color_palette( 'colorblind', n_colors=10000 )
from fcts_support_select import *

# preparing letters for subplots
list_letters = [
    "$a$",
    "$b$",
    "$c$",
    "$d$",
    "$e$",
    "$f$",
    "$g$",
    "$h$",
    "$i$",
    "$j$",
    "$k$",
    "$l$",
    "$m$",
    "$n$",
    "$o$",
    "$p$",
    "$q$",
    "$r$",
    "$s$",
    "$t$",
    "$u$",
    "$v$",
    "$w$",
    "$x$",
    "$y$",
    "$z$",
    "$\\alpha$",
    "$\\beta$",
    "$\\gamma$",
    "$\\delta$",
    "$\\epsilon$",
    "$\\zeta$",
    "$\\eta$",
    "$\\theta$",
    "$\\iota$",
    "$\\kappa$",
    "$\\lambda$",
    "$\\mu$",
    "$\\mu$",
    "$\\mu$",
    "$\\mu$",
    "$\\nu$",
    "$\\xi$",
    "$\\pi$",
    "$\\rho$",
    "$\\sigma$",
    "$\\tau$",
    "$\\phi$",
    "$\\chi$",
    "$\\psi$",
    "$\\omega$"
]

# for maps inside
ar6 = regionmask.defined_regions.ar6.land



#==================================================================================================================================
class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd, threshold_std, threshold_coerr, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False, fact_sizes=1):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """


        self.refstd = refstd # Reference standard deviation
        self.fact_sizes = fact_sizes

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = gf.FixedLocator(tlocs)    # Positions
        tf1 = gf.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = fa.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = fa.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # "Angle axis"
        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].major_ticklabels.set_size(fontsize=6*fact_sizes)
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")
        ax.axis["top"].label.set_size(fontsize=8*fact_sizes)
        
        # "X axis"
        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")
        ax.axis["left"].label.set_size(fontsize=8*fact_sizes)
        ax.axis["left"].major_ticklabels.set_size(fontsize=6*fact_sizes)
        
        # "Y-axis"
        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction( "bottom" if extend else "left")
        ax.axis["right"].major_ticklabels.set_size(fontsize=6*fact_sizes)

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=8*fact_sizes, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # adding thresholds
        t = np.linspace(0, self.tmax * (1 - threshold_coerr) * 4 / (1+extend))
        r1 = np.zeros_like(t) + self.refstd * (1 - threshold_std)
        r2 = np.zeros_like(t) + self.refstd * (1 + threshold_std)
        self.ax.fill_between(t, r1, r2, alpha=0.3, facecolor='black', edgecolor=None, label='_')
        
        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


def do_taylor( data_obs, data_esm, labels, threshold_std, threshold_coerr, lat_gp, lon_gp, fact_sizes=1, ncols_lgd=1, sorted_colors=False ):
    # preparation of observations
    refstd = np.std(data_obs.values, ddof=1) # Reference standard deviation
    
    # preparation of model data: interpolating data with non-365 days time format
    data_esm2, dico_stddev, dico_corrcoef = {}, {}, {}
    for esm in data_esm:
        data_esm2[esm], dico_stddev[esm], dico_corrcoef[esm] = compare_seasonalities( data_obs=data_obs, data_esm=data_esm )

    # selection of models according to given criterias
    selected = []
    for esm in data_esm:
        if (np.abs( dico_stddev[esm] - refstd ) <= threshold_std * refstd) and (dico_corrcoef[esm] >= threshold_coerr):
            selected.append( esm )
        
    # preparation of figure
    figsize = (10*fact_sizes, 4*fact_sizes)
    fontsize = 10*fact_sizes
    if sorted_colors:
        colors = cm.Reds(np.linspace(0.4, 1, len(data_esm)))# jet, Reds, RdYlGn, viridis
        #colors = cm.Reds(np.linspace(0.4, 1, len(data_esm)))# jet, Reds, RdYlGn, viridis
        vals = np.array( [[(dico_stddev[esm] - refstd) ** 2, (1 - dico_corrcoef[esm])**2] for esm in data_esm] )
        values = np.sum( vals / np.max(vals, 0), axis=1 )
        order_colors = values.argsort()[::-1]
    else:
        #colors = CB_color_cycle
        colors = cm.viridis(np.linspace(0, 1, len(data_esm)))# jet, Reds, RdYlGn, viridis
        order_colors = np.arange(len(data_esm))

    # figure
    fig = plt.figure(figsize=figsize)
    #--------------------
    # left side: evolutions
    ax1 = fig.add_subplot(1, 2, 1)

    # adding observations
    ax1.plot( data_obs.dayofyear.values, data_obs, color='black', lw=3, label='obs.', zorder=len(data_esm)+1)

    # adding models
    for i, esm in enumerate(data_esm.keys()):
        if esm in selected:
            ax1.plot(data_obs.dayofyear, data_esm2[esm], c=colors[order_colors[i]], label=esm, ls='-' )
        else:
            ax1.plot(data_obs.dayofyear, data_esm2[esm], c=colors[order_colors[i]], label=esm, ls=':' )
    ax1.legend(numpoints=1, prop=dict(size=0.7*fontsize), ncol=ncols_lgd, loc='center', bbox_to_anchor=(0.5,-0.4+0.060*(fact_sizes-1)))
    _ = plt.xticks( size=6*fact_sizes )
    _ = plt.yticks( size=6*fact_sizes )
    plt.xlabel(labels['X'], fontsize=8*fact_sizes)
    plt.ylabel(labels['Y'], fontsize=8*fact_sizes)
    ax1.grid()
    ax1.set_xlim( data_obs.dayofyear.min(), data_obs.dayofyear.max() )
    #--------------------


    #--------------------
    # adding map inside timeseries
    subax = add_subplot_axes(fig=fig, ax=ax1, rect=[0.0115,0.615,0.4,0.5], facecolor='w', proj=ccrs.Robinson()) #x0,y0,D_x,(D_y)
    ar6.plot(ax=subax, add_label=False, label='abbrev', projection=ccrs.Robinson(), line_kws={'lw':0.5*fact_sizes, 'color':'0.35'})
    pos_robinson = ccrs.Robinson().transform_point(lon_gp, lat_gp, ccrs.Geodetic())
    subax.scatter( pos_robinson[0], pos_robinson[1], marker='x', color='red', edgecolor=None, s=80*fact_sizes, lw=2*fact_sizes ) # 50 *
    subax.set_global()
    #--------------------


    #--------------------
    # right side: Taylor diagram
    s = (int( 10 * np.max( [np.abs(1 - dico_stddev[esm] / refstd) for esm in dico_stddev] ) ) + 1) / 10
    dia = TaylorDiagram(refstd, fig=fig, rect=122, label="Reference", srange=(1-s, 1+s), fact_sizes=fact_sizes, threshold_std=threshold_std, threshold_coerr=threshold_coerr)

    # Add the models to Taylor diagram
    for i, esm in enumerate(data_esm.keys()):
        #dia.add_sample(stddev, corrcoef, marker='$%d$' % (i+1), ms=fontsize*8/10, ls='', mfc=colors[order_colors[i]], mec=colors[order_colors[i]], label=esm) # numbers
        dia.add_sample(dico_stddev[esm], dico_corrcoef[esm], marker=list_letters[i], ms=fontsize*5/10, ls='', mfc=colors[order_colors[i]], mec=colors[order_colors[i]], label=esm) # letters

    # Add grid
    dia.add_grid()

    # Add RMS contours, and label them
    contours = dia.add_contours(colors='0.5')
    plt.clabel(contours, inline=1, fontsize=0.6*fontsize, fmt='%.2f')

    # add a legend for Taylor's diagram
    dia._ax.legend(dia.samplePoints, [ p.get_label() for p in dia.samplePoints ], numpoints=1, prop=dict(size=0.7*fontsize), ncol=ncols_lgd, loc='center', bbox_to_anchor=(0.5,-0.4+0.060*(fact_sizes-1)))
    #--------------------

    return dia, fig, refstd, dico_stddev, dico_corrcoef, selected
#==================================================================================================================================




#==================================================================================================================================
def add_subplot_axes(fig, ax, rect, facecolor='w', proj=ccrs.Robinson()):
    '''
    Insert a subplot inside another subplot at the given positions. Normally, one could use "inset_locator.inset_axes", but it doesnt work with projection Robinson.
    
        fig : figure where to anchor the new subplot
        ax : subplot where to anchor the new subplot
        rect : [X_corner_bottom_left, Y_corner_bottom_left, width, height]
        facecolor : color of the subplot
        proj : projection of ax
        
    returns:
        subax : subplot within the subplot
    '''
    
    # preparing subplot
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    
    # actually defining the subplot
    subax = fig.add_axes( [x,y,width,height], facecolor=facecolor, projection=proj)
    
    # defining sizes
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


#==================================================================================================================================

