#!/usr/bin/python
"""
Process EOS Fisher matrices and plot P(k).
"""

import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
from units import *
from mpi4py import MPI

import os
import euclid

cosmo = rf.experiments.cosmo

names = ["gSKA2", ]
#colours = ['#CC0000', '#ED5F21', '#FAE300', '#5B9C0A', '#1619A1', '#56129F', '#990A9C']
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C'] # DETF/F/M/S
labels = ['Full SKA (galaxy survey)', ]

names = ["FASThighz_hrx_opt_n_5","MeerKAT10hrs_hrx_opt_n_5"]# ["FAST_hrx_opt","MeerKATb1_hrx_opt","MeerKATb2_hrx_opt","yTIANLAI_hrx_opt"]
colours = ['#CC0000', '#1619A1', '#5B9C0A', '#990A9C',"#FF1493","#FF69B4","#DAA520","#BDB76B"] # DETF/F/M/S
labels = ['FAST 1050-1150MHz','MeerKAT 11hrs','MeerKAT Lband','Tianlai']# ['DETF IV', 'Facility', 'Stage II', 'Stage I']

# Get f_bao(k) function
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

# Fiducial value and plotting
fig = P.figure()
axes = [fig.add_subplot(111), ]

for k in range(len(names)):
    root = "output/" + names[k]

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    z, H, dA, D, f = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T

    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI',]
    excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'fs8', 'bs8', 'gamma', 'N_eff']
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    
    # Just do the simplest thing for P(k) and get 1/sqrt(F)
    cov = [np.sqrt(1. / np.diag(F)[lbls.index(lbl)]) for lbl in lbls if "pk" in lbl]
    cov = np.array(cov)
    pk = cosmo['pk_nobao'](kc) * (1. + fbao(kc))
    
    # Plot errorbars
    yup, ydn = rf.fix_log_plot(pk, cov)
    
    # Fix for PDF
    yup[np.where(yup > 1e1)] = 1e1
    ydn[np.where(ydn > 1e1)] = 1e1
    axes[k].errorbar( kc, fbao(kc), yerr=[ydn, yup], color=colours[k], ls='none', 
                      lw=1.8, capthick=1.8, label=names[k], ms='.' )

    # Plot f_bao(k)
    kk = np.logspace(-3., 1., 2000)
    axes[k].plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6)
    
    # Set limits
    axes[k].set_xscale('log')
    axes[k].set_xlim((4e-3, 1e0))
    axes[k].set_ylim((-0.13, 0.13))
    #axes[k].set_ylim((-0.08, 0.08))
    
    axes[k].text( 0.105, 0.105, labels[k], fontsize=16, 
                  bbox={'facecolor':'white', 'alpha':1., 'edgecolor':'white',
                        'pad':15.} )

"""
# Move subplots
# pos = [[x0, y0], [x1, y1]]
l0 = 0.15
b0 = 0.1
ww = 0.75
hh = 0.8 / 4.
for i in range(len(names))[::-1]:
    axes[i].set_position([l0, b0 + hh*i, ww, hh])
"""

# Resize labels/ticks
for i in range(len(axes)):
    ax = axes[i]
    
    ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=8.)
    ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5)
    
    if i != 0: ax.tick_params(axis='x', which='major', labelbottom='off')

axes[0].set_xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$", fontdict={'fontsize':'xx-large'}, labelpad=10.)
#ax.set_ylabel(r"$P(k)$", fontdict={'fontsize':'20'})

P.tight_layout()

# Set size
#P.gcf().set_size_inches(8.5,12.)
P.gcf().set_size_inches(8.4, 6.8)
P.savefig('ska-fbao-ska2.pdf', transparent=True)
P.show()
