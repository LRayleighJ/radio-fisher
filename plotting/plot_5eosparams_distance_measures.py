#!/usr/bin/python
"""
Make a triangle plot for EOS/dark energy params.
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from units import *
from mpi4py import MPI

import os
import euclid

USE_DETF_PLANCK_PRIOR = True # If False, use Euclid prior instead

cosmo = rf.experiments.cosmo

# FIXME
names = ['cexptL_bao', 'cexptL_bao_rsd', 'cexptL_bao_pkshift', 'cexptL_bao_vol', 'cexptL_bao_allap', 'cexptL_bao_all']
labels = ['BAO only', 'BAO + RSD', 'BAO + P(k) shift', 'BAO + Volume', 'BAO + AP', 'All']
colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['c', '#FFEA28'],
            ['m', '#F09B9B'],
            ['k', '#B1C9FD'],
            ['#5B9C0A', '#BAE484'],
            ['#FFB928', '#FFEA28'] ]

scale_idx = 1 # Index of experiment to use as reference for setting the x,y scales
nsigma = 3.4 # No. of sigma (of reference experiment 1D marginal) to plot out to

# Set-up triangle plot
Nparam = 6 # No. of parameters
fig = P.figure()
axes = [[fig.add_subplot(Nparam, Nparam, (j+1) + i*Nparam) for i in range(j, Nparam)] for j in range(Nparam)]

# Fixed width and height for each subplot
w = 1.0 / (Nparam+1.)
h = 1.0 / (Nparam+1.)
l0 = 0.1
b0 = 0.1

# Prepare to save 1D marginals
params_1d = []; params_lbls = []

# Loop though rf.experiments.
_k = list(range(len(names)))[::-1] # Reverse order of rf.experiments.
for k in _k:
    root = "output/" + names[k]
    
    print("-"*50)
    print(names[k])
    print("-"*50)

    # Load cosmo fns.
    dat = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T
    zc, Hc, dAc, Dc, fc = dat
    zs, Hs, dAs, Ds, fs = np.genfromtxt(root+"-cosmofns-smooth.dat").T
    kc = np.genfromtxt(root+"-fisher-kc.dat").T
    
    # Load Fisher matrices as fn. of z
    Nbins = zc.size
    F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(Nbins)]
    
    # EOS FISHER MATRIX
    # Actually, (aperp, apar) are (D_A, H)
    pnames = ['A', 'b_HI', 'Tb', 'sigma_NL', 'sigma8', 'n_s', 'f', 'aperp', 'apar', 
             'omegak', 'omegaDE', 'w0', 'wa', 'h', 'gamma'] #, 'Mnu']
    pnames += ["pk%d" % i for i in range(kc.size)]
    
    zfns = [1,]
    excl = [2,  6,7,8, ] # 14
    excl  += [i for i in range(len(pnames)) if "pk" in pnames[i]]
    
    F, lbls = rf.combined_fisher_matrix( F_list,
                                                expand=zfns, names=pnames,
                                                exclude=excl )
    # Apply Planck prior
    if USE_DETF_PLANCK_PRIOR:
        # DETF Planck prior
        print("*** Using DETF Planck prior ***")
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
        Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    else:
        # Euclid Planck prior
        print("*** Using Euclid (Mukherjee) Planck prior ***")
        l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
        Fe = euclid.planck_prior_full
        F_eucl = euclid.euclid_to_rf(Fe, cosmo)
        Fpl, lbls = rf.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)
    
    # Add Planck H_0 prior
    #if 'H_0' in labels[k]:
    #    ph = lbls.index('h')
    #    Fpl[ph, ph] += 1./(0.012)**2.
    
    # Invert matrices
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    params_1d.append(np.sqrt(np.diag(cov_pl)))
    params_lbls.append(lbls)
    
    # Set which parameters are going into the triangle plot
    params = ['h', 'omegaDE', 'omegak', 'w0', 'wa', 'gamma'][::-1]
    label = ["$h$", "$\Omega_\mathrm{DE}$", "$\Omega_K$", "$w_0$", "$w_a$", "$\gamma$"][::-1]
    
    fid = [ cosmo['h'], cosmo['omega_lambda_0'], 0., cosmo['w0'], cosmo['wa'], cosmo['gamma'] ][::-1]
    
    # Loop through rows, columns, repositioning plots
    # i is column, j is row
    for j in range(Nparam):
        for i in range(Nparam-j):
            ax = axes[j][i]
            
            # Hide tick labels for subplots that aren't on the main x,y axes
            if j != 0:
                for tick in ax.xaxis.get_major_ticks():
                    tick.label1.set_visible(False)
            if i != 0:
                for tick in ax.yaxis.get_major_ticks():
                    tick.label1.set_visible(False)
            
            # Fiducial values
            ii = Nparam - i - 1
            x = fid[ii] #rf.experiments.cosmo['w0']
            y = fid[j] #rf.experiments.cosmo['wa']
            p1 = lbls.index(params[ii])
            p2 = lbls.index(params[j])
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            
            # Plot ellipse *or* 1D
            if p1 != p2:
                # Plot contours
                ww, hh, ang, alpha = rf.ellipse_for_fisher_params(
                                                      p1, p2, None, Finv=cov_pl)
                ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*ww, 
                            height=alpha[kk]*hh, angle=ang, fc='none', 
                            ec=colours[k][0], lw=1.5, alpha=1.) for kk in [0,]]
                for e in ellipses: ax.add_patch(e)
                
                # Centroid and axis scale
                if k == scale_idx:
                    sig1 = np.sqrt(cov_pl[p1,p1])
                    sig2 = np.sqrt(cov_pl[p2,p2])
                    ax.plot(x, y, 'kx')
                    ax.set_xlim((x-nsigma*sig1, x+nsigma*sig1))
                    ax.set_ylim((y-nsigma*sig2, y+nsigma*sig2))
            else:
                sig = np.sqrt(cov_pl[p1,p1])
                xx = np.linspace(x-20.*sig, x+20.*sig, 4000)
                yy = 1./np.sqrt(2.*np.pi*sig**2.) * np.exp(-0.5 * ((xx-x)/sig)**2.)
                yy /= np.max(yy)
                ax.plot(xx, yy, ls='solid', color=colours[k][0], lw=1.5)
                
                # Match x scale, and hide y ticks
                if k == scale_idx:
                    ax.set_xlim((x-nsigma*sig, x+nsigma*sig))
                ax.tick_params(axis='y', which='both', left='off', right='off')
            
            # Set position of subplot
            pos = ax.get_position().get_points()
            ax.set_position([l0+w*i, b0+h*j, w, h])
            
            if j == 0:
                ax.set_xlabel(label[ii], fontdict={'fontsize':'20'}, labelpad=20.)
            #if i == Nparam-j-1: ax.set_title(label[ii], fontdict={'fontsize':'20'})
            if i == 0:
                ax.set_ylabel(label[j], fontdict={'fontsize':'20'}, labelpad=20.)
                ax.get_yaxis().set_label_coords(-0.3,0.5)

# Add legend
labels = [labels[k] for k in range(len(labels))]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in range(len(labels))]

P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'xx-large'}, bbox_to_anchor=(-0.15, -0.2, 1, 1))

# Set every other label to be invisible
# Decide to trim either even/odd labels for each plot on x/y-axis
row_step = [0, 0, 0, 2, 2, 1]
col_step = [1, 2, 2, 0, 1, -1]

fontsize = 12.
for j in range(Nparam):
    for i in range(Nparam-j):
        ax = axes[j][i]
        ii = 0
        for tick in ax.xaxis.get_major_ticks():
            if j != 0: tick.label1.set_visible(False)
            if j == 0 and (ii%2 == row_step[i]): tick.label1.set_visible(False)
            tick.label1.set_fontsize(fontsize)
            ii += 1
        ii = 0
        for tick in ax.yaxis.get_major_ticks():
            if i != 0: tick.label1.set_visible(False)
            if i == 0 and (ii%2 == col_step[j]): tick.label1.set_visible(False)
            if i == 0 and col_step[j] == -1: tick.label1.set_visible(False)
            tick.label1.set_fontsize(fontsize)
            ii += 1

params = ['h', 'omega_b', 'omegaDE', 'n_s', 'sigma8']

"""
print [names[kk] for kk in _k]
for p in params:
    idxs = [params_lbls[i].index(p) for i in range(3)]
    print "%9s: %5.5f %5.5f %5.5f" % (p, params_1d[0][idxs[0]], params_1d[1][idxs[1]], params_1d[2][idxs[2]])
"""


# Set size and save
P.gcf().set_size_inches(16.5,10.5)
#P.savefig('pub-6params-eos.pdf', dpi=100)
P.show()
