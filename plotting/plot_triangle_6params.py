#!/usr/bin/python
"""
Make a triangle plot for a set of parameters (Fig. 11).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
import os
from radiofisher import euclid

MARGINALISE_OVER_W0WA = True # Whether to fix or marginalise over (w0, wa)

cosmo = rf.experiments.cosmo

colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'] ]
if not MARGINALISE_OVER_W0WA:
    names = ['EuclidRef_paper', 'exptL_paper', 'aexptM_paper']
    labels = ['DETF IV + Planck', 'Facility + Planck', 'Stage II + Planck']
    colours.append(['#5B9C0A', '#BAE484']) # Green
else:
    names = ['FAST_hrx_opt','MeerKATb1_hrx_opt','MeerKATb2_hrx_opt']# ['EuclidRef_paper', 'exptL_paper', 'exptL_paper']
    labels = ['FAST','MeerKATb1','MeerKATb2']# ['DETF IV + Planck', 'Facility + Planck', 'Planck only'] 
    colours.append(['#6B6B6B', '#BDBDBD']) # Grey

scale_idx = 1 # Index of experiment to use as reference for setting the x,y scales
nsigma = 4.2 # No. of sigma (of reference experiment 1D marginal) to plot out to

# Set-up triangle plot
Nparam = 5 # No. of parameters
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
    pnames = rf.load_param_names(root+"-fisher-full-0.dat")
    zfns = ['b_HI', ]
    excl = ['Tb', 'f', 'aperp', 'apar', 'fs8', 'bs8', 'DA', 'H', 'gamma', 'N_eff', 'pk*']
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    # Apply DETF Planck prior
    print("*** Using DETF Planck prior ***")
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    
    if 'Planck only' in labels[k]:
        # Just do Planck, on its own
        lbls = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
        Fpl = F_detf
    
    # Remove w0, wa from Fisher matrix
    if MARGINALISE_OVER_W0WA:
        print("*** Marginalising over (w0, wa) ***")
    else:
        print("*** Fixing (w0, wa) ***")
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], names=lbls,
                                     exclude=['w0', 'wa'] )
    # Exclude Omega_K (flatness prior)
    Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], names=lbls,
                                     exclude=['omegak',] )
    
    # Invert matrices
    cov_pl = np.linalg.inv(Fpl)
    
    # Store 1D marginals
    params_1d.append(np.sqrt(np.diag(cov_pl)))
    params_lbls.append(lbls)
    
    # Set which parameters are going into the triangle plot
    params = ['h', 'omega_b', 'omegaDE', 'n_s', 'sigma8'][::-1]
    label = ["$h$", "$\omega_b$", "$\Omega_\mathrm{DE}$", "$n_s$", "$\sigma_8$"][::-1]
    fid = [ cosmo['h'], cosmo['omega_b_0']*cosmo['h']**2., 
            cosmo['omega_lambda_0'], cosmo['ns'], cosmo['sigma_8'] ][::-1]
        
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
            x = fid[ii]
            y = fid[j]
            p1 = lbls.index(params[ii])
            p2 = lbls.index(params[j])
            
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            
            # Plot ellipse *or* 1D
            transp = [1., 0.85]
            if p1 != p2:
                # Plot contours
                ww, hh, ang, alpha = rf.ellipse_for_fisher_params(
                                                      p1, p2, None, Finv=cov_pl)
                ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*ww, 
                            height=alpha[kk]*hh, angle=ang, fc=colours[k][kk], 
                            ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
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
if MARGINALISE_OVER_W0WA:
    # Decide to trim either even/odd labels for each plot on x/y-axis
    row_step = [0, 1, 1, 1, 1, 1]
    col_step = [1, 1, 1, 1, 1, -1]
else:
    row_step = [0, 1, 1, 0, 1]
    col_step = [1, 2, 1, 0, -1]
    
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

print([names[kk] for kk in _k])
for p in params:
    idxs = [params_lbls[i].index(p) for i in range(3)]
    print("%9s: %5.5f %5.5f %5.5f" % (p, params_1d[0][idxs[0]], params_1d[1][idxs[1]], params_1d[2][idxs[2]]))


# Set size and save
P.gcf().set_size_inches(16.5,10.5)
if MARGINALISE_OVER_W0WA:
    P.savefig('pub-5params-w0wamarg.pdf', dpi=100)
else:
    P.savefig('fig11-5params.pdf', dpi=100)
P.show()
