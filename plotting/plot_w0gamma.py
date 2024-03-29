#!/usr/bin/python
"""
Plot 2D constraints on (w0, gamma) for various setups (Fig. 19) (SKA chapters).
"""
import numpy as np
import pylab as P
from rfwrapper import rf
import matplotlib.patches
import matplotlib.cm
import matplotlib.ticker
from radiofisher import euclid
import copy

USE_DETF_PLANCK_PRIOR = True
MARGINALISE_CURVATURE = True # Marginalise over Omega_K
MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

cosmo = rf.experiments.cosmo

colours = [ ['#1619A1', '#B1C9FD'], 
            ['#FFA728', '#F8D24B'],
            ['#6B6B6B', '#BDBDBD'],
            ['#CC0000', '#F09B9B'], 
            ['#5B9C0A', '#BAE484'], 
            ['#990A9C', '#F4BAF5'], 
            ['#CC0000', '#F09B9B'], 
            ['#FFB928', '#FFEA28'],
            ['#6B6B6B', '#BDBDBD'],
            ['#5B9C0A', '#BAE484'],
            ['#6B6B6B', '#BDBDBD'],
            ['#6B6B6B', '#BDBDBD'],
            ['#6B6B6B', '#BDBDBD'] ]

#--------------------------------------------------------------
# Bull et al. 2014 IM paper: All distance measures (w0, gamma)
#--------------------------------------------------------------

fig_name = "fig19-w0gamma.pdf"
names = ["FAST_hrx_opt",]# ['EuclidRef_paper', 'exptL_paper', 'exptL_paper']
labels = ['DETF IV + Planck', 'Facility + Planck', 'Combined + Planck']

ADD_EUCLID = [False, False, True]
ADD_PLANCK = True
EXCLUDE = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']

legend_order = [0,1,2]
colours = [ ['#CC0000', '#F09B9B'],
            ['#1619A1', '#B1C9FD'],
            ['#6B6B6B', '#BDBDBD'],
            ['#FFA728', '#F8D24B'], ]
XLIM = (0.37, 0.72)
YLIM = (-1.215, -0.76)


"""
#-------------------------------------------------
# SKA galaxy survey chapter: BAO-only (w0, gamma)
#-------------------------------------------------
names = ['gSKASURASKAP_baoonly', 'SKA1MIDfull2_baoonly', 'EuclidRef_baoonly', 
         'gSKA2_baoonly']
labels = ['SKA1-SUR (gal.)', 'SKA1-MID B2 (IM)', 'Euclid (gal.)', 'SKA 2 (gal.)']
fig_name = "ska-w0gamma-gal.pdf"

ADD_PLANCK = True
ADD_BOSS = True
EXCLUDE = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8',]

legend_order = [0,1,3,2]
colours = [ ['#1619A1', '#B1C9FD'], 
            ['#FFA728', '#F8D24B'],
            ['#6B6B6B', '#BDBDBD'],
            ['#CC0000', '#F09B9B'], ]
XLIM = (-0.6, 0.6)
YLIM = (-1.24, -0.76)
"""
"""
#----------------------------------------------------
# SKA RSD chapter: All distance measures (w0, gamma)
#----------------------------------------------------

names = ['fSKA1SUR350_25000', 'EuclidRef', 'SKA1MID350_25000', 'gSKA2']
labels = ['SKA1-SUR B1 (IM)', 'Euclid (gal.)', 'SKA1-MID B1 (IM)', 'SKA2 (gal.)']
fig_name = "ska-w0gamma-rsd-planck-boss.pdf"

ADD_PLANCK = True
ADD_BOSS = True
EXCLUDE = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']

legend_order = [2,0,1,3]
colours = [ ['#FFA728', '#F8D24B'],
            ['#6B6B6B', '#BDBDBD'],
            ['#1619A1', '#B1C9FD'],
            ['#CC0000', '#F09B9B'], ]
XLIM = (0.43, 0.7)
YLIM = (-1.17, -0.83)
"""
"""
################################################################################
# Load BOSS
################################################################################

root = "output/" + 'BOSS'
zc = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T[0]
F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(zc.size)]

pnames = rf.load_param_names(root+"-fisher-full-0.dat")
zfns = ['b_HI',]
excl = EXCLUDE
Fboss, lbl_boss = rf.combined_fisher_matrix( F_list, expand=zfns, 
                                             names=pnames, exclude=excl )
# Relabel galaxy bias
for i in range(len(lbl_boss)):
    if "b_HI" in lbl_boss[i]: lbl_boss[i] = "gal%s" % lbl_boss[i]
"""

################################################################################
# Load Euclid
################################################################################

root = "output/" + 'EuclidRef_paper'
zc = np.atleast_2d( np.genfromtxt(root+"-cosmofns-zc.dat") ).T[0]
F_list = [np.genfromtxt(root+"-fisher-full-%d.dat" % i) for i in range(zc.size)]

pnames = rf.load_param_names(root+"-fisher-full-0.dat")
zfns = ['b_HI',]
excl = EXCLUDE
Feuc, lbl_euc = rf.combined_fisher_matrix( F_list, expand=zfns, 
                                           names=pnames, exclude=excl )
# Relabel galaxy bias
for i in range(len(lbl_euc)):
    if "b_HI" in lbl_euc[i]: lbl_euc[i] = "gal%s" % lbl_euc[i]



################################################################################
# Fiducial value and plotting
fig = P.figure()
ax = fig.add_subplot(111)

_k = list(range(len(names))) #[::-1]
for k in _k:
    root = "output/" + names[k]

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
    excl = EXCLUDE
    F, lbls = rf.combined_fisher_matrix( F_list, expand=zfns, names=pnames,
                                         exclude=excl )
    # Combine with Euclid if requested
    if ADD_EUCLID[k]:
        F, lbls = rf.add_fisher_matrices(F, Feuc, lbls, lbl_euc, expand=True)
    
    # Add Planck prior
    print("*** Using DETF Planck prior ***")
    l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
    F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
    Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
    if not ADD_PLANCK: Fpl = F # Do not add Planck prior
    
    # Decide whether to fix various parameters
    fixed_params = []
    if not MARGINALISE_CURVATURE: fixed_params += ['omegak',]
    if not MARGINALISE_INITIAL_PK: fixed_params += ['n_s', 'sigma8']
    if not MARGINALISE_OMEGAB: fixed_params += ['omega_b',]
    
    if len(fixed_params) > 0:
        Fpl, lbls = rf.combined_fisher_matrix( [Fpl,], expand=[], 
                     names=lbls, exclude=fixed_params )
    
    # Get indices of w0, wa
    pw0 = lbls.index('w0'); pgam = lbls.index('gamma')
    
    print("-"*50)
    print(names[k])
    print("-"*50)
    
    # Invert matrix
    cov_pl = np.linalg.inv(Fpl)
    
    # Print 1D marginals
    print("1D sigma(w_0) =   %3.4f" % np.sqrt(cov_pl[pw0,pw0]))
    print("1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pgam,pgam]))
    print(lbls)
    
    x = rf.experiments.cosmo['gamma']
    y = rf.experiments.cosmo['w0']
    
    # Plot contours for gamma, w0
    transp = [1., 0.85]
    if k == 1: transp = [1., 0.95]
    w, h, ang, alpha = rf.ellipse_for_fisher_params(pgam, pw0, None, Finv=cov_pl)
    ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
                height=alpha[kk]*h, angle=ang, fc=colours[k][kk], 
                ec=colours[k][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
    for e in ellipses: ax.add_patch(e)
    
    # Centroid
    if k == 0: ax.plot(x, y, 'ko')

# Report on what options were used
print("-"*50)
s1 = "Marginalised over Omega_K" if MARGINALISE_CURVATURE else "Fixed Omega_K"
s2 = "Marginalised over ns, sigma8" if MARGINALISE_INITIAL_PK else "Fixed ns, sigma8"
s3 = "Marginalised over Omega_b" if MARGINALISE_OMEGAB else "Fixed Omega_b"
print("NOTE:", s1)
print("NOTE:", s2)
print("NOTE:", s3)

"""
################################################################################
# Add combined constraint for Facility + Euclid

# Relabel galaxy bias from Euclid and sum Facility + Euclid
for i in range(len(lbl1)):
    if "b_HI" in lbl1[i]: lbl1[i] = "gal%s" % lbl1[i]
Fc, lbls = rf.add_fisher_matrices(F1, F2, lbl1, lbl2, expand=True)

# Add Planck prior
l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
Fc, lbls = rf.add_fisher_matrices(Fc, F_detf, lbls, l2, expand=True)
cov_pl = np.linalg.inv(Fc)

# Plot contours for gamma, w0
transp = [1., 0.95]
w, h, ang, alpha = rf.ellipse_for_fisher_params(pgam, pw0, None, Finv=cov_pl)
ellipses = [matplotlib.patches.Ellipse(xy=(x, y), width=alpha[kk]*w, 
            height=alpha[kk]*h, angle=ang, fc=colours[-1][kk], 
            ec=colours[-1][0], lw=1.5, alpha=transp[kk]) for kk in [1,0]]
for e in ellipses: ax.add_patch(e)
labels += ['Combined']

print "\nCOMBINED"
pw0 = lbls.index('w0'); pgam = lbls.index('gamma')
print "1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0])
print "1D sigma(gamma) = %3.4f" % np.sqrt(cov_pl[pgam,pgam])
"""


################################################################################

# Plot datapoints for other theories
ax.plot(0.68, -0.8, 'kD') # DGP
ax.plot(0.4, -0.99, 'kD') # f(r)
#ax.plot(0.48, -1.22, 'kD') # Minimal massive bigravity, arXiv:1404.4061

P.annotate("DGP", xy=(0.68, -0.8), xytext=(0., -20.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center')

P.annotate("f(R)", xy=(0.4, -0.99), xytext=(0., -20.), fontsize='large', 
                       textcoords='offset points', ha='center', va='center')

#P.annotate("Mass. Bigrav.", xy=(0.48, -1.22), xytext=(0., 20.), fontsize='large', 
#                       textcoords='offset points', ha='center', va='center')

################################################################################


# Legend
labels = [labels[k] for k in legend_order]
lines = [ matplotlib.lines.Line2D([0.,], [0.,], lw=8.5, color=colours[k][0], alpha=0.65) for k in legend_order]
#P.gcf().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, bbox_to_anchor=[0.96, 0.95], frameon=False)
P.gca().legend((l for l in lines), (name for name in labels), prop={'size':'large'}, loc='upper left', frameon=False)

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)

ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))

ax.tick_params(axis='both', which='major', labelsize=20, size=8., width=1.5, pad=10.)
ax.tick_params(axis='both', which='minor', labelsize=20, size=5., width=1.5, pad=10.)

ax.set_xlabel(r"$\gamma$", fontdict={'fontsize':'xx-large'}, labelpad=15.)
ax.set_ylabel(r"$w_0$", fontdict={'fontsize':'xx-large'})

# Set size and save
P.gcf().set_size_inches(8.,6.)
P.tight_layout()

print("Output figure:", fig_name)
P.savefig(fig_name, transparent=True)
P.show()
