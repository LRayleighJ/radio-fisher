#!/usr/bin/python
"""
Plot P(k) constraints on BAO wiggles (Fig. 5).
"""
import numpy as np
import pylab as P
import radiofisher as rf
import matplotlib.patches
import matplotlib.cm
import os
from radiofisher import euclid
import matplotlib.pyplot as plt
cosmo = rf.experiments.cosmo

gen_data = False


expt_id = 96

D = 300 # meters
tpix = 24 #seconds
Sarea_list = np.array([100,200,500,1000,2000,5000,10000,20000])
ttot_list = 2*Sarea_list*tpix/(1*0.21/D*180/np.pi)**2/(3600)
lbladd_list = [1,2,3,4,5,6]
times_list = [1,2,3,5,10,20]

print(Sarea_list)
print(ttot_list)

if gen_data:
    for i in range(len(Sarea_list)):
        commandir = "python /home/zerui603/work/bao21cm-master/full_expt_origin.py %d %.2f %.2f"%(expt_id, Sarea_list[i],ttot_list[i],)
        os.system(commandir)

print(Sarea_list)
print(ttot_list)


survey_name="FASThighz_hrx_opt"

names = [survey_name+"_%d"%(sa,) for sa in Sarea_list]


# Get f_bao(k) function
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

for k in range(len(names)):
    print(names[k])
    title = "FAST 1000-1150MHz %d deg2, drift scan for 2 times"%(Sarea_list[k],)
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
    

    print(np.mean(fbao(kc)))

    kk = np.logspace(-3., 1., 2000)

    plt.figure()
    plt.errorbar( kc, fbao(kc), yerr=[ydn, yup], ls='none', lw=1.8, capthick=1.8, label=names[k] )
    plt.plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6)
    plt.xscale('log')
    plt.xlim((4e-3, 1e0))
    plt.ylim((-0.13, 0.13))
    plt.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$")
    plt.title(title)
    plt.savefig("test_%s_pk.pdf"%(names[k]))
    plt.close()


# comparation

compare_name_list = ["FASThighz_hrx_opt_5000","MeerKATb1_hrx_opt","MeerKATb2_hrx_opt"]
lbl_list = ["FAST 1000-1150MHz 5000$deg^2$","MeerKAT UHFband 580-1015MHz","MeerKAT Lband 900-1420MHz"]
elw=[1,2,3,4]
plt.figure()
for k in range(len(compare_name_list)):
    root = "output/" + compare_name_list[k]

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
    

    print(np.mean(fbao(kc)))

    kk = np.logspace(-3., 1., 2000)

    
    plt.errorbar( kc, fbao(kc), yerr=[ydn, yup], ls='none', lw=1.8, capthick=1.8, elinewidth=elw[k], label=lbl_list[k],alpha=0.5)
    if k==0:
        plt.plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6,c="black")
plt.axhline(0,ls="--",lw=1.8, alpha=0.6,c="black")
plt.xscale('log')
plt.xlim((2e-2, 5e-1))
plt.ylim((-0.17, 0.17))
plt.xlabel(r"$k \,[\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P(k)-P_{smooth}(k)$")
plt.legend()
plt.savefig("test_compare_pk.pdf")
plt.close()


