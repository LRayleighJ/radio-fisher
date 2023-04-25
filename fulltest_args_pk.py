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
from radiofisher.units import *
from radiofisher import euclid

USE_DETF_PLANCK_PRIOR = True
# MARGINALISE_CURVATURE = True # Marginalise over Omega_K
# MARGINALISE_INITIAL_PK = True # Marginalise over n_s, sigma_8
# MARGINALISE_OMEGAB = True # Marginalise over Omega_baryons

get_w0wa_plot = False

cosmo = rf.experiments.cosmo

gen_data = False
clean = False


expt_id = 22

D = 300 # meters
tpix = 24 #seconds

'''
Sarea_list = np.array([100,200,500,1000,2000,5000,10000,20000])
ttot_list = 2*Sarea_list*tpix/(1*0.21/D*180/np.pi)**2/(3600)
lbladd_list = [1,2,3,4,5,6]
times_list = [1,2,3,5,10,20]
'''

# generate according experiments

t_unit = 12*24*HRS_MHZ

args_name = 'ttot'
args_list = np.array([0.1,0.2,0.5,1,2,5,10])*t_unit
lbl_name_list = list(range(len(args_list)))


if gen_data:
    if clean:
        os.system("rm -f /home/zerui603/work/bao21cm-master/output/*.dat")
    for i in range(len(args_list)):
        commandir = "python /home/zerui603/work/bao21cm-master/full_expt_args.py %d %s %.2f %s"%(expt_id, args_name, args_list[i],args_name+str(lbl_name_list[i]),)
        os.system(commandir)
    exit()


# test args

survey_name="FAST_hrx_opt"

# names = [survey_name+"_%d"%(sa,) for sa in Sarea_list]

names = [survey_name+"_"+args_name+str(x) for x in lbl_name_list]


# Get f_bao(k) function
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", force_load=True)
fbao = cosmo['fbao']

w0_list = []
wa_list = []
nu_list = []

dpk_list = []
k_list = []
label_dpk_list = []

for k in range(len(names)):
    print(names[k])
    title = "FAST"#"FAST 1000-1150MHz %d deg2, drift scan for 2 times"%(Sarea_list[k],)
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

    dpk_list.append([yup,ydn])
    k_list.append(kc)
    label_dpk_list.append(r"ttot = %d day"%(args_list[k]/(24*HRS_MHZ),))

    ## get cosmological parameters such as w0 and wa
    if get_w0wa_plot:
        pnames = rf.load_param_names(root+"-fisher-full-0.dat")
        zfns = ['b_HI',]
        #excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'gamma', 'N_eff', 'pk*', 'fs8', 'bs8']
        excl = ['Tb', 'f', 'aperp', 'apar', 'DA', 'H', 'N_eff', 'pk*', 'fs8', 'bs8']
        F, lbls = rf.combined_fisher_matrix( F_list,expand=zfns, names=pnames,exclude=excl )
        if USE_DETF_PLANCK_PRIOR:
            # DETF Planck prior
            # print("*** Using DETF Planck prior ***")
            l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h', 'sigma8']
            F_detf = euclid.detf_to_rf("DETF_PLANCK_FISHER.txt", cosmo, omegab=False)
            Fpl, lbls = rf.add_fisher_matrices(F, F_detf, lbls, l2, expand=True)
        else:
            # Euclid Planck prior
            # print("*** Using Euclid (Mukherjee) Planck prior ***")
            l2 = ['n_s', 'w0', 'wa', 'omega_b', 'omegak', 'omegaDE', 'h']
            Fe = euclid.planck_prior_full
            F_eucl = euclid.euclid_to_rf(Fe, cosmo)
            Fpl, lbls = rf.add_fisher_matrices(F, F_eucl, lbls, l2, expand=True)

        # Get indices of w0, wa
        pw0 = lbls.index('w0'); pwa = lbls.index('wa'); pA = lbls.index('A')
        
        try:
            # Invert matrix
            cov_pl = np.linalg.inv(Fpl)
            # print("1D sigma(w_0) = %3.4f" % np.sqrt(cov_pl[pw0,pw0]))
            # print("1D sigma(w_a) = %3.4f" % np.sqrt(cov_pl[pwa,pwa]))
            w0_value = np.sqrt(cov_pl[pw0,pw0])
            wa_value = np.sqrt(cov_pl[pwa,pwa])

            w0_list.append(w0_value)
            wa_list.append(wa_value)
            nu_list.append(args_list[k])
        except:
            print("singular matrix at nu_max=%.2fMHz"%(args_list[k],))
            continue

plt.figure()
for i in range(len(dpk_list)):
    plt.plot(k_list[i],dpk_list[i][1]+dpk_list[i][0],label = label_dpk_list[i])
plt.xlabel(r"k")
plt.ylabel(r"$\Delta P(k)$")
plt.xlim((2e-2, 2e-1))
plt.ylim((0, 0.2))
plt.xscale('log')
plt.legend()
plt.title(survey_name)
plt.savefig("test_%s_dpk.pdf"%(survey_name,))
plt.close()

# wiggle

color_edge_list = ["#191970","#4682B4", "#3CB371", "#DAA520", "#FF8C00", "#A0522D", "#B22222"]
color_fill_list = ["#9370DB","#87CEFA", "#90EE90", "#FFD700", "#F4A460", "#F4A460", "#F08080"]

plt.figure()
for i in range(len(dpk_list)):
    plt.plot(k_list[i], fbao(k_list[i])+dpk_list[i][0],color=color_edge_list[i],alpha=0.9)
    plt.plot(k_list[i], fbao(k_list[i])-dpk_list[i][1],color=color_edge_list[i],alpha=0.9,label = label_dpk_list[i])
    plt.fill_between(k_list[i], fbao(k_list[i])-dpk_list[i][1], fbao(k_list[i])+dpk_list[i][0],color=color_fill_list[i],alpha=0.2)
plt.plot(kk, fbao(kk), 'k-', lw=1.8, alpha=0.6)
plt.xlabel(r"k")
plt.ylabel(r"$P(k)-P_{smooth}(k)$")
plt.xlim((2e-2, 3e-1))
plt.ylim((-0.2, 0.2))
plt.xscale('log')
plt.legend()
plt.title(r"FAST 1050-1350MHz, $Sarea=1000deg^2$")
plt.savefig("test_%s_pk_merge.pdf"%(survey_name,))
plt.close()



'''
if get_w0wa_plot:
    plt.figure(figsize=[10,4])
    plt.subplot(121)
    plt.plot(args_list, w0_list, marker="x")
    plt.xlabel(r"$\nu_{max}/MHz$, total bandwidth = 300MHz")
    plt.ylabel(r"$\omega_0$")
    plt.subplot(122)
    plt.plot(args_list, wa_list, marker="x")
    plt.xlabel(r'$\nu_{max}/MHz$, total bandwidth = 300MHz')
    plt.ylabel(r"$\omega_a$")
    plt.suptitle("FAST")
    plt.savefig("test_numax_w0wa_%s.pdf"%(survey_name,))
    plt.close()
'''
'''
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


'''