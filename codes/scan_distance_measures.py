#!/usr/bin/python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a given 
experiment, scanning through different sets of distance measures (BAO, RSD etc.)
"""
import numpy as np
import radiofisher as rf
from mpi4py import MPI
from radiofisher import experiments
from radiofisher.units import *
import sys, copy

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Set-up experiment parameters
################################################################################

# Load cosmology and experimental settings
e = experiments
cosmo = experiments.cosmo

expts = [e.exptS, e.exptM, e.exptL]
names = ['exptS_paper', 'aexptM_paper', 'exptL_paper']

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    k = int(sys.argv[1])
else:
    raise IndexError("Need to specify ID for experiment.")
if myid == 0:
    print("="*50)
    print("Survey:", names[k])
    print("="*50)

# Default distance measures to use
DEFAULT_USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    False,    # D(z) constraint on f(z)
  'alpha_all':         False,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False,
  'alpha_rsd_shift':   False,
  'alpha_bao_shift':   False,
  'alpha_pk_shift':    False
}

# List of distance measure settings to change for each run
dm_names = ['dm_bao', 'dm_bao_rsd', 'dm_bao_pk', 'dm_vol', 'dm_all']
dm = [ {'alpha_bao_shift':True},
       {'alpha_bao_shift':True, 'alpha_rsd_shift':True},
       {'alpha_bao_shift':True, 'alpha_pk_shift':True},
       {'alpha_bao_shift':True, 'alpha_volume':True},
       {'alpha_all':True},
     ]

# Tweak settings depending on chosen experiment
cv_limited = False
expts[k]['mode'] = "dish"
if names[k][0] == "i": expts[k]['mode'] = "idish"
if names[k][0] == "c": expts[k]['mode'] = "combined"
if names[k][0] == "y": expts[k]['mode'] = "icyl"
if names[k][0] == "f": expts[k]['mode'] = "paf"
if names[k][0] == "t": expts[k]['mode'] = "ipaf"
if names[k][0] == "a": expts[k]['mode'] = "iaa"
expt = expts[k]

# Define redshift bins
expt_zbins = rf.overlapping_expts(expt)
#zs, zc = rf.zbins_equal_spaced(expt_zbins, dz=0.1)
#zs, zc = rf.zbins_const_dr(expt_zbins, cosmo, bins=14)
zs, zc = rf.zbins_const_dnu(expt_zbins, cosmo, dnu=60.)

# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns = rf.background_evolution_splines(cosmo)
cosmo['mnu'] = 0.
if cosmo['mnu'] != 0.:
    # Massive neutrinos
    mnu_str = "mnu%03d" % (cosmo['mnu']*100.)
    fname_pk = "cache_pk_%s.dat" % mnu_str
    fname_nu = "cache_%s" % mnu_str
    survey_name += mnu_str; root += mnu_str
    
    cosmo = rf.load_power_spectrum(cosmo, fname_pk, comm=comm)
    Neff_fn = rf.deriv_neutrinos(cosmo, fname_nu, Neff=cosmo['N_eff'], comm=comm)
else:
    # Normal operation (no massive neutrinos or non-Gaussianity)
    cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", comm=comm)
    massive_nu_fn = None

# Non-Gaussianity
#transfer_fn = rf.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None

# Effective no. neutrinos, N_eff
#Neff_fn = rf.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)
Neff_fn = None

# MG/scale-dep. bias switches
switches = []

H, r, D, f = cosmo_fns

################################################################################
# Store cosmological functions
################################################################################

# Save list of scanned parameter values
froot = "output/%s_dm" % names[k]

# Store values of cosmological functions
if myid == 0:
    # Calculate cosmo fns. at redshift bin centroids and save
    _H = H(zc)
    _dA = r(zc) / (1. + np.array(zc))
    _D = D(zc)
    _f = f(zc)
    np.savetxt(froot+"-cosmofns-zc.dat", np.column_stack((zc, _H, _dA, _D, _f)))
    
    # Calculate cosmo fns. as smooth fns. of z and save
    zz = np.linspace(0., 1.05*np.max(zc), 1000)
    _H = H(zz)
    _dA = r(zz) / (1. + zz)
    _D = D(zz)
    _f = f(zz)
    np.savetxt(froot+"-cosmofns-smooth.dat", np.column_stack((zz, _H, _dA, _D, _f)) )

# Precompute derivs for all processes
eos_derivs = rf.eos_fisher_matrix_derivs(cosmo, cosmo_fns)

################################################################################
# Loop through parameter values, then redshift bins
################################################################################

for v in range(len(dm_names)):
    # For each value of experimental parameter, do survey forecast
    survey_name = names[k]
    root = "output/%s_%s" % (names[k], dm_names[v])
    
    # Update list of distance measures to use
    print("-"*50)
    print(dm_names[v])
    print(dm[v])
    print(root)
    print("-"*50)
    expt['use'] = copy.copy(DEFAULT_USE)
    expt['use'].update(dm[v])
    
    # Loop through redshift bins
    for i in range(zs.size-1):
        if i % size != myid:
          continue
        
        print(">>> %2d working on redshift bin %2d -- z = %3.3f" % (myid, i, zc[i]))
        
        # Calculate effective experimental params. in the case of overlapping expts.
        expt_eff = rf.overlapping_expts(expt, zs[i], zs[i+1])
        
        # Calculate basic Fisher matrix
        # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
        F_pk, kc, binning_info, paramnames = rf.fisher( 
                                             zs[i], zs[i+1], cosmo, expt_eff, 
                                             cosmo_fns=cosmo_fns,
                                             transfer_fn=transfer_fn,
                                             massive_nu_fn=massive_nu_fn,
                                             Neff_fn=Neff_fn,
                                             return_pk=True,
                                             cv_limited=cv_limited, 
                                             switches=switches,
                                             kbins=kbins )
        
        # Expand Fisher matrix with EOS parameters
        F_eos, paramnames = rf.expand_fisher_matrix(zc[i], eos_derivs, F_pk, 
                                                    names=paramnames, exclude=[])
        
        # Expand Fisher matrix for H(z), dA(z)
        # Replace aperp with dA(zi), using product rule. aperp(z) = dA(fid,z) / dA(z)
        # (And convert dA to Gpc, to help with the numerics)
        paramnames[paramnames.index('aperp')] = 'DA'
        da = r(zc[i]) / (1. + zc[i]) / 1000. # Gpc
        F_eos[7,:] *= -1. / da
        F_eos[:,7] *= -1. / da
        
        # Replace apar with H(zi)/100, using product rule. apar(z) = H(z) / H(fid,z)
        paramnames[paramnames.index('apar')] = 'H'
        F_eos[8,:] *= 1. / H(zc[i]) * 100.
        F_eos[:,8] *= 1. / H(zc[i]) * 100.
        
        # Save Fisher matrix and k bins
        np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos, header=" ".join(paramnames))
        if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc)
        
        # Save P(k) rebinning info
        np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) )
        np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']) )
        np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']) )
        np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]) )

comm.barrier()
