#!/usr/bin/python
"""
Calculate Fisher matrix and P(k) constraints for all redshift bins for a given 
experiment.
"""
import numpy as np
import pylab as P
import radiofisher as rf
from radiofisher import experiments
from radiofisher.units import *
from mpi4py import MPI
import sys
import matplotlib.pyplot as plt

#test

def print_test(x):
    with open("/home/zerui603/work/bao21cm-master/zoutput.txt","a") as f114514:
        print(x, file=f114514)


comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

################################################################################
# Set-up experiment parameters
################################################################################

# Load cosmology and experimental settings
e = experiments
cosmo = experiments.cosmo

# Label experiments with different settings
#EXPT_LABEL = "_mg_Dz_kmg0.1" #"_mnu" #"_mg_Dz" "_baoonly" "_uls"
#EXPT_LABEL = "_mgtest_fnl10" #"_mg_Dz_kmg0.01"

#TTOT = 1e3
#TTOT = 3e3
#TTOT = 5e3
#TTOT = 10e3
#EXPT_LABEL = "_survparams_ttot"

#EXPT_LABEL = "_paper"
#EXPT_LABEL = "_rerun"
#EXPT_LABEL = "_mgD"
#EXPT_LABEL = "_mg"
#EXPT_LABEL = "_mgD_scaledep"
#EXPT_LABEL = "_mg_Axi0.01_kmg0.005"
#EXPT_LABEL = "_nokfg_paper"
#EXPT_LABEL = "_efg12_paper"

KMAXVAL = 0.14
EXPT_LABEL = "_hrx_opt"

#cosmo['A_xi'] = 0.01
#cosmo['logkmg'] = np.log10(0.005)

# A_xi: 0.01 0.1
# logkmg: 0.05, 0.01, 0.005, 0.001

expt_list = [
    ( 'exptS',            e.exptS ),        # 0
    ( 'aexptM',           e.exptM ),        # 1
    ( 'exptL',            e.exptL ),        # 2
    ( 'iexptL',           e.exptL ),        # 3
    ( 'cexptL',           e.exptL ),        # 4
    ( 'GBT',              e.GBT ),          # 5
    ( 'Parkes',           e.Parkes ),       # 6
    ( 'GMRT',             e.GMRT ),         # 7
    ( 'WSRT',             e.WSRT ),         # 8
    ( 'APERTIF',          e.APERTIF ),      # 9
    ( 'VLBA',             e.VLBA ),         # 10
    ( 'JVLA',             e.JVLA ),         # 11
    ( 'iJVLA',            e.JVLA ),         # 12
    ( 'BINGO',            e.BINGO ),        # 13
    ( 'iBAOBAB32',        e.BAOBAB32 ),     # 14
    ( 'iBAOBAB128',       e.BAOBAB128 ),    # 15
    ( 'yCHIME',           e.CHIME ),        # 16
    ( 'iAERA3',           e.AERA3 ),        # 17
    ( 'iMFAA',            e.MFAA ),         # 18
    ( 'yTIANLAIpath',     e.TIANLAIpath ),  # 19
    ( 'yTIANLAI',         e.TIANLAI ),      # 20
    ( 'yTIANLAIband2',    e.TIANLAIband2 ), # 21
    ( 'FAST',             e.FAST ),         # 22
    ( 'KAT7',             e.KAT7 ),         # 23
    ( 'iKAT7',            e.KAT7 ),         # 24
    ( 'cKAT7',            e.KAT7 ),         # 25
    ( 'MeerKATb1',        e.MeerKATb1 ),    # 26
    ( 'iMeerKATb1',       e.MeerKATb1 ),    # 27
    ( 'cMeerKATb1',       e.MeerKATb1 ),    # 28
    ( 'MeerKATb2',        e.MeerKATb2 ),    # 29
    ( 'iMeerKATb2',       e.MeerKATb2 ),    # 30
    ( 'cMeerKATb2',       e.MeerKATb2 ),    # 31
    ( 'ASKAP',            e.ASKAP ),        # 32
    ( 'SKA1MIDbase1',     e.SKA1MIDbase1 ), # 33
    ( 'iSKA1MIDbase1',    e.SKA1MIDbase1 ), # 34
    ( 'cSKA1MIDbase1',    e.SKA1MIDbase1 ), # 35
    ( 'SKA1MIDbase2',     e.SKA1MIDbase2 ), # 36
    ( 'iSKA1MIDbase2',    e.SKA1MIDbase2 ), # 37
    ( 'cSKA1MIDbase2',    e.SKA1MIDbase2 ), # 38
    ( 'SKA1MIDfull1',     e.SKA1MIDfull1 ), # 39
    ( 'iSKA1MIDfull1',    e.SKA1MIDfull1 ), # 40
    ( 'cSKA1MIDfull1',    e.SKA1MIDfull1 ), # 41
    ( 'SKA1MIDfull2',     e.SKA1MIDfull2 ), # 42
    ( 'iSKA1MIDfull2',    e.SKA1MIDfull2 ), # 43
    ( 'cSKA1MIDfull2',    e.SKA1MIDfull2 ), # 44
    ( 'fSKA1SURbase1',    e.SKA1SURbase1 ), # 45
    ( 'fSKA1SURbase2',    e.SKA1SURbase2 ), # 46
    ( 'fSKA1SURfull1',    e.SKA1SURfull1 ), # 47
    ( 'fSKA1SURfull2',    e.SKA1SURfull2 ), # 48
    ( 'exptCV',           e.exptCV ),       # 49
    ( 'GBTHIM',           e.GBTHIM ),       # 50
    ( 'SKA0MID',          e.SKA0MID ),      # 51
    ( 'fSKA0SUR',         e.SKA0SUR ),      # 52
    ( 'SKA1MID900',       e.SKA1MID900 ),   # 53
    ( 'SKA1MID350',       e.SKA1MID350 ),   # 54
    ( 'iSKA1MID900',      e.SKA1MID900 ),   # 55
    ( 'iSKA1MID350',      e.SKA1MID350 ),   # 56
    ( 'fSKA1SUR650',      e.SKA1SUR650 ),   # 57
    ( 'fSKA1SUR350',      e.SKA1SUR350 ),   # 58
    ( 'aSKA1LOW',         e.SKA1LOW ),      # 59
    ( 'SKAMID_PLUS',      e.SKAMID_PLUS ),  # 60
    ( 'SKAMID_PLUS2',     e.SKAMID_PLUS2 ), # 61
    ( 'yCHIME_nocut',     e.CHIME_nocut ),  # 62
    ( 'yCHIME_avglow',    e.CHIME_avglow ), # 63
    ( 'MID_B1_Base',      e.MID_B1_Base ),  # 64
    ( 'MID_B1_Alt',       e.MID_B1_Alt ),   # 65
    ( 'MID_B2_Base',      e.MID_B2_Base ),  # 66
    ( 'MID_B2_Upd',       e.MID_B2_Upd ),   # 67
    ( 'MID_B2_Alt',       e.MID_B2_Alt ),   # 68
    ( 'aLOW_Base',        e.LOW_Base ),     # 69
    ( 'aLOW_Upd',         e.LOW_Upd ),      # 70
    ( 'aLOW_Alt',         e.LOW_Alt ),      # 71
    ( 'MID_B2_Alt2',      e.MID_B2_Alt2 ),  # 72
    ( 'iMID_B1_Base',     e.MID_B1_Base ),  # 73
    ( 'iMID_B1_Alt',      e.MID_B1_Alt ),   # 74
    ( 'iMID_B2_Base',     e.MID_B2_Base ),  # 75
    ( 'hMID_B1_Rebase',   e.MID_B1_Rebase), # 76
    ( 'hMID_B1_Octave',   e.MID_B1_Octave), # 77
    ( 'hMID_B2_Rebase',   e.MID_B2_Rebase), # 78
    ( 'hMID_B2_Octave',   e.MID_B2_Octave), # 79
    ( 'iCVTEST1',         e.CVlimited_z0to3), # 80
    ( 'iCVTEST2',         e.CVlimited_z2to5), # 81
    ( 'iHIRAX',           e.HIRAX),         # 82
    ( 'FASTWB',           e.FASTWB),         # 83
    ( 'FAST01',           e.FAST_01),         # 84
    ( 'FAST02',           e.FAST_02),         # 85
    ( 'FAST03',           e.FAST_03),         # 86
    ( 'FAST05',           e.FAST_05),         # 87
    ( 'FAST07',           e.FAST_07),         # 88
    ( 'FASTWB20K',           e.FASTWB20K),         # 89
    ( 'FASTLB00',           e.FASTLB00),         # 90
    ( 'FASTLB05',           e.FASTLB05),         # 91
    ( 'FASTLB10',           e.FASTLB10),         # 92
    ( 'FASTLB15',           e.FASTLB15),         # 93
    ( 'FASTLB20',           e.FASTLB20),         # 94
    ( 'FASTwide',           e.FASTwide19Beam),         # 95
    ( 'FASThighz',           e.FASThighZLband),         # 96
    ( 'MeerKAT10hrs',           e.MeerKAT10hrs),         # 97
]
names, expts = list(zip(*expt_list))
names = list(names); expts = list(expts)

################################################################################

# Take command-line argument for which survey to calculate, or set manually
Sarea = None
dnu = None
if len(sys.argv) > 1:
    k = int(sys.argv[1])
    try:
        if sys.argv[2] == "s":
            Sarea = float(sys.argv[3])
        elif sys.argv[2] == "n":
            dnu = float(sys.argv[3])
        elif sys.argv[2] == "sn":
            Sarea = float(sys.argv[3])
            dnu = float(sys.argv[4])
    except:
        Sarea = None
        dnu = None
        pass

else:
    raise IndexError("Need to specify ID for experiment.")

names[k] += EXPT_LABEL
if myid == 0:
    print("="*50)
    print("Survey:", names[k])
    print("="*50)

# Tweak settings depending on chosen experiment
cv_limited = False
expts[k]['mode'] = "dish"
if names[k][0] == "i": expts[k]['mode'] = "idish"
if names[k][0] == "c": expts[k]['mode'] = "combined"
if names[k][0] == "y": expts[k]['mode'] = "icyl"
if names[k][0] == "f": expts[k]['mode'] = "paf"
if names[k][0] == "t": expts[k]['mode'] = "ipaf"
if names[k][0] == "a": expts[k]['mode'] = "iaa"
if names[k][0] == "h": expts[k]['mode'] = "hybrid"

expt = expts[k]
'''
if Sarea is None:
    survey_name = names[k]
else:
    expt['Sarea'] = Sarea * (D2RAD)**2.
    survey_name = names[k] + "_" + str(int(Sarea))
'''
try:
    if sys.argv[2]=="s":
        survey_name = names[k]+"_s_"+str(int(Sarea))
        expt['Sarea'] = Sarea * (D2RAD)**2.
    elif sys.argv[2]=="n":
        survey_name = names[k]+"_n_"+str(int(dnu))
    elif sys.argv[2]=="sn":
        survey_name = names[k]+"_sn_"+str(int(Sarea))+str(int(dnu))
        expt['Sarea'] = Sarea * (D2RAD)**2.
    else:
        survey_name = names[k]
except:
    survey_name = names[k]
root = "output/" + survey_name

# Define redshift bins
expt_zbins = rf.overlapping_expts(expt)
###zs, zc = rf.zbins_equal_spaced(expt_zbins, dz=0.2)
if dnu is None:
    zs, zc =  rf.zbins_const_dnu(expt_zbins, cosmo, dnu=20)
else:
    zs, zc =  rf.zbins_const_dnu(expt_zbins, cosmo, dnu=dnu)
#zs, zc = rf.zbins_const_dr(expt_zbins, cosmo, bins=14)
#zs, zc = rf.zbins_const_dnu(expt_zbins, cosmo, dnu=60.)
#zs, zc = rf.zbins_const_dnu(expt_zbins, cosmo, dnu=30.)
#zs = rf.zbins_fixed(expt_zbins, dz=0.1)

print_test(1420/(1+zc))


# Define kbins (used for output)
kbins = np.logspace(np.log10(0.001), np.log10(50.), 91)
#cosmo['f0_kbins'] = np.array([1e-4, 1e-2, 1e-1, 1e1])

# FIXME
expt['epsilon_fg'] = 1e-6 #1e-14
expt['ttot'] *= 8765. / 1e4 # 1 year
#expt['ttot'] *= 1e10 / 1e4
#expt['ttot'] = 3.1536e+13 # FIXME: Amadeus' value
expt['k_nl0'] = 0.14 #KMAXVAL

# Neutrino mass
cosmo['mnu'] = 0.

# Precompute cosmological functions, P(k), massive neutrinos, and T(k) for f_NL
cosmo_fns = rf.background_evolution_splines(cosmo)
if cosmo['mnu'] != 0.:
    # Massive neutrinos
    mnu_str = "mnu%03d" % (cosmo['mnu']*100.)
    fname_pk = "cache_pk_%s.dat" % mnu_str
    fname_nu = "cache_%s" % mnu_str
    survey_name += mnu_str; root += mnu_str
    cosmo = rf.load_power_spectrum(cosmo, fname_pk, comm=comm)
    mnu_fn = rf.deriv_neutrinos(cosmo, fname_nu, mnu=cosmo['mnu'], comm=comm)
else:
    # Normal operation (no massive neutrinos or non-Gaussianity)
    cosmo = rf.load_power_spectrum(cosmo, "cache_pk.dat", comm=comm)
    mnu_fn = None

# Non-Gaussianity
#transfer_fn = rf.deriv_transfer(cosmo, "cache_transfer.dat", comm=comm)
transfer_fn = None

# Effective no. neutrinos, N_eff
#Neff_fn =  rf.deriv_neutrinos(cosmo, "cache_Neff", Neff=cosmo['N_eff'], comm=comm)
Neff_fn = None

# Optional additional parameters
switches = []
#switches = ['mg', ] #'sdbias']

# Scale-dependent growth
#cosmo['fs8_kbins'] = [0., 1e-2, 1e-1, 1e0, 1e2]

H, r, D, f = cosmo_fns

################################################################################
# Store cosmological functions
################################################################################

# Store values of cosmological functions
if myid == 0:
    # Calculate cosmo fns. at redshift bin centroids and save
    _H = H(zc)
    _dA = r(zc) / (1. + np.array(zc))
    _D = D(zc)
    _f = f(zc)
    np.savetxt(root+"-cosmofns-zc.dat", np.column_stack((zc, _H, _dA, _D, _f)),fmt='%.64f')
    
    # Calculate cosmo fns. as smooth fns. of z and save
    zz = np.linspace(0., 1.05*np.max(zc), 1000)
    _H = H(zz)
    _dA = r(zz) / (1. + zz)
    _D = D(zz)
    _f = f(zz)
    np.savetxt(root+"-cosmofns-smooth.dat", np.column_stack((zz, _H, _dA, _D, _f)),fmt='%.64f' )

# Precompute derivs for all processes
eos_derivs = rf.eos_fisher_matrix_derivs(cosmo, cosmo_fns, fsigma8=True)

# FIXME
print("*"*50)
for key in list(cosmo.keys()):
    print("%20s: %s" % (key, cosmo[key]))
print("*"*50)
for key in list(expt.keys()):
    print("%20s: %s" % (key, expt[key]))
print("*"*50)
#exit()


################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################
print_test(survey_name)
print_test(zs.size)
print_test(zs)
for i in range(zs.size-1):
    if i % size != myid:
      continue
    print(">>> %2d working on redshift bin %2d -- z = %3.3f" % (myid, i, zc[i]))
    
    # Calculate effective experimental params. in the case of overlapping expts.
    print("test sarea: 114", Sarea)
    Sarea_rad = Sarea*(D2RAD)**2. if Sarea is not None else None
    print("test sarea: 1919", Sarea_rad)
    expt_eff = rf.overlapping_expts(expt, zs[i], zs[i+1], Sarea=Sarea_rad)
    
    # Calculate basic Fisher matrix
    # (A, bHI, Tb, sigma_NL, sigma8, n_s, f, aperp, apar, [Mnu], [fNL], [pk]*Nkbins)
    F_pk, kc, binning_info, paramnames = rf.fisher( 
                                         zs[i], zs[i+1], cosmo, expt_eff, 
                                         cosmo_fns=cosmo_fns,
                                         transfer_fn=transfer_fn,
                                         massive_nu_fn=mnu_fn,
                                         Neff_fn=Neff_fn,
                                         return_pk=True,
                                         cv_limited=cv_limited, 
                                         switches=switches,
                                         kbins=kbins )
    
    # Expand Fisher matrix with EOS parameters
    ##F_eos =  rf.fisher_with_excluded_params(F, [10, 11, 12]) # Exclude P(k)
    F_eos, paramnames = rf.expand_fisher_matrix(zc[i], eos_derivs, F_pk, 
                                                names=paramnames, exclude=[], 
                                                fsigma8=True)
    
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
    np.savetxt(root+"-fisher-full-%d.dat" % i, F_eos, header=" ".join(paramnames),fmt='%.64f')
    if myid == 0: np.savetxt(root+"-fisher-kc.dat", kc,fmt='%.64f')
    
    # Save P(k) rebinning info
    np.savetxt(root+"-rebin-Fbase-%d.dat" % i, np.array(binning_info['F_base']) ,fmt='%.64f')
    np.savetxt(root+"-rebin-cumul-%d.dat" % i, np.array(binning_info['cumul']),fmt='%.64f' )
    np.savetxt(root+"-rebin-kgrid-%d.dat" % i, np.array(binning_info['kgrid']),fmt='%.64f' )
    np.savetxt(root+"-rebin-Vfac-%d.dat" % i, np.array([binning_info['Vfac'],]),fmt='%.64f' )

comm.barrier()
if myid == 0: print("Finished.")
