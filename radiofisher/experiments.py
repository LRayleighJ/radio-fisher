import numpy as np
import scipy.interpolate
from .units import *

# Define fiducial cosmology and parameters
# Planck-only best-fit parameters, from Table 2 of Planck 2013 XVI.
cosmo = {
    'omega_M_0':        0.316,
    'omega_lambda_0':   0.684,
    'omega_b_0':        0.049,
    'omega_HI_0':       4.86e-4, #6.50e-4,
    'N_eff':            3.046,
    'h':                0.67,
    'ns':               0.962,
    'sigma_8':          0.834,
    'gamma':            0.55,
    'w0':               -1.,
    'wa':               0.,
    'fNL':              0.,
    'mnu':              0.,
    'k_piv':            0.05, # n_s
    'aperp':            1.,
    'apar':             1.,
    'bHI0':             0.677, #0.702,
    'A':                1.,
    'sigma_nl':         7.,
    'b_1':              0.,         # Scale-dependent bias (k^2 term coeff.)
    'k0_bias':          0.1,        # Scale-dependent bias pivot scale [Mpc^-1]
    'gamma0':           0.55,
    'gamma1':           0.,
    'eta0':             0.,
    'eta1':             0.,
    'A_xi':             0.00,         # Modified gravity growth amplitude
    'logkmg':           np.log10(0.05) # New modified gravity growth scale
}

# Define which measurements to include in forecasts
USE = {
  'f_rsd':             True,     # RSD constraint on f(z)
  'f_growthfactor':    True,    # D(z) constraint on f(z)
  'alpha_all':         True,     # Use all constraints on alpha_{perp,par}
  'alpha_volume':      False,
  'alpha_rsd_angle':   False,
  'alpha_rsd_shift':   False,
  'alpha_bao_shift':   True,
  'alpha_pk_shift':    False
}

SURVEY = {
    'ttot':             10e3*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}
SURVEY_FAST = {
    'ttot':             60*24*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

SURVEY_FAST_no_ttot = {
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

SURVEY_no_ttot = {
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

SURVEY_FASTWB = {
    'ttot':             82000*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}
SURVEY_MEERKATB1 = {
    'ttot':             4000*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}
SURVEY_MEERKATB2 = {
    'ttot':             4000*HRS_MHZ,      # Total integration time [MHz^-1]
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

SURVEY_MEERKAT10hrs = {
    'nu_line':          1420.406,          # Rest-frame freq. of emission line [MHz]
    'epsilon_fg':       1e-6,              # FG subtraction residual amplitude
    'k_nl0':            0.14,              # Non-linear scale at z=0 (sets kmax)
    'use':              USE                # Which constraints to use/ignore
}

# Add foreground components to cosmology dict.
# (Extragal. ptsrc, extragal. free-free, gal. synch., gal. free-free)
foregrounds = {
    'A':     [57.0, 0.014, 700., 0.088],        # FG noise amplitude [mK^2]
    'nx':    [1.1, 1.0, 2.4, 3.0],              # Angular scale powerlaw index
    'mx':    [-2.07, -2.10, -2.80, -2.15],      # Frequency powerlaw index
    'l_p':   1000.,                             # Reference angular scale
    'nu_p':  130.                               # Reference frequency [MHz]
}
cosmo['foregrounds'] = foregrounds


################################################################################
# Illustrative experiments used in paper
################################################################################

exptS = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            50,                # No. of beams (for multi-pixel detectors)
    'Ddish':            30.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
exptS.update(SURVEY)

exptM = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            160,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            4.,                # Single dish diameter [m]
    'Tinst':            35.*(1e3),         # System temp. [mK]
    'nu_crit':          1000.,             # Critical frequency [MHz]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             53.,               # Max. interferom. baseline [m]
    'Dmin':             4.                 # Min. interferom. baseline [m]
    }
exptM.update(SURVEY)

exptL = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             600.,              # Max. interferom. baseline [m]
    'Dmin':             15.                # Min. interferom. baseline [m]
    }
exptL.update(SURVEY)

# Matched to Euclid redshift/Sarea
exptCV = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1e10,              # No. of dishes (HUGE!)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     860.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            15e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
exptCV.update(SURVEY)

################################################################################
# IM experiment configurations
################################################################################

GBT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            100.,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    240.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     920.,              # Max. freq. of survey
    'dnu':              0.1,             # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
GBT.update(SURVEY)

GBTHIM = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            7,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            100.,              # Single dish diameter [m]
    'Tinst':            33.*(1e3),         # System temp. [mK]
    'survey_dnutot':    200.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.78,              # Bandwidth of single channel [MHz]
    'Sarea':            5e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
GBTHIM.update(SURVEY)

Parkes = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            13,                # No. of beams (for multi-pixel detectors)
    'Ddish':            64.,               # Single dish diameter [m]
    'Tinst':            23.*(1e3),         # System temp. [mK]
    'survey_dnutot':    265.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
Parkes.update(SURVEY)

GMRT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            30,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            45.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e2*(D2RAD)**2.,   # Total survey area [radians^2]
    }
GMRT.update(SURVEY)

# FIXME: What is the actual bandwidth of WSRT?
WSRT = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            120.*(1e3),        # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
WSRT.update(SURVEY)

APERTIF = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            14,                # No. of dishes
    'Nbeam':            37,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            52.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1300.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
APERTIF.update(SURVEY)

VLBA = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            10,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            27.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
VLBA.update(SURVEY)

JVLA = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            27,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    420.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_VLAD_dec90.dat" # Interferometer antenna density
    }
JVLA.update(SURVEY)

BINGO = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            30*28,                # No. of beams (for multi-pixel detectors)
    'Ddish':            25.,               # Single dish diameter [m]
    'Tinst':            70.*(1e3),         # System temp. [mK]
    'survey_dnutot':    280.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1260.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2.9e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'ttot':             2*365*24*HRS_MHZ,      # Total integration time [MHz^-1]
    }
BINGO.update(SURVEY_no_ttot)

BAOBAB32 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            32,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            1.6,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             13.8,              # Max. interferom. baseline [m]
    'Dmin':             1.6                # Min. interferom. baseline [m]
    }
BAOBAB32.update(SURVEY)

BAOBAB128 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            128,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            1.6,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             26.,               # Max. interferom. baseline [m]
    'Dmin':             1.6                # Min. interferom. baseline [m]
    }
BAOBAB128.update(SURVEY)

KZN = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            1225,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            5.0,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             200.,              # Max. interferom. baseline [m]
    'Dmin':             5.0                # Min. interferom. baseline [m]
    }
KZN.update(SURVEY)

# FIXME
MFAA = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            100*31,            # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.4,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     950.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             250.,              # Max. interferom. baseline [m]
    'Dmin':             0.1                # Min. interferom. baseline [m]
    }
MFAA.update(SURVEY)

CHIME = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            1280,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             5,                 # No. cylinders
    'Ddish':            20.,               # Single dish diameter [m]
    'cyl_area':         20.*80.,           # Single dish area [m^2]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             128.,              # Max. interferom. baseline [m]
    'Dmin':             20.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800.dat" # Interferometer antenna density
    }
CHIME.update(SURVEY)

CHIME_nocut = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            1280,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             5,                 # No. cylinders
    'Ddish':            20.,               # Single dish diameter [m]
    'cyl_area':         20.*80.,           # Single dish area [m^2]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             128.,              # Max. interferom. baseline [m]
    'Dmin':             20.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800_nocut.dat" # Interferometer antenna density
    }
CHIME_nocut.update(SURVEY)

CHIME_avglow = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            1280,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             5,                 # No. cylinders
    'Ddish':            20.,               # Single dish diameter [m]
    'cyl_area':         20.*80.,           # Single dish area [m^2]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             128.,              # Max. interferom. baseline [m]
    'Dmin':             20.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_CHIME_800_avg.dat" # Interferometer antenna density
    }
CHIME_avglow.update(SURVEY)

TIANLAI = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            8*256,             # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             8,                 # No. cylinders
    'cyl_area':         15.*120.,           # Single dish area [m^2]
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     950.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             159.1,             # Max. interferom. baseline [m]
    'Dmin':             15.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_TIANLAI_1000.dat" # Interferometer antenna density
    }
TIANLAI.update(SURVEY)

TIANLAIband2 = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            8*256,             # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             8,                 # No. cylinders
    'cyl_area':         15.*120.,           # Single dish area [m^2]
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    400.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             159.1,             # Max. interferom. baseline [m]
    'Dmin':             15.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_TIANLAI_1000.dat" # Interferometer antenna density
    }
TIANLAIband2.update(SURVEY)

TIANLAIpath = {
    'mode':             'cylinder',        # Interferometer or single dish
    'Ndish':            3*32,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ncyl':             3,                 # No. cylinders
    'cyl_area':         15.*40.,           # Single dish area [m^2]
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    100.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e2*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             33.8,              # Max. interferom. baseline [m]
    'Dmin':             15.,               # Min. interferom. baseline [m]
    'n(x)': "array_config/nx_TIANLAIpath_1000.dat" # Interferometer antenna density
    }
TIANLAIpath.update(SURVEY)

AERA3 = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            100,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            5.,                # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,             # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'Dmax':             80.,               # Max. interferom. baseline [m]
    'Dmin':             5.                 # Min. interferom. baseline [m]
    }
AERA3.update(SURVEY)

FAST = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1350.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FAST.update(SURVEY_FAST)

KAT7 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            7,                 # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    220.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            2e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_KAT7_dec30.dat" # Interferometer antenna density
    }
KAT7.update(SURVEY)

MeerKATb1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            29.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.2,               # Bandwidth of single channel [MHz]
    'Sarea':            4e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKATb1.update(SURVEY_MEERKATB1)

MeerKATb2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.2,               # Bandwidth of single channel [MHz]
    'Sarea':            4e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKATb2.update(SURVEY_MEERKATB2)

ASKAP = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            36,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':            12.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_ASKAP_dec30.dat" # Interferometer antenna density
    }
ASKAP.update(SURVEY)

SKA1MIDbase1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,                # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
SKA1MIDbase1.update(SURVEY)

SKA1MIDbase2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    470.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAM190_dec30.dat" # Interferometer antenna density
    }
SKA1MIDbase2.update(SURVEY)

SKA1SURbase1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          710.,              # PAF critical frequency
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURbase1.update(SURVEY)

SKA1SURbase2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          1300.,             # PAF critical frequency
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURbase2.update(SURVEY)

SKA1MIDfull1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish': (190.*15. + 64.*13.5)/(254.), # Single dish diameter [m]
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    435.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MIDfull1.update(SURVEY)

SKA1MIDfull2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish': (190.*15. + 64.*13.5)/(254.), # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    470.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MIDfull2.update(SURVEY)

SKA1SURfull1 = {
    'mode':             'dish',            # Interferometer or single dish
    'nu_crit':          710.,              # PAF critical frequency
    'Ndish':            96,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':    (60.*15. + 36.*12.)/(96.), # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURfull1.update(SURVEY)

SKA1SURfull2 = {
    'mode':             'dish',            # Interferometer or single dish
    'nu_crit':          1300.,             # PAF critical frequency
    'Ndish':            96,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'Ddish':    (60.*15. + 36.*12.)/(96.), # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
SKA1SURfull2.update(SURVEY)


########################################
# SKA configurations from HI IM chapter
########################################

SKA0MID = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            127,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'theta_b':          0.714*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
SKA0MID.update(SURVEY)

SKA0MIDB1 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            95,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'theta_b':          1.334*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
SKA0MIDB1.update(SURVEY)

SKA0SUR = {
    'mode':             'paf',             # Interferometer or single dish
    'Ndish':            48,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          1300.,             # PAF critical frequency
    'theta_b':          0.714*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
SKA0SUR.update(SURVEY)

SKA1MID900 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'theta_b':          0.714*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MID900.update(SURVEY)

SKA1MID350 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'theta_b':          1.334*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MID350.update(SURVEY)

SKA1SUR350fullband = {
    'mode':             'paf',             # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          710.,              # PAF critical frequency
    'theta_b':          1.31*(D2RAD),      # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    550.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
SKA1SUR350fullband.update(SURVEY)

SKA1SUR650 = {
    'mode':             'paf',             # Interferometer or single dish
    'Ndish':            96,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          1300.,             # PAF critical frequency
    'theta_b':          0.714*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            30.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
SKA1SUR650.update(SURVEY)

SKA1SUR350 = {
    'mode':             'paf',             # Interferometer or single dish
    'Ndish':            60,                # No. of dishes
    'Nbeam':            36,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          710.,              # PAF critical frequency
    'theta_b':          1.31*(D2RAD),      # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            50.*(1e3),         # System temp. [mK]
    'survey_dnutot':    500.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     900.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.    # Total survey area [radians^2]
    }
SKA1SUR350.update(SURVEY)

SKA1LOW = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            911,               # No. of dishes
    'Nbeam':            3,                 # No. of beams (for multi-pixel detectors)
    'nu_crit':          110.,              # PAF critical frequency
    'theta_b':          5.29*(D2RAD),      # Beam at critical frequency
    'Aeff':             925.,              # Effective area at critical freq.
    'Ddish':            35.,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'Tsky_factor':      0.1,               # Additional factor of Tsky to add to Tsys
    'survey_dnutot':    100.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     350.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKALOW_190_dec30.dat" # Interferometer antenna density
    }
SKA1LOW.update(SURVEY)

SKA2 = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            4000,              # No. of dishes
    'Nbeam':            10,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          500.,              # PAF critical frequency
    'theta_b':          2.828*(D2RAD), #0.707*(D2RAD),     # Beam at critical frequency
    'Aeff':             63.,               # Effective area at critical freq.
    'Ddish':            10.,               # Single dish diameter [m]
    'Tinst':            15.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Dmin':             1.0,
    'Dmax':             1e3,
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
SKA2.update(SURVEY)

SKA1MID350upd = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            190,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'theta_b':          1.334*(D2RAD),     # Beam at critical frequency
    'Aeff':             140.,              # Effective area at critical freq.
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            5e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
SKA1MID350upd.update(SURVEY)


########################################
# Updated SKA configs for MG paper
########################################

MID_B1_Rebase = {
    'mode':             'hybrid',          # Interferometer or single dish
    'Ndish':            130,               # No. of dishes (MID dishes)
    'Ndish2':           64,                # No. of dishes (MeerKAT dishes)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Ddish2':           13.5,              # Single dish diameter (2) [m]
    'effic':            0.75,              # Aperture efficiency
    'effic2':           0.80,              # Aperture efficiency
    'Tinst':            23.*(1e3),         # System temp. [mK]
    'Tinst2':           23.*(1e3),         # System temp. (2) [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of *entire* survey
    'array_numax1':     1050.,             # Max. freq. of survey 1
    'array_numax2':     1015.,             # Max. freq. of survey 2
    'array_dnutot1':    700.,              # Total bandwidth of array 1 [MHz]
    'array_dnutot2':    435.,              # Total bandwidth of array 2 [MHz]
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B1_Rebase.update(SURVEY)

MID_B1_Octave = {
    'mode':             'hybrid',          # Interferometer or single dish
    'Ndish':            130,               # No. of dishes (MID dishes)
    'Ndish2':           64,                # No. of dishes (MeerKAT dishes)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Ddish2':           13.5,              # Single dish diameter (2) [m]
    'effic':            0.75,              # Aperture efficiency
    'effic2':           0.80,              # Aperture efficiency
    'Tinst':            12.*(1e3),         # System temp. [mK]
    'Tinst2':           23.*(1e3),         # System temp. (2) [mK]
    'survey_dnutot':    565.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of *entire* survey
    'array_numax1':     825.,              # Max. freq. of survey 1
    'array_numax2':     1015.,             # Max. freq. of survey 2
    'array_dnutot1':    375.,              # Total bandwidth of array 1 [MHz]
    'array_dnutot2':    435.,              # Total bandwidth of array 2 [MHz]
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B1_Octave.update(SURVEY)

MID_B2_Rebase = {
    'mode':             'hybrid',          # Interferometer or single dish
    'Ndish':            130,               # No. of dishes (MID dishes)
    'Ndish2':           64,                # No. of dishes (MeerKAT dishes)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Ddish2':           13.5,              # Single dish diameter (2) [m]
    'effic':            0.85,              # Aperture efficiency
    'effic2':           0.85,              # Aperture efficiency
    'Tinst':            15.5*(1e3),        # System temp. [mK]
    'Tinst2':           30.*(1e3),         # System temp. (2) [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of *entire* survey
    'array_numax1':     1420.,             # Max. freq. of survey 1
    'array_numax2':     1420.,             # Max. freq. of survey 2
    'array_dnutot1':    470.,              # Total bandwidth of array 1 [MHz]
    'array_dnutot2':    520.,              # Total bandwidth of array 2 [MHz]
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B2_Rebase.update(SURVEY)

MID_B2_Octave = {
    'mode':             'hybrid',          # Interferometer or single dish
    'Ndish':            130,               # No. of dishes (MID dishes)
    'Ndish2':           64,                # No. of dishes (MeerKAT dishes)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Ddish2':           13.5,              # Single dish diameter (2) [m]
    'effic':            0.85,              # Aperture efficiency
    'effic2':           0.85,              # Aperture efficiency
    'Tinst':            15.5*(1e3),        # System temp. [mK]
    'Tinst2':           30.*(1e3),         # System temp. (2) [mK]
    'survey_dnutot':    625.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of *entire* survey
    'array_numax1':     1420.,             # Max. freq. of survey 1
    'array_numax2':     1420.,             # Max. freq. of survey 2
    'array_dnutot1':    625.,              # Total bandwidth of array 1 [MHz]
    'array_dnutot2':    520.,              # Total bandwidth of array 2 [MHz]
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B2_Octave.update(SURVEY)


########################################
# SKA configurations from MG paper (MID specs are obsolete)
########################################

MID_B1_Base = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            28.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_254.dat" # Interferometer antenna density
    }
MID_B1_Base.update(SURVEY)

MID_B1_Alt = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            200,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            23.*(1e3),         # System temp. [mK]
    'survey_dnutot':    370.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     800.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B1_Alt.update(SURVEY)

MID_B1_Octave = {
    'mode':             'hybrid',          # Interferometer or single dish
    'Ndish':            130,               # No. of dishes (MID dishes)
    'Ndish2':           64,                # No. of dishes (MeerKAT dishes)
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Ddish2':           13.5,              # Single dish diameter (2) [m]
    'effic':            0.75,              # Aperture efficiency
    'effic2':           0.80,              # Aperture efficiency
    'Tinst':            12.*(1e3),         # System temp. [mK]
    'Tinst2':           19.*(1e3),         # System temp. (2) [mK]
    'survey_dnutot':    450.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     950.,              # Max. freq. of *entire* survey
    'array_numax1':     925.,              # Max. freq. of survey 1
    'array_numax2':     950.,              # Max. freq. of survey 2
    'array_dnutot1':    425.,              # Total bandwidth of array 1 [MHz]
    'array_dnutot2':    370.,              # Total bandwidth of array 2 [MHz]
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30_200.dat" # Interferometer antenna density
    }
MID_B1_Octave.update(SURVEY)

MID_B2_Base = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            254,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
MID_B2_Base.update(SURVEY)

MID_B2_Upd = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            200,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    520.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
MID_B2_Upd.update(SURVEY)

MID_B2_Alt = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            200,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    635.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
MID_B2_Alt.update(SURVEY)

MID_B2_Alt2 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            200,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    515.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1300.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            25e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKAMREF2_dec30.dat" # Interferometer antenna density
    }
MID_B2_Alt2.update(SURVEY)

LOW_Base = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            911,               # No. of dishes
    'Nbeam':            3,                 # No. of beams (for multi-pixel detectors)
    'nu_crit':          110.,              # PAF critical frequency
    'theta_b':          5.29*(D2RAD),      # Beam at critical frequency
    'Aeff':             925.,              # Effective area at critical freq.
    'Ddish':            35.,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'Tsky_factor':      0.1,               # Additional factor of Tsky to add to Tsys
    'survey_dnutot':    150.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     350.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKALOW_190_dec30.dat" # Interferometer antenna density
    }
LOW_Base.update(SURVEY)

LOW_Upd = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            455,               # No. of dishes
    'Nbeam':            3,                 # No. of beams (for multi-pixel detectors)
    'nu_crit':          110.,              # PAF critical frequency
    'theta_b':          5.29*(D2RAD),      # Beam at critical frequency
    'Aeff':             925.,              # Effective area at critical freq.
    'Ddish':            35.,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'Tsky_factor':      0.1,               # Additional factor of Tsky to add to Tsys
    'survey_dnutot':    150.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     350.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKALOW_190_dec30_Ndish455.dat" # Interferometer antenna density
    }
LOW_Upd.update(SURVEY)

LOW_Alt = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            455,               # No. of dishes
    'Nbeam':            3,                 # No. of beams (for multi-pixel detectors)
    'nu_crit':          110.,              # PAF critical frequency
    'theta_b':          5.29*(D2RAD),      # Beam at critical frequency
    'Aeff':             925.,              # Effective area at critical freq.
    'Ddish':            35.,               # Single dish diameter [m]
    'Tinst':            40.*(1e3),         # System temp. [mK]
    'Tsky_factor':      0.1,               # Additional factor of Tsky to add to Tsys
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     500.,              # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    'n(x)': "array_config/nx_SKALOW_190_dec30_Ndish455.dat" # Interferometer antenna density
    }
LOW_Alt.update(SURVEY)

# FIXME
SKA2_MG = {
    'mode':             'iaa',             # Interferometer or single dish
    'Ndish':            7000,              # No. of dishes
    'Nbeam':            10,                # No. of beams (for multi-pixel detectors)
    'nu_crit':          500.,              # PAF critical frequency
    'theta_b':          2.828*(D2RAD), #0.707*(D2RAD),     # Beam at critical frequency
    'Aeff':             63.,               # Effective area at critical freq.
    'Ddish':            10.,               # Single dish diameter [m]
    'Tinst':            15.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1000.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Dmin':             1.0,
    'Dmax':             1e3,
    'Sarea':            1e3*(D2RAD)**2.,   # Total survey area [radians^2]
    }
SKA2_MG.update(SURVEY)


##################

SKAMID_PLUS = {
    'overlap':          [SKA1MID350, MeerKATb1],
    'Sarea':            1e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
}

SKAMID_PLUS2 = {
    'overlap':          [SKA1MID900, MeerKATb2],
    'Sarea':            1e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
}


"""
# Surveys that are defined as overlap between two instruments
SKAMID_PLUS = {
    'overlap':          [SKA1MID, MeerKAT],
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKAMID_PLUS_band1 = {
    'overlap':          [SKA1MID, MeerKAT_band1],
    'Sarea':            25e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)':             "array_config/nx_SKAMREF2COMP_dec30.dat"
    }

SKASUR_PLUS = {
    'overlap':          [SKA1SUR, ASKAP],
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }

SKASUR_PLUS_band1 = {
    'overlap':          [SKA1SUR_band1, ASKAP],
    'Sarea':            25e3*(D2RAD)**2.   # Total survey area [radians^2]
    }
"""

"""
exptO = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.5,               # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1200.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             44.,               # Max. interferom. baseline [m]
    'Dmin':             2.5                # Min. interferom. baseline [m]
    }
exptO.update(SURVEY)

exptX = {
    'mode':             'combined',        # Interferometer or single dish
    'Ndish':            250000,               # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            15.,               # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1100.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            30e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             600.,              # Max. interferom. baseline [m]
    'Dmin':             15.                # Min. interferom. baseline [m]
    }
exptX.update(SURVEY)

# FIXME
exptOpt = {
    'mode':             'interferom',      # Interferometer or single dish
    'Ndish':            1500,              # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            2.,                # Single dish diameter [m]
    'Tinst':            25.*(1e3),         # System temp. [mK]
    'survey_dnutot':    1070.,             # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            10e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'Dmax':             85, #125.,         # Max. interferom. baseline [m]
    'Dmin':             2.                 # Min. interferom. baseline [m]
    }
exptOpt.update(SURVEY)
"""

HIRAX = {
    'mode':          'interferom', # Interferometer or single dish
    'Ndish':         1024,         # No. of dishes
    'Nbeam':         1,            # No. of beams (for multi-pixel detectors)
    'Ddish':         6.,           # Single dish diameter [m]
    'Tinst':         50.*(1e3),    # System temp. [mK]
    'survey_dnutot': 400.,
    'survey_numax':  800.,
    'dnu':           0.4,          # Bandwidth of single channel [MHz]
    'Sarea':         2.*np.pi,     # Total survey area [radians^2]
    'n(x)':          "array_config/hirax_Ndish1024_baseline1.dat",
}
HIRAX.update(SURVEY)



CVlimited_z0to3 = {
    'mode':          'interferom', # Interferometer or single dish
    'Ndish':         1024,         # No. of dishes
    'Nbeam':         1,            # No. of beams (for multi-pixel detectors)
    'Ddish':         6.,           # Single dish diameter [m]
    'Tinst':         50.*(1e3),    # System temp. [mK]
    #'nu_crit':      1000.,        # critical frequency, UNCLEAR
    'survey_dnutot': 850., #1065., 
    'survey_numax':  1200., #1419.,  
    'dnu':           0.4,          # Bandwidth of single channel [MHz]
    'Sarea':         2.*np.pi,     # Total survey area [radians^2]
    'n(x)':          "array_config/hirax_Ndish1024_baseline1.dat",
}
CVlimited_z0to3.update(SURVEY)


CVlimited_z2to5 = {
    'mode':          'interferom', # Interferometer or single dish
    'Ndish':         1024,         # No. of dishes
    'Nbeam':         1,            # No. of beams (for multi-pixel detectors)
    'Ddish':         6.,           # Single dish diameter [m]
    'Tinst':         50.*(1e3),    # System temp. [mK]
    #'nu_crit':      1000.,        # critical frequency, UNCLEAR
    'survey_dnutot': 236.7,
    'survey_numax':  473.5,
    'dnu':           0.4,          # Bandwidth of single channel [MHz]
    'Sarea':         2.*np.pi,     # Total survey area [radians^2]
    'n(x)':          "array_config/hirax_Ndish1024_baseline1.dat",
}
CVlimited_z2to5.update(SURVEY)

# Forecast for different FAST IM survey

FASTWB = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            60.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTWB.update(SURVEY_FASTWB)
FASTWB20K = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            1,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTWB20K.update(SURVEY_FASTWB)

FAST_01 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1290.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'ttot':             100000*HRS_MHZ,      # Total integration time [MHz^-1]
    }
FAST_01.update(SURVEY_FAST_no_ttot)

FAST_02 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1180.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'ttot':             92000*HRS_MHZ,      # Total integration time [MHz^-1]
    }
FAST_02.update(SURVEY_FAST_no_ttot)

FAST_03 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1090.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'ttot':             85000*HRS_MHZ,      # Total integration time [MHz^-1]
    }
FAST_03.update(SURVEY_FAST_no_ttot)
FAST_05 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     946.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'ttot':             74000*HRS_MHZ,      # Total integration time [MHz^-1]
    }
FAST_05.update(SURVEY_FAST_no_ttot)
FAST_07 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     835.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    'ttot':             65000*HRS_MHZ,      # Total integration time [MHz^-1]
    }
FAST_07.update(SURVEY_FAST_no_ttot)

# remove non-linear redshift period
FASTLB00 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    370.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTLB00.update(SURVEY_FAST)
FASTLB05 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    300.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1350.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTLB05.update(SURVEY_FAST)
FASTLB10 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    240.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1290.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTLB10.update(SURVEY_FAST)
FASTLB15 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    180.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1230.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTLB15.update(SURVEY_FAST)
FASTLB20 = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    130.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1180.,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTLB20.update(SURVEY_FAST)
FASTwide19Beam = {
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    700.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1420,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            20e3*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASTwide19Beam.update(SURVEY_FAST)

FASThighZLband = {
    'ttot':             800*HRS_MHZ,      # Total integration time [MHz^-1]
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    150.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1150,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e2*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASThighZLband.update(SURVEY_FAST_no_ttot)

FASThighZLbandshift = {
    'ttot':             800*HRS_MHZ,      # Total integration time [MHz^-1]
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            1,                 # No. of dishes
    'Nbeam':            19,                # No. of beams (for multi-pixel detectors)
    'Ddish':            300.,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    150.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1050,             # Max. freq. of survey ## zmin = 0.05
    'dnu':              0.1,               # Bandwidth of single channel [MHz]
    'Sarea':            2e2*(D2RAD)**2.,  # Total survey area [radians^2]
    }
FASThighZLbandshift.update(SURVEY_FAST_no_ttot)

MeerKAT10hrs= {
    'ttot':             640*HRS_MHZ,      # Total integration time [MHz^-1]
    'mode':             'dish',            # Interferometer or single dish
    'Ndish':            64,                # No. of dishes
    'Nbeam':            1,                 # No. of beams (for multi-pixel detectors)
    'Ddish':            13.5,              # Single dish diameter [m]
    'Tinst':            20.*(1e3),         # System temp. [mK]
    'survey_dnutot':    42.,              # Total bandwidth of *entire* survey [MHz]
    'survey_numax':     1015.,             # Max. freq. of survey
    'dnu':              0.2,               # Bandwidth of single channel [MHz]
    'Sarea':            2e2*(D2RAD)**2.,  # Total survey area [radians^2]
    'n(x)': "array_config/nx_MKREF2_dec30.dat" # Interferometer antenna density
    }
MeerKAT10hrs.update(SURVEY_MEERKAT10hrs)

