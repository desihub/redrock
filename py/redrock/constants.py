"""
redrock.constants
=================

Set constants used by the rest of the package.
"""
max_velo_diff = 1000.0  # km/s
max_velo_diff_star = 100.0  # km/s
min_resolution_integral = 0.99

min_deltachi2 = 9.

# Lyman-alpha from eqn 5 of Calura et al. 2012 (Arxiv: 1201.5121)
# Other from eqn 1.1 of Irsic et al. 2013 , (Arxiv: 1307.3403)
LyA_wavelength = 1215.67
Lyman_series = dict()
Lyman_series['Calura12'] = {
    'Lya'     : { 'line':LyA_wavelength, 'A':0.0023,    'B':3.64, 'var_evol':3.8 },
    'Lyb'     : { 'line':1025.72,  'A':0.0023/5.2615,   'B':3.64, 'var_evol':3.8 },
    'Ly3'     : { 'line':972.537,  'A':0.0023/14.356,   'B':3.64, 'var_evol':3.8 },
    'Ly4'     : { 'line':949.7431, 'A':0.0023/29.85984, 'B':3.64, 'var_evol':3.8 },
    'Ly5'     : { 'line':937.8035, 'A':0.0023/53.36202, 'B':3.64, 'var_evol':3.8 },
}

# Lyman-alpha values from Kamble et al. 2020 (Arxiv: 1904.01110)
# Other from eqn 1.1 of Irsic et al. 2013 , (Arxiv: 1307.3403)
# Optical depth model tuned to QSO HIZ v1.1, corrected only N=2 only
# (Lyman-alpha, but not Lyman-beta or beyond)
# TODO: resolve discrepancy on Irsic+ 2013 eqn 1.1 constants
# TODO:    with what is listed in Lyman_series_Calura12
Lyman_series['Kamble20'] = {
    'Lya'     : { 'line':LyA_wavelength,  'A':0.00554,          'B':3.182 },
    #'Lyb'     : { 'line':1025.72,  'A':0.00554/5.2615,   'B':3.182 },
    #'Ly3'     : { 'line':972.537,  'A':0.00554/14.356,   'B':3.182 },
    #'Ly4'     : { 'line':949.7431, 'A':0.00554/29.85984, 'B':3.182 },
    #'Ly5'     : { 'line':937.8035, 'A':0.00554/53.36202, 'B':3.182 },
}

# Lyman alpha values from Turner et.al 2024: https://arxiv.org/pdf/2405.06743
# higher order scaled accordingly
Lyman_series['Turner24'] = {
    'Lya'     : { 'line':LyA_wavelength, 'A':0.00246, 'B':3.62 },
    # scaling as lambda_n*oscillator_strength_n/lya*oscillator_strength_alpha
    'Lyb'     : { 'line':1025.72, 'A':0.00246*(1025.72/LyA_wavelength)/5.2615,   'B':3.62 },
    'Ly3'     : { 'line':972.537,  'A':0.00246*(972.537/LyA_wavelength)/14.356,   'B':3.62 },
    'Ly4'     : { 'line':949.7431, 'A':0.00246*(949.7431/LyA_wavelength)/29.85984, 'B':3.62 },
    'Ly5'     : { 'line':937.8035, 'A':0.00246*(937.8035/LyA_wavelength)/53.36202, 'B':3.62 },
    'Ly6'     : { 'line':930.7483, 'A':0.00246*(930.7483/LyA_wavelength)/86.49356, 'B':3.62 },
    'Ly7'     : { 'line':926.2257, 'A':0.00246*(926.2257/LyA_wavelength)/130.81370, 'B':3.62 },
    'Ly8'     : { 'line':923.1504, 'A':0.00246*(923.1504/LyA_wavelength)/187.90614, 'B':3.62 },
    'Ly9'    : { 'line':920.9631, 'A':0.00246*(920.9631/LyA_wavelength)/259.43925, 'B':3.62 }
    }
