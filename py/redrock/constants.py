"""
redrock.constants
=================

Set constants used by the rest of the package.
"""
max_velo_diff = 1000.0  # km/s
min_resolution_integral = 0.99

### From eqn 5 of Calura et al. 2012 (Arxiv: 1201.5121)
Lyman_series = {
    'LYA' : { 'line':1215.67, 'A':0.0023, 'B':3.64 }
}
