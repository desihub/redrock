"""
redrock.constants
=================

Set constants used by the rest of the package.
"""
max_velo_diff = 1000.0  # km/s
min_resolution_integral = 0.99

### From `get_tau` in `desisim/py/desisim/lya_mock_p1d.py`
Lyman_series = {
    'LYA' : { 'line':1215.67, 'A':0.000318, 'B':5.10 }
}
