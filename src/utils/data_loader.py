"""
Data loading utilities for MATLAB simulation outputs
"""
import numpy as np
import scipy.io as sio


def load_grey_scott(filepath):
    """Load Gray-Scott simulation data from .mat file"""
    data = sio.loadmat(filepath)
    
    return {
        'u': data['usol'],
        'v': data['vsol'],
        't': data['t'].flatten(),
        'x': data['x'].flatten(),
        'y': data['y'].flatten(),
        'params': {
            'b1': float(data['b1']),
            'b2': float(data['b2']),
            'ep1': float(data['ep1']),
            'ep2': float(data['ep2'])
        }
    }
