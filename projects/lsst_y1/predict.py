import torch
import torch.nn as nn
import numpy as np
from cocoa_emu import cocoa_config
from cocoa_emu import nn_pca_emulator 
from cocoa_emu.nn_emulator import Affine, Better_ResBlock
import sys, platform, os
# sys.path.insert(0, os.environ['ROOTDIR'] + 
#                    '/external_modules/code/CAMB/build/lib.linux-x86_64-'
#                    +os.environ['PYTHON_VERSION'])
# import functools
# import numpy as np
# import ipyparallel
# import sys, platform, os
# import math
# import euclidemu2
# import scipy
# from getdist import IniFile
# import itertools
# import iminuit
# import camb
# import cosmolike_lsst_y1_interface as ci
# import copy
# import argparse
# import random
# import emcee

# setting up device: cuda or cpu
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


### Define the model here
#   it must be exactly the same as used when training

in_dim=12
int_dim_res = 256
out_dim = 780
layers = []
layers.append(nn.Linear(in_dim, int_dim_res))
layers.append(Better_ResBlock(int_dim_res, int_dim_res))
layers.append(Better_ResBlock(int_dim_res, int_dim_res))
layers.append(Better_ResBlock(int_dim_res, int_dim_res))
layers.append(nn.Linear(int_dim_res,out_dim))
layers.append(Affine())

model = nn.Sequential(*layers)

# initialize the emulator with non-sense data and replace it with the loaded model
emu_xi = nn_pca_emulator(model, 0, 0, 0, 0, device=device) 
emu_xi.load('./projects/lsst_y1/cocoa_emu/models/demo_resmlp_xi',state_dict=True)

# load fiducial datavector, mask, and covariance
config = cocoa_config('./projects/lsst_y1/train_emulator.yaml')

# load the shear calibration mask, we need to handle this ourselves when using the emulator
shear_calib_mask = np.load('./projects/lsst_y1/cocoa_emu/shear_calib_mask.npy')[:,:780]

def compute_dv(params):
    """
    takes in cosmological+nuisance parameters needed for the emulator
    returns the masked emulated datavector

    note: The emulator takes log(10^10 As), n_s, H0, Omegabh2, Omegach2 (+ nuisance params)
    """
    As_1e9 = params[0]
    ns     = params[1]
    H0     = params[2]
    omegab = params[3]
    omegam = params[4]
    
    DZ_S1  = params[5]
    DZ_S2  = params[6]
    DZ_S3  = params[7]
    DZ_S4  = params[8]
    DZ_S5  = params[9]    
    
    A1_1   = params[10] 
    A1_2   = params[11]

    mnu = 0.06

    omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
    omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708

    X = np.array([np.log(10*As_1e9), ns, H0, omegabh2(omegab, H0), omegach2(omegam, omegab, mnu, H0),
        DZ_S1, DZ_S2, DZ_S3, DZ_S4, DZ_S5, A1_1, A1_2])
    X = torch.Tensor(X)

    return (emu_xi.predict(X)[0])[config.mask[:780]]

    

def add_shear_calib(params, datavector):
    """
    adds the shear calibration corrections to the datavector.
    Note: this requires an extra file not used in cocoa: the shear_calib_mask.
          this maskes out components so that we get something of the form
          xi[ij] * (1+m[i])(1+m[j])
    """
    m = params[12:17]
    for i in range(5):
        factor = (1 + m[i])**shear_calib_mask[i]
        factor = (factor[0:780])[config.mask[:780]] # for cosmic shear
        datavector = factor * datavector
    return datavector

def compute_chi2(dv_predict, dv_exact, cov_inv):
    """
    you can use this instead of cosmolike interface. With this, the code is independent of ci
    it uses the fiducial DV and covariance matrix that are specified in the cobaya yaml
    """
    delta_dv = (dv_predict - np.float32(dv_exact))
    chi2 = np.matmul( np.matmul(np.transpose(delta_dv), np.float32(cov_inv)) , delta_dv  )   
    return chi2

def chi2(params):
    dv = compute_dv(params)
    dv = add_shear_calib(params, dv)

    ### change this depending on if you want cosmolike or not

    #chi2 = ci.compute_chi2(dv)
    chi2 = compute_chi2(dv, config.dv_masked, config.cov_inv_masked)

    return chi2



### example of using this:
print(chi2([2.4,0.97,68,0.048,0.3,0,0,0,0,0,0,0,0,0,0,0,0]))