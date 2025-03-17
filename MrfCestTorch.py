#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mrf cest pulse sequence 

implementation of the mrf cest pulse sequence using pytorch libraries to enable gpu acceleration.

Created on Thu May 23 19:17:43 2019
author: ouri cohen    

"""
import math
import scipy.sparse as sps
import scipy
import numpy as np
import time
import torch

class MrfCest():
    def __init__(self,tissue_dict,acq_dict,seq_params):
        """
        input:
            tissue_dict: (dict) tissue parameters
            acq_dict: (dict) acquisition parameters
            offset_dict: (dict) offset frequencies
            seq_params: (dict) sequence parameters
        output: 
            
        """
        self.gpu=0 # TODO: need to read in this from the parameters
        
        self.gamma = 42.57e6 #define gyro-magnetic ratio, Hz/T
        self.parse_tissue_params(tissue_dict)
        self.parse_acquisition_params(acq_dict)        
        
        self.B0 = self.gamma * self.field_strength
        self.num_pools = seq_params['num_pools']
        self.spin_grad_flag = seq_params['spin_grad_flag']
        
        self.delta_water_array, self.delta_solute_array, self.delta_semi_solid_array, self.delta_aliphatic_array = self.calculate_freq_deltas() 
        self.mt_lineshape = self.calculate_mt_lineshape() 

    def parse_tissue_params(self,tissue_dict):
        
        self.T1w = tissue_dict['T1w'] # water T1s (ms)
        self.R1w = 1e3/self.T1w #water R1 (seconds)

        self.T1s = tissue_dict['T1s'] # solute T1s (ms)
        self.R1s = 1e3/self.T1s #solute R1 (seconds)

        self.T1ss = tissue_dict['T1ss'] # semi-solid T1s (ms)
        self.R1ss = 1e3/self.T1s #semi-solid R1 (seconds)
        
        self.T1a = tissue_dict['T1a'] # aliphatic T1s (ms)
        self.R1a = 1e3/self.T1a #aliphatic R1 (seconds)

        self.T2w = tissue_dict['T2w'] # water T2s (ms)
        self.R2w = 1e3/self.T2w #water R2 (seconds)

        self.T2s = tissue_dict['T2s'] # solute T2s (ms)
        self.R2s = 1e3/self.T2s #solute R2 (seconds)
        
        self.T2ss = tissue_dict['T2ss'] # semi-solid T2s (ms)
        self.R2ss = 1e3/self.T2ss #semi-solid R2 (seconds)
        
        self.T2a = tissue_dict['T2a'] # aliphatic T2s (ms)
        self.R2a = 1e3/self.T2a #aliphatic R2 (seconds)

        self.M0s = tissue_dict['M0s'] # solute M0 [0...1](a.u.)
        self.M0ss = tissue_dict['M0ss'] # semi-solid M0 [0...1](a.u.)
        self.M0a = tissue_dict['M0a'] # aliphatic M0 [0...1](a.u.)
        self.M0w = np.ones(self.M0s.shape) - self.M0s - self.M0ss - self.M0a # water M0
        
        self.Ksw = tissue_dict['Ksw'] # solute-water Ksw (Hz)
        self.Kssw = tissue_dict['Kssw'] # semi-solid-water Ksw (Hz)
        self.Kaw = tissue_dict['Kaw'] # aliphatic-water Kaw (Hz)
        self.Kssw = tissue_dict['Kssw'] # semi-solid-Water Kssw (Hz)        

        self.Kws = self.Ksw * self.M0s # water-solute Kws (Hz)
        self.Kwss = self.Kssw * self.M0ss # water-semi-solid Kwss (Hz)
        self.Kwa = self.Kaw * self.M0a # water-aliphatic Kwa (Hz)
        
        # these are typically set to zero
        self.KsSS = tissue_dict['KsSS'] # solute-semi-solid KsSS (Hz)
        self.KSSs = tissue_dict['KSSs'] # semi-solid-solute KSSs (Hz)
        self.Kas = tissue_dict['Kas'] # aliphatic-solute Kas (Hz)
        self.Ksa = tissue_dict['Ksa'] # solute-aliphatic Ksa (Hz)
        self.Kass = tissue_dict['Kass'] # aliphatic-semi-solid Kass (Hz)
        self.Kssa = tissue_dict['Kssa'] # semi-solid-aliphatic Kssa (Hz)

        self.B0shift = tissue_dict['B0shift'] # B0 inhomogeneity (ppm) 
        self.B1scaling = tissue_dict['B1scaling'] # B1 inhomogeneity ([0...1]) 
       
        self.r1s = self.R1s + self.Ksw + self.Ksa + self.KsSS
        self.r1a = self.R1a + self.Kaw + self.Kas + self.Kass
        self.r1ss = self.R1ss + self.Kssw + self.KSSs + self.Kssa
        self.r1w = self.R1w + self.Kws + self.Kwa + self.Kwss

        self.r2w = self.R2w + self.Kws + self.Kwa + self.Kwss
        self.r2s = self.R2s + self.Ksw + self.Ksa + self.KsSS
        self.r2a = self.R2a + self.Kaw + self.Kas + self.Kass
        self.r2ss = self.R2ss + self.Kssw + self.Kssa + self.KSSs


    def parse_acquisition_params(self,acq_dict):
        self.imaging_FA = acq_dict['imaging_FA'] #flip angles (degrees) for the imaging pulses
        
        self.imaging_time_duration_ms = acq_dict['imaging_time_duration']        
        
        self.sat_pulse_duration_ms = acq_dict['sat_pulse_duration']         
        
        self.TR_ms = np.array(acq_dict['TR'])
        self.TR = 1e-3 * self.TR_ms # TR (seconds)
        
        self.field_strength = acq_dict['field_strength'] #magnet field strength (const) (Tesla)
        
        self.sat_pulse_B1_uT = np.array(acq_dict['sat_pulse_B1']) #B1 of the saturation pulse (uT)
        self.sat_pulse_B1_Hz = 1e-6 * self.gamma * self.sat_pulse_B1_uT # saturation pulse B1 (Hz)                 
        
        # now parse the offset parameters
        self.sat_pulse_freq_Hz = acq_dict['sat_pulse_freq'] #offset at which the saturation takes place (Hz) 
        self.solute_freq = acq_dict['solute_freq'] #offset of the solute (ppm)
        self.water_freq = acq_dict['water_freq'] #offset of the water  (usually taken to be 0ppm)
        self.semi_solid_freq = acq_dict['semi_solid_freq'] #offset of the water  (usually taken to be 0ppm)
        self.aliphatic_freq = acq_dict['aliphatic_freq'] #offset of the water  (usually taken to be 0ppm)
                
        # parse the multi-slice sequence parameters
        self.time_per_slice = acq_dict['time_per_slice']
        self.slice_order = np.array(acq_dict['slice_order'])        
    
    def calculate_freq_deltas(self):
        """
        calculates the frequency differences between the various pools                        
        """        
        # convert to ppm (even though we'll convert to Hz in the next line) for compatibility with remainer of code
        self.sat_pulse_freq = self.sat_pulse_freq_Hz / (1e-6 * self.B0) # Hz to ppm

        self.delta_water_Hz = 1e-6 * self.B0 * (self.water_freq - self.sat_pulse_freq - self.B0shift) # Water, Hz
        delta_water = 2 * math.pi * self.delta_water_Hz # convert from Hz to rads/s
        
        
        self.delta_solute_Hz = 1e-6 * self.B0 * (self.solute_freq - self.sat_pulse_freq - self.B0shift) # Solute, Hz
        delta_solute = 2 * math.pi * self.delta_solute_Hz # convert from Hz to rads/s

        self.delta_semi_solid_Hz = 1e-6 * self.B0 * (self.semi_solid_freq- self.sat_pulse_freq - self.B0shift) # Semi-solid, Hz
        delta_semi_solid = 2 * math.pi * self.delta_semi_solid_Hz # convert from Hz to rads/s

        self.delta_aliphatic_Hz = 1e-6 * self.B0 * (self.aliphatic_freq - self.sat_pulse_freq - self.B0shift) # Aliphatic, Hz
        delta_aliphatic  = 2 * math.pi * self.delta_aliphatic_Hz # convert from Hz to rads/s

        # The build_matrix_A() method expects a 1D array so we need to reshape the deltas here. 
        if (len(delta_water.shape) > 1):
            delta_water = delta_water[:,0,None]
            delta_solute = delta_solute[:,0,None]
            delta_semi_solid = delta_semi_solid[:,0,None]
            delta_aliphatic = delta_aliphatic[:,0,None]                  
                
        return delta_water, delta_solute, delta_semi_solid, delta_aliphatic
    
    def calculate_mt_lineshape(self):
        """
        calculates the MT lineshape. Lorentzian only for now although super-Lorentzian can be added here later 
        """
        x = (self.delta_semi_solid_Hz * self.T2ss)**2
        g_b = self.T2ss/(math.pi*(1+x)) # Lorentzian
        mt_lineshape = (self.sat_pulse_B1_Hz**2) * math.pi * g_b          
        
        mt_lineshape = mt_lineshape[:,None]
        
        return mt_lineshape

    def run_sequence(self):
        # define quantities used in the vector definition
        Min  = self.build_matrix_b(self.M0s,self.M0a,self.M0ss,self.M0w)
        Mout = Min[:,:,None]
                
        traj = torch.zeros(len(self.M0s),len(self.TR)).cuda(device=self.gpu)
        
        # loop over the TRs
        for ii in range(len(self.TR)):
            print ('Processing image {}/{}'.format(ii,len(self.TR)))
            
            omega1 = 2 * math.pi * self.sat_pulse_B1_Hz[ii] * self.B1scaling # rads/s 
            mt_line = self.mt_lineshape[:,ii]
            mt_line = mt_line[:,None] # cast for compatibility with r1ss in subsequent code           
            sat_pulse_duration = 1e-3 * self.sat_pulse_duration_ms[ii] #duration of the saturation pulse (seconds)
                        
            # we assume that the saturation offset is read from the schedule (i.e. is an array) so we need to set 
            # the relevant quantities here for each iteration 
            self.delta_water = self.delta_water_array
            self.delta_solute = self.delta_solute_array
            self.delta_semi_solid = self.delta_semi_solid_array
            self.delta_aliphatic = self.delta_aliphatic_array

            Mout = self.calc_cest_signal(Mout,omega1,sat_pulse_duration,mt_line) # formatted as [tissues, pools]
            
            # record the signal as the water longitudinal magnetization
            traj[:,ii,None] = Mout[:,11]
                                                
        return Mout, traj.t().cpu().numpy()
            
    def calc_cest_signal(self,Min,omega1,sat_pulse_duration,mt_line):
        """
        calculate the cest signal produced by a saturation pulse 
        """
        
        A = self.build_matrix_A(omega1,mt_line)
       
        # define quantities used in the vector definition 
        bb = self.build_matrix_b(self.R1s*self.M0s, self.R1a*self.M0a, self.R1ss*self.M0ss, self.R1w*self.M0w)
                    
        Mp = torch.matmul(-torch.inverse(A),bb[:,:,np.newaxis])

        C = torch.matrix_exp(A*sat_pulse_duration)
        D = Min-Mp
        Mout = Mp + torch.matmul(C,D)
        
        return Mout      
                       
    def build_matrix_A(self,omega1,mt_line):
        """
        this function builds the system matrix A used to calculate the effect of 
        a saturation pulse

        """        
        
        # define a torch tensor to hold the tissue matrix
        # it's important to specify the dtype as float64 to maintain the desired accuracy
        A = torch.zeros([len(self.T1w),3*self.num_pools,3*self.num_pools],dtype=torch.float64, device=self.gpu)

        # define the rows and columns where we have data (everything else is zero by default)                        
        row = [0,0,0,0,0,\
               1,1,1,1,1,1,
               2,2,2,2,2,
               3,3,3,3,3,
               4,4,4,4,4,4,
               5,5,5,5,5,
               6,6,6,6,6,
               7,7,7,7,7,7,
               8,8,8,8,8,
               9,9,9,9,9,
               10,10,10,10,10,10,
               11,11,11,11,11
               ] # row indices where data is 

        col = [0,1,3,6,9,\
               0,1,2,4,7,10,
               1,2,5,8,11,
               0,3,4,6,9,
               1,3,4,5,7,10,
               2,4,5,8,11,
               0,3,6,7,9,
               1,4,6,7,8,10,
               2,5,7,8,11,
               0,3,6,9,10,
               1,4,7,9,10,11,
               2,5,8,10,11
               ] # col indicies where data is 

        data = [-self.r2s, self.delta_solute, self.Kas, self.KSSs, self.Kws,
                -self.delta_solute, -self.r2s, omega1, self.Kas, self.KSSs, self.Kws,
                -omega1, -self.r1s, self.Kas, self.KSSs, self.Kws,
                self.Ksa, -self.r2a, self.delta_aliphatic, self.Kssa, self.Kwa,
                self.Ksa, -self.delta_aliphatic, -self.r2a, omega1, self.Kssa, self.Kwa,
                self.Ksa, -omega1, -self.r1a, self.Kssa, self.Kwa,
                self.KsSS, self.Kass, -self.r2ss, self.delta_semi_solid, self.Kwss,
                self.KsSS, self.Kass, -self.delta_semi_solid, -self.r2ss, omega1, self.Kwss,
                self.KsSS, self.Kass, -omega1, -self.r1ss-mt_line, self.Kwss, 
                self.Ksw, self.Kaw, self.Kssw, -self.r2w, self.delta_water,
                self.Ksw, self.Kaw, self.Kssw, -self.delta_water, -self.r2w, omega1,
                self.Ksw, self.Kaw, self.Kssw, -omega1, -self.r1w
                ] # data to insert

        # loop over the row/cols and insert the data in the appropriate location
        for ii in range(len(row)):
            A[:,row[ii],col[ii]] = torch.DoubleTensor(np.squeeze(data[ii])).cuda(device=self.gpu)
        
        return A 
            
    def build_matrix_b(self,b2,b5,b8,b11):

        # define a torch tensor to hold the b matrix
        bb = torch.zeros([len(self.T1w),3*self.num_pools],dtype=torch.float64, device=self.gpu)
                                  
        row = [2,5,8,11]        
        
        # to build the sparse matrix we need to convert the array to a 1D list 
        data = [b2, b5, b8, b11]
                  
        # loop over the row/col
        for ii in range(len(row)):            
            bb[:,row[ii]] = torch.DoubleTensor(np.squeeze(data[ii])).cuda(device=self.gpu)
        
        return bb
        
def main():
    # define tissue dictionary
    # the conversion to a numpy array is needed so we can operate on the list 
    # in the code below

    # These values can be modified. If an array of values is defined the signals for all entries are calculated in parallel. 
    solute_concentration_range = np.array([600]) # mM
    semi_solid_concentration_range = np.array([10e3]) # mM
    aliphatic_concentration_range = np.array([0]) #mM
    water_proton_concentration = np.array(2*55e3) # mM, 55M * 2 exchangeable protons
    num_exchangeable_protons_solute = 1
    
    mole_fraction_range_solute = solute_concentration_range * num_exchangeable_protons_solute/water_proton_concentration
    mole_fraction_range_semi_solid = semi_solid_concentration_range/water_proton_concentration
    mole_fraction_range_aliphatic = aliphatic_concentration_range/water_proton_concentration

                    
    tissue_ranges= {'T1w': np.array([1000.0]), 'T2w': np.array([80.0]), #ms
                        'T1s': np.array([1450.0]), 'T2s':np.array([4.0]), #ms
                        'T1ss': np.array([1450.0]), 'T2ss':np.array([0.02]), #ms
                        'T1a':np.array([1450.0]),'T2a': np.array([1.0]), #ms
                        'Ksw':np.array([35.0]),'Kssw':np.array([25.0]),'Kaw':np.array([10.0]),
                        'Kas':np.array([0.0]),'Ksa': np.array([0.0]),'KsSS': np.array([0.0]),
                        'KSSs':np.array([0.0]),'Kass':np.array([0.0]),'Kssa': np.array([0.0]), #Hz
                        'M0s':mole_fraction_range_solute,
                        'M0ss':mole_fraction_range_semi_solid,
                        'M0a':mole_fraction_range_aliphatic, #mM
                        'B0shift': np.array([0.0]), # Hz
                        'B1scaling': np.array([1.0]) #a.u.
                    }
                        
    acquisition_params = {'imaging_FA': np.array([90]),#degrees
            'imaging_time_duration':np.array([24]), #ms
            'sat_pulse_duration':np.array([16]), #ms
            'TR':np.array([3500]), #ms
            'field_strength': 3, # Teslas
            'sat_pulse_B1':np.array([1]), #uT
            'sat_pulse_freq':np.array([447]), #Hz, Amide resonance freq.
            'solute_freq': np.array([3.5]), #ppm, Amide
            'water_freq': np.array([0]), #ppm
            'semi_solid_freq': np.array([-2.5]), #ppm
            'aliphatic_freq': np.array([-3.5]), #ppm            
            'normalize_flag': 1,
            'time_per_slice': np.array([64]), #ms  
            'slice_order':0
            }    
    
    seq_params = {'seq_name': 'mrf_cest_torch_seg',\
              'num_pools': 4,
              'spin_grad_flag': 'grad',
              'seg_dict_entries': 1
              }
    
    mrf_cest = MrfCest(tissue_ranges,acquisition_params,seq_params)                
    _, traj = mrf_cest.run_sequence()    

if __name__ == '__main__':
  main()       
        
