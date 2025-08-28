'''
RFTest
=======================================================================
RF Test class used to program sequences to evaluate the RFSoC RF Power
and Rise times.
'''




from qick.averager_program import QickSweep
from .nvaverageprogram import NVAveragerProgram
from itemattribute import ItemAttribute
from ..util import apply_on_axis_0_n_times


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import floor
import os


class RFPulse(NVAveragerProgram):
   '''
   An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
   '''
   required_cfg = [
       "pulse_len_treg",
       "relax_delay_treg",
       "mw_channel",
       "mw_nqz",
       "reps",
       "init_delay_treg",
       "gain",
       "num_pulses"]
  
   def initialize(self):
       # NVConfiguration class does not have Gain units unlike freq, time, or phase
       # need to call: cfg.add_unitless_linear_sweep(gain, start, stop, delta, nsweep_points)
       #self.check_cfg() #?


       # Get mw registers
       self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
       #self.setup_readout() # required




       self.set_pulse_registers(
           ch=self.cfg.mw_channel,
           style='const',
           freq=self.cfg.mw_freg,
           gain=self.cfg.gain,
           length=self.cfg.pulse_len_treg, # pulse len
           phase=0)
      
       self.synci(self.cfg.init_delay_treg)  # give processor some time to configure pulses




   def body(self):
       # Pulse MW channel
       for rep in range(self.cfg.num_pulses):
           self.pulse(ch=self.cfg.mw_channel) # rep is needed b/c need it on the timeline
           self.sync_all(self.cfg.relax_delay_treg)

