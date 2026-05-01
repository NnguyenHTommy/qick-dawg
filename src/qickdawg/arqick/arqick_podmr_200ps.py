'''
PDOMR
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
import numpy as np
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep

class PDOMRFineRes(NVAveragerProgram):
    '''
    PDOMR
    '''
    required_cfg = [        
        "mw_duration_tdds", # length of mw
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
        "freq_start_freg", # units of freg are Mhz
        "freq_end_freg",
        "nsweep_points",
    
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version rfsoc 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        # Configure the waveforms for different fine resolution pulse steps
        # Waveforms must have at least a length of 3 treg units 
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_duration_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units
        
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
        
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")
        self.set_pulse_registers(ch=self.cfg.mw_channel, freq = self.cfg.freq_start_freg, waveform="pulse")

        self.add_sweep(NVQickSweep(self,
                                 self.mw_frequency_register,
                                 self.cfg.freq_start_fMHz,
                                 self.cfg.freq_end_fMHz,
                                 self.cfg.nsweep_points))
        
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.pulse(ch=self.cfg.mw_channel) 
        self.sync_all(self.cfg.pulse_seq_delay_treg)