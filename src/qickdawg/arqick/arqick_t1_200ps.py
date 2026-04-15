'''
T1 sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for delay steps 
using fine control of waveform start address and phase.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np

class T1(NVAveragerProgram):
    '''
    T1 sub-nanosecond resolution pulsing program
    '''
    required_cfg = [        
        "mw_duration_tdds", # length of mw
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "scaling_mode", # 'linear' or 'exponential' spacing of delay points in sweep
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
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
        # self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_duration_tdds / self.samps_per_clk)), 3)
        # self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units
        
        self.mw_pulse_waveform_len_treg = max(int(np.ceil((self.cfg.mw_duration_tdds + self.samps_per_clk-1) / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in tdds units

        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=i_data, qdata=q_data)
        
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr') # for specifying which waveform

        self.pi_len_unused_tdds = self.mw_pulse_waveform_len_tdds - self.cfg.mw_duration_tdds

        self.tdds_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='tdds_offset',
                                                    init_val=-self.pi_len_unused_tdds)

        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
        
        # Set up register for storing and sweeping delays
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                            name='delay',
                                            init_val=self.cfg.delay_tdds_start)
        
        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self, 
                self.delay_register,
                self.cfg.delay_tdds_start,
                self.cfg.delay_tdds_end,
                self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor))
        else:
            self.add_sweep(NVQickSweep(
                self, 
                self.delay_register,
                self.cfg.delay_tdds_start,
                self.cfg.delay_tdds_end,
                self.cfg.nsweep_points))
            
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.tdds_offset_register.reset() # reset the dds_offset adjustment
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse")
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.offset_computations() # compute offsets and set waveform address and phase
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.sync_all(self.cfg.pulse_seq_delay_treg)
    
    def offset_computations(self):
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)
        # Computes how long to stall the FPGA output in tproc cycles from the total delay.
        self.bitwi(self.tdds_offset_register.page, self.treg_offset_register.addr, self.tdds_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # Computes the remaining samples that the pulse should be delayed by
        self.bitwi(self.tdds_offset_register.page, self.tdds_offset_register.addr, self.tdds_offset_register.addr, "&", self.samps_per_clk - 1)

        # updating address register to select correct waveform based on the current offset
        self.address_register.set_to(self.tdds_offset_register, '*', self.mw_pulse_waveform_len_treg, physical_unit = False)
