'''
Ramsey sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for delay steps between pulses in Ramsey sequence
using fine control of waveform start address and phase.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np

class RamseyFineRes(NVAveragerProgram):
    '''
    Ramsey sub-nanosecond resolution pulsing program
    '''
    required_cfg = [        
        "mw_pi2_tdds", # length of pi/2 pulse
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
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

        # Get samps per clk for later calculations. should be 16 for mw with current version 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Configure the waveforms for different fine resolution delay steps
        # Waveforms must have at least a length of 3 treg units
        self.half_pi_waveform_len_treg = max(int(np.ceil((self.cfg.mw_pi2_tdds + self.samps_per_clk-1) / self.samps_per_clk)), 3)
        self.pi2_to_pi2_correction_tdds = self.cfg.mw_pi2_tdds

        for i in range(16):
            # pi/2 pulse
            i_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            q_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            i_data[i : i + self.cfg.mw_pi2_tdds] = 1
            q_data[i : i + self.cfg.mw_pi2_tdds] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=i_data, qdata=q_data)

        # default special registers but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr') # for specifying which waveform

        # there is dead time in the waveforms due to the waveform length being in treg units and using dds units
        # need to account for this in delays so need the values 
        self.half_pi_len_unused_tdds = self.half_pi_waveform_len_treg*self.samps_per_clk - self.cfg.mw_pi2_tdds

        # two registers needed to account for unused time 
        # one for coarse adjustment (treg_offset) and one for fine adjustment (tdds_offset)
        self.tdds_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='tdds_offset',
                                                    init_val=-self.half_pi_len_unused_tdds)

        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain)
        
        # Set up register for storing and sweeping delays
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                            name='delay',
                                            init_val=0)
        
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
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=self.deg2reg(180))
        self.offset_computations() # compute offsets and set waveform address and phase
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)
    
    def offset_computations(self):
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)
        self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi2_to_pi2_correction_tdds)

        # Computes how long to stall the FPGA output in tproc cycles from the total delay.
        self.bitwi(self.tdds_offset_register.page, self.treg_offset_register.addr, self.tdds_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # Computes the remaining samples that the pulse should be delayed by
        self.bitwi(self.tdds_offset_register.page, self.tdds_offset_register.addr, self.tdds_offset_register.addr, "&", self.samps_per_clk - 1)

        # updating address register to select correct waveform based on the current offset
        self.address_register.set_to(self.tdds_offset_register, '*', self.half_pi_waveform_len_treg, physical_unit = False)
