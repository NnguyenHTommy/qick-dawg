'''
Rabi sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for steps between pulses in Rabi sequence
using fine control of waveform start address and phase.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np

class RabiFineRes(NVAveragerProgram):
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [        
        "mw_duration_tdds_start",
        "mw_duration_tdds_end",
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
        # Configure the waveforms for different fine resolution pulse steps
        # Waveforms must have at least a length of 3 treg units but want multiple of 2 so 64 = 2^6 for 16 samps per clk
        self.mw_pulse_waveform_len_tdds = 4 * self.samps_per_clk  # in tdds units
        
        for i in np.arange(0, self.mw_pulse_waveform_len_tdds+1, 1):
            i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
            q_data = np.zeros(self.mw_pulse_waveform_len_tdds)
            i_data[:i] = 1
            q_data[:i] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pulse_{i}", idata=i_data, qdata=q_data)

        # default special registers but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr') # for specifying which waveform
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
        
        # mw duration register
        self.mw_duration_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_duration',
                                                   init_val=self.cfg.mw_duration_tdds_start)

        # mw coarse and fine pulse loop register
        self.coarse_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_coarse',
                                                   init_val=0)
        self.fine_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_fine',
                                                   init_val=0)

        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_duration_register,
                                   start=self.cfg.mw_duration_tdds_start,
                                   stop=self.cfg.mw_duration_tdds_end,
                                   num=self.cfg.nsweep_points))

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)


        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pulse_{self.mw_pulse_waveform_len_tdds}")
        # set coarse and fine registers based on duration
        self.bitwi(self.coarse_mw_register.page, self.coarse_mw_register.addr, self.mw_duration_register.addr, ">>", int(np.log2(self.mw_pulse_waveform_len_tdds)))
        # if not 0 subtract 1 but if 0 then jump after the coarse loop
        self.coarse_mw_register.set_to(self.coarse_mw_register, "-", 1, physical_unit=False) 
        self.bitwi(self.fine_mw_register.page, self.fine_mw_register.addr, self.mw_duration_register.addr, "&", self.mw_pulse_waveform_len_tdds - 1)

        
        self.label("LOOP_COARSE")
        self.pulse(ch=self.cfg.mw_channel)
        self.loopnz(
                self.coarse_mw_register.page,
                self.coarse_mw_register.addr,
                "LOOP_COARSE")

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_16")
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)
    
    