'''
program to test the phase difference between switching IQ and NCO phase
=======================================================================
Min resolution of 200ps for steps between pulses

'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
import numpy as np

class MWPulseIQtest(NVAveragerProgram):
    '''
    Microwave pulse with envelope at 200ps resolution
    '''
    required_cfg = [        
        "mw_duration_tdds",
        "btwn_mw_delay_tdds",
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg" # should be 209.27ns 
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version rfsoc 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        # Configure the waveforms for different fine resolution pulse steps

        # Waveforms must have at least a length of 3 treg units. will check
        self.mw_pulse_sequence_len_tdds = 4*(self.cfg.mw_duration_tdds + self.cfg.btwn_mw_delay_tdds)
        if np.floor(self.mw_pulse_sequence_len_tdds) < 3:
            print("Error: total pulse sequence length is less than 3 treg units. Increase mw duration or delay between pulses")
            raise ValueError
        if self.mw_pulse_sequence_len_tdds % self.samps_per_clk != 0:
            print("Error: total pulse sequence length is not a multiple of samps per clk. Adjust mw duration or delay between pulses")
            raise ValueError

        # making the envelope for the pulse sequence which is X, Y, -X, -Y supposedly
        i_data = np.zeros(self.mw_pulse_sequence_len_tdds)
        q_data = np.zeros(self.mw_pulse_sequence_len_tdds)
        i_data[:self.cfg.mw_duration_tdds] = 1
        q_data[:self.cfg.mw_duration_tdds] = 1
        i_data[(self.cfg.mw_duration_tdds + self.cfg.btwn_mw_delay_tdds):(self.cfg.mw_duration_tdds*2 + self.cfg.btwn_mw_delay_tdds)] = -1
        q_data[(self.cfg.mw_duration_tdds + self.cfg.btwn_mw_delay_tdds):(self.cfg.mw_duration_tdds*2 + self.cfg.btwn_mw_delay_tdds)] = 1
        i_data[(self.cfg.mw_duration_tdds*2 + self.cfg.btwn_mw_delay_tdds*2):(self.cfg.mw_duration_tdds*3 + self.cfg.btwn_mw_delay_tdds*2)] = -1
        q_data[(self.cfg.mw_duration_tdds*2 + self.cfg.btwn_mw_delay_tdds*2):(self.cfg.mw_duration_tdds*3 + self.cfg.btwn_mw_delay_tdds*2)] = -1
        i_data[(self.cfg.mw_duration_tdds*3 + self.cfg.btwn_mw_delay_tdds*3):(self.cfg.mw_duration_tdds*4 + self.cfg.btwn_mw_delay_tdds*3)] = 1
        q_data[(self.cfg.mw_duration_tdds*3 + self.cfg.btwn_mw_delay_tdds*3):(self.cfg.mw_duration_tdds*4 + self.cfg.btwn_mw_delay_tdds*3)] = -1
        
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse_seq", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq", phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)

        # another pulse sequence with different NCO phase
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq", phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq", phase=self.deg2reg(-90))
        self.pulse(ch=self.cfg.mw_channel)

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq", phase=self.deg2reg(180))
        self.pulse(ch=self.cfg.mw_channel)

        self.sync_all()
