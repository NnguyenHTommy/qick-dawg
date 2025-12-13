'''
Bootstrap sub-nanosecond resolution pulsing program
=======================================================================
Min resolution of 200ps for delay steps 
using fine control of waveform start address and phase.
Follows https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.077601?__cf_chl_tk=OJit3IH_f_.tMb6W6g0SHQM6.uV36H9ieFmu3en4wj4-1759769753-1.0.1.1-67pXH.nAif8JKOAf_MDe0Gjfad2FLsFhsTxxyD9G7lo 
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
import numpy as np

class BootstrapFineRes(NVAveragerProgram):
    '''
    Bootstrap sub-nanosecond resolution pulsing program.
    '''
    required_cfg = [        
        "mw_pi2_tdds", # length of mw
        "btwn_mw_delay_tdds", # delay between mw pulses
        "bootstrap_experiment_number",
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

    # helper function to make pulse envelope for bootstrap sequences
    # returns I and Q data arrays
    def make_pulse_envelope(self):

        if self.cfg.bootstrap_experiment_number not in range(1,13):
            print("Error: bootstrap_experiment_number must be an integer between 1 and 12")
            raise ValueError
        
        # Configure the waveforms for different fine resolution pulse steps
        # Waveforms must have at least a length of 3 treg units 
        if self.cfg.bootstrap_experiment_number in [1,2]:
            self.mw_pulse_seq_len_tdds = self.cfg.mw_pi2_tdds 
        elif self.cfg.bootstrap_experiment_number in [3,4,5,6]:
            self.mw_pulse_seq_len_tdds = 3*self.cfg.mw_pi2_tdds + self.cfg.btwn_mw_delay_tdds
        elif self.cfg.bootstrap_experiment_number in [7,8]:
            self.mw_pulse_seq_len_tdds = 2*self.cfg.mw_pi2_tdds + self.cfg.btwn_mw_delay_tdds
        elif self.cfg.bootstrap_experiment_number in [9,10,11,12]:
            self.mw_pulse_seq_len_tdds = 4*self.cfg.mw_pi2_tdds + 2*self.cfg.btwn_mw_delay_tdds
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.mw_pulse_seq_len_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk  
        
        i_data = np.zeros(self.mw_pulse_waveform_len_tdds)
        q_data = np.zeros(self.mw_pulse_waveform_len_tdds)

        axis = {"x": (1, 1), "y": (-1, 1)}
        pulse_type = {"pi2": self.cfg.mw_pi2_tdds, "pi": 2 * self.cfg.mw_pi2_tdds}
        seqs = {
            1: [("pi2", "x")],
            2: [("pi2", "y")],
            3: [("pi2", "x"), ("delay", None), ("pi", "x")],
            4: [("pi2", "y"), ("delay", None), ("pi", "y")],
            5: [("pi", "y"), ("delay", None), ("pi2", "x")],
            6: [("pi", "x"), ("delay", None), ("pi2", "y")],
            7: [("pi2", "y"), ("delay", None), ("pi2", "x")],
            8: [("pi2", "x"), ("delay", None), ("pi2", "y")],
            9: [("pi2", "x"), ("delay", None), ("pi", "x"), ("delay", None), ("pi2", "y")],
            10: [("pi2", "y"), ("delay", None), ("pi", "x"), ("delay", None), ("pi2", "x")],
            11: [("pi2", "x"), ("delay", None), ("pi", "y"), ("delay", None), ("pi2", "y")],
            12: [("pi2", "y"), ("delay", None), ("pi", "y"), ("delay", None), ("pi2", "x")],
        }

        # fill i/q arrays
        idx = 0
        for typ, ax in seqs[self.cfg.bootstrap_experiment_number]:
            if typ == "delay":
                idx += self.cfg.btwn_mw_delay_tdds
                continue
            length = pulse_type[typ]
            i_val, q_val = axis[ax]
            i_data[idx : idx + length] = i_val
            q_data[idx : idx + length] = q_val
            idx += length

        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        return i_data, q_data

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version rfsoc 11/14/2025
        # if this changes from 16 then need to change waveform generation part
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']
        i_data , q_data = self.make_pulse_envelope()
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse_seq", idata=i_data, qdata=q_data)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = self.deg2reg(0))
        
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pulse_seq")
        self.pulse(ch=self.cfg.mw_channel) 
        self.sync_all(self.cfg.pulse_seq_delay_treg)