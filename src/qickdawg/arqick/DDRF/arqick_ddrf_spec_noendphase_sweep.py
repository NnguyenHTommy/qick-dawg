'''
DDRf spec with no end phase so just set the last pi/2 pulse to be the correct phase
=======================================================================
Min resolution of 200ps for delay steps between pulses in CPMG XY8 sequence
using fine control of waveform start address and phase.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.arqick.standard_ops import StandardOps

class DDRFSpecFineRes(StandardOps, NVAveragerProgram):
    '''
    DDRF CPMG XY8 sub-nanosecond resolution pulsing program with DDRF
    '''
    required_cfg = [        
        "mw_pi2_tdds", # length of pi/2 pulse
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
        
        "delay_tdds",
        "n_cpmg", 
    ]

    def initialize(self):
        self.init()     
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.pmod_trigger_sequence()

        self.tdds_offset_register.reset()
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds, n_cpmg_pulses=self.cfg.n_cpmg)

        # Final tau and readout pi/2 pulse.
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            waveform="half_pi_0",
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
            phase=self.deg2reg(-90),
        )
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)