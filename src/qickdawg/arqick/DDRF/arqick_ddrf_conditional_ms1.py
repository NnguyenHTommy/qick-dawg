'''
conditional rotation ms=1 with reduced swap in beginning with DDRF
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.arqick.standard_ops import StandardOps
import numpy as np

class ConditionalRotationMS1DDRF(StandardOps, NVAveragerProgram):

    required_cfg = [
        # params that usually won't change
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start
        
        # for the normal pulse part
        "mw_gain",  # MW gain
        "freq_freg",  # microwave freq
        "mw_pi2_tdds",  # length of pi/2 pulse

        "n_cpmg_sweep",
        "C13_measurement_delay_tdds",
        "delay_tdds_gate_crxpi2",
        "n_cpmg_gate_crxpi2",
    ]

    def initialize(self):
        self.init()
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.pmod_trigger_sequence()
        self.tdds_offset_register.reset()

        # NUCLEAR SPIN INITIALIZATION 
        # e RY(pi/2) 
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # e-n CROTY(-+pi/2)
        self.cpmg_xy8_gate(gate_index=1, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # wait for optical pump
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.C13_measurement_delay_tdds)
        
        # e RX(pi)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.delay_tdds_gate_crxpi2 - self.pi_to_pi_correction_tdds - self.pi_len_unused_tdds)
        # Coarse delay in treg units.
        self.bitwi(
            self.tdds_offset_register.page,
            self.treg_offset_register.addr,
            self.tdds_offset_register.addr,
            ">>",
            int(np.log2(self.samps_per_clk)),
        )

        # Fine offset in tdds units.
        self.bitwi(
            self.tdds_offset_register.page,
            self.tdds_offset_register.addr,
            self.tdds_offset_register.addr,
            "&",
            self.samps_per_clk - 1,
        )

        # Waveform select from fine offset.
        self.address_register.set_to(
            self.tdds_offset_register,
            '*',
            self.pi_waveform_len_treg + self.half_pi_waveform_len_treg,
            physical_unit=False,
        )
        self.address_register.set_to(
                self.address_register,
                '+',
                self.half_pi_waveform_len_treg,
                physical_unit=False,
            )
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.cfg.delay_tdds_gate_crxpi2) 

        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=2, pi_2_pulse_before=False, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_sweep)
        
       # e RY(-pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(-90))
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # e-n CROTY(+-pi/2)
        self.cpmg_xy8_gate(gate_index=3, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)