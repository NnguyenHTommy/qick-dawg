'''
Correlation spectroscopy of C13 
===================================================================
https://pubs.rsc.org/en/content/articlelanding/2015/fd/c5fd00113g
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.arqick.standard_ops import StandardOps


class CorrelationC13(StandardOps, NVAveragerProgram):
    '''
    Correlation spectroscopy of C13 with sub-nanosecond resolution
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg",  # microwave freq
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW gain
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start

        "delay_tdds_gate_crxpi2",
        "n_cpmg_gate_crxpi2",
    ]

    def initialize(self):
        self.init()

        self.delay_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='delay',
            init_val=0,
        )

        self.add_sweep(
            NVQickSweep(
                self,
                self.delay_register,
                self.cfg.delay_tdds_start,
                self.cfg.delay_tdds_end,
                self.cfg.nsweep_points,
            )
        )

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.pmod_trigger_sequence()

        # reset things that need to be reset each sequence iteration
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

        # wait 
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.pi_len_unused_tdds - self.half_pi_len_unused_tdds + self.pi_to_pi2_correction_tdds - self.pi2_to_pi2_correction_tdds)
        # e RY(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(90))
        self.offset_computations(pi2_after=True, delay_tau_tdds=0)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()   

        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=1, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(180))
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)

