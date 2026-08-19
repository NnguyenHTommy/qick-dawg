'''
Duration-sweep Rabi -- Hermite pulse -- ARQICK
=======================================================================
TODO: n_cpmg/duration point counts for the actual run (currently sized
for a coarse sweep) may need adjusting once tomorrow's run parameters
are settled.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
import numpy as np


def hermite_envelope(n):
    """(1 - x^2) * exp(-x^2) sampled over x in [-2.5009, 2.5009] -- truncated where
    the envelope falls to 1% of peak, matching the simulation's sigma_frac = 0.199932.
    Peak-normalized to 1. Bipolar (one negative lobe per side, min -0.135)."""
    n = int(n)
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    x = np.linspace(-2.5009, 2.5009, n)
    env = (1.0 - x**2) * np.exp(-x**2)
    return env / np.max(np.abs(env))


class RabiFineRes(NVAveragerProgram):
    '''
    Single-point program: one Hermite MW pulse of fixed duration and gain,
    triggers ARTIQ for laser/readout, no hardware duration sweep. Meant to
    be compiled and run once per point in an ARTIQ-side duration loop.
    '''
    required_cfg = [
        "mw_channel",
        "mw_pulse_tdds",       # this point's pulse duration (200ps units)
        "mw_nqz",               # 1 for f < 2.495 GHz
        "freq_freg",
        "mw_gain",
        "reps",
        "pmod_out_pin",                        # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",            # 50ns is reasonable
        "pmod_out_trig_delay_treg",              # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns
        "pulse_seq_delay_treg",                  # delay between pulse seq end and next trigger
    ]

    def initialize(self):
        self.check_cfg()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Single Hermite envelope at this point's duration
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_tdds)
        data[:self.cfg.mw_pulse_tdds] = hermite_envelope(self.cfg.mw_pulse_tdds)
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.program_pulse()

        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def program_pulse(self):
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
