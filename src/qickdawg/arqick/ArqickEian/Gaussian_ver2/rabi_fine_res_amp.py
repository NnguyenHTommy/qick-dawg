'''
Amplitude (gain) Rabi -- Gaussian pulse -- ARQICK (ARTIQ+QICK) variant
=======================================================================
Same on-chip live-register gain sweep as the QICK-only Gaussian
RabiGainFineRes: one fixed-duration Gaussian envelope, gain swept via
NVQickSweep on the live "gain" pulse register (no per-point recompile
needed -- amplitude is a live value, not baked into the envelope, so
this works identically for all four pulse shapes, unlike the duration
sweep). The only change is the readout mechanism: laser gating and
photon counting are no longer done on this board -- this program
triggers ARTIQ's sequence once per rep and plays the pulse; ARTIQ
handles laser/readout externally on that trigger.

mw_gain_register is fetched directly via get_gen_reg(name='gain') --
'gain' is a standard built-in generator register, same as 'addr'/
'phase' elsewhere in this suite, so no setup_helper_registers() call
is needed (confirmed unavailable without the ReadoutHelpers mixin).
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np


def gaussian_envelope(n):
    """exp(-x^2 / 2) sampled over x in [-3.0334, 3.0334] -- truncated where the
    envelope falls to 1% of peak, matching the simulation's sigma_frac = 0.164832.
    Peak-normalized to 1. Strictly positive -- no negative lobes."""
    n = int(n)
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    x = np.linspace(-3.0334, 3.0334, n)
    env = np.exp(-x**2 / 2.0)
    return env / np.max(np.abs(env))


class RabiGainFineRes(NVAveragerProgram):
    '''
    Amplitude (gain) Rabi with a Gaussian fine-resolution MW pulse (ARTIQ-triggered readout)
    '''
    required_cfg = [
        "mw_channel",
        "mw_pulse_tdds",        # fixed MW pulse duration (200ps units)
        "mw_nqz",                # 1 for f < 2.495 GHz
        "freq_freg",
        "mw_gain_start",
        "mw_gain_end",
        "nsweep_points",
        "reps",
        "pmod_out_pin",                        # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",            # 50ns is reasonable
        "pmod_out_trig_delay_treg",              # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns
        "pulse_seq_delay_treg",                  # delay between pulse seq end and next trigger
    ]

    def initialize(self):
        self.check_cfg()

        if self.cfg.mw_gain_start < 0 or self.cfg.mw_gain_end < 0:
            assert 0, 'Smallest Microwave gain must be positive'
        elif self.cfg.mw_gain_start > 32767 or self.cfg.mw_gain_end > 32767:
            assert 0, 'Largest Microwave gain exceeds maximum value'
        assert self.cfg.nsweep_points >= 2, 'nsweep_points must be >= 2 (NVQickSweep divides by nsweep_points-1)'

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Single Gaussian envelope at the fixed duration -- amplitude comes
        # from the live gain register, not from per-point envelope data.
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_tdds)
        data[:self.cfg.mw_pulse_tdds] = gaussian_envelope(self.cfg.mw_pulse_tdds)
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

        self.mw_gain_register = self.get_gen_reg(self.cfg.mw_channel, name='gain')

        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain_start,
                                     waveform="pulse",
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_gain_register,
                                   start=self.cfg.mw_gain_start,
                                   stop=self.cfg.mw_gain_end,
                                   expts=self.cfg.nsweep_points))

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.program_pulse()

        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def program_pulse(self):
        # Bare pulse: the swept gain register IS the live pulse parameter;
        # do NOT re-arm with set_pulse_registers(gain=...) here (RuntimeError:
        # gain would be set in both default and set_pulse_registers).
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

