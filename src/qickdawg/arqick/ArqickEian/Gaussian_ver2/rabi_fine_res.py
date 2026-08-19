'''
Duration-sweep Rabi -- Gaussian pulse -- ARQICK (ARTIQ+QICK) variant
=======================================================================
Same reasoning as the QICK-only version: truncating a Gaussian envelope
doesn't yield a shorter Gaussian pulse, so duration can't be swept
on-chip the way square's periodic-replay trick does. Duration is still
swept point-by-point, but the loop now lives on the ARTIQ side (one
run_rounds() call per duration, same pattern as arqick_artiq_red's
pulse_qick(), just called once per point instead of once total) --
this file only defines the single-duration program that gets compiled
and triggered per point.

Readout is no longer done on this board: no laser gating, no photon
counting here. This program triggers ARTIQ's sequence once per rep and
plays the pulse; ARTIQ handles laser/readout externally on that trigger.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
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


class RabiFineRes(NVAveragerProgram):
    '''
    Single-point program: one Gaussian MW pulse of fixed duration and gain,
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

        # Single Gaussian envelope at this point's duration
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_tdds / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_tdds = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_tdds)
        data[:self.cfg.mw_pulse_tdds] = gaussian_envelope(self.cfg.mw_pulse_tdds)
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

def sweep_rabi_duration(config, durations_ftns, progress=True):
    """
    Run a shape-faithful duration-sweep Rabi: one compiled single-point
    program per duration in `durations_ftns` (nanoseconds).

    Parameters
    ----------
    config : NVConfiguration
        Same config you would hand any suite program. mw_gain is the fixed
        drive amplitude; get_reference=True is supported and recommended.
    durations_ftns : array-like of float
        Pulse durations to measure, in ns. Keep this to tens of points --
        each point compiles and runs its own program (see module docstring).
    progress : bool
        Show a tqdm bar over duration points.

    Returns
    -------
    ItemAttribute with signal1/reference1 (+2 if get_reference), *_cts_s,
    contrast (if get_reference), and mw_duration_ftsamp/ftns/ftus axes --
    directly compatible with Visualizer.plot_rabi().
    """
    durations_ftns = np.asarray(durations_ftns, dtype=float)
    assert durations_ftns.ndim == 1 and len(durations_ftns) >= 2, \
        "durations_ftns must be a 1D list of at least 2 durations (in ns)"

    get_reference = bool(config.get_reference)
    readouts = 4 if get_reference else 2

    per_point = {k: [] for k in
                 (['signal1', 'reference1', 'signal2', 'reference2'] if get_reference
                  else ['signal1', 'reference1'])}
    actual_ftsamp = []
    sample2us = None

    iterator = tqdm(durations_ftns, disable=not progress, desc="gaussian duration Rabi")
    for dur_ns in iterator:
        cfg = copy(config)
        cfg.mw_pulse_ftns = float(dur_ns)  # NVConfiguration converts to mw_pulse_ftsamp

        prog = RabiFineRes(cfg)

        if sample2us is None:
            sample2us = prog.soccfg.cycles2us(1) / prog.samps_per_clk
        actual_ftsamp.append(int(cfg.mw_pulse_ftsamp))

        # raw_data=True -> ReadoutHelpers.acquire returns the raw buffer,
        # shape (reps, readouts); sum over reps per readout slot.
        raw = prog.acquire(raw_data=True, progress=False)
        raw = np.reshape(raw, (cfg.reps, readouts))

        per_point['signal1'].append(raw[:, 0].sum())
        per_point['reference1'].append(raw[:, 1].sum())
        if get_reference:
            per_point['signal2'].append(raw[:, 2].sum())
            per_point['reference2'].append(raw[:, 3].sum())

    d = ItemAttribute()
    norm_factor = config.readout_integration_tns * 1e-9 * config.reps

    for key, vals in per_point.items():
        d[key] = np.asarray(vals, dtype=float)
        d[key + '_cts_s'] = d[key] / norm_factor

    if get_reference:
        d.contrast = d.signal1 / d.signal2

    ftsamp = np.asarray(actual_ftsamp, dtype=int)
    ftus = ftsamp * sample2us
    d.mw_duration_ftsamp = ftsamp
    d.mw_duration_ftus = ftus
    d.mw_duration_ftns = ftus * 1000.0
    d.sweep_pts = ftsamp
    d.sweep_param = 'mw_duration_ftsamp'
    d.pulse_shape = 'gaussian'

    return d
