'''
Duration-sweep Rabi -- Sinc pulse (Python-side sweep)
=======================================================================
The square-pulse RabiFineRes sweeps duration in hardware with a
periodic-coarse + truncated-fine envelope trick that fundamentally
requires a FLAT pulse: truncating a square wave to any length still
yields a square wave. Truncating a Sinc pulse does NOT yield the same
shape at a shorter duration -- short truncations mostly play the
near-zero leading tail and barely rotate the spin.

This module therefore sweeps duration on the Python side: for each
requested duration it builds a self-similar Sinc envelope (the shape
stretched to that duration), compiles a small single-point program, and
acquires it. Cost scales as (#points) x (per-point overhead), so keep
the duration list to tens of points, not thousands -- the amplitude
(gain) Rabi in rabi_fine_res_amp.py remains the fast, high-resolution
calibration scan; this is the shape-faithful duration analogue.

The returned data object exposes the same keys ReadoutHelpers'
analyze_results_helper() produces (signal1/2, reference1/2, *_cts_s,
contrast, and an mw_duration_ftsamp/ftns/ftus sweep axis), so
Visualizer.plot_rabi() works on it directly.

get_reference=True is fully supported here (gain is static per point,
so the base readout_and_reference()'s zero-then-restore-cfg.mw_gain
logic is exactly correct), and is recommended -- it enables the
contrast view in Visualizer.plot_rabi().
'''

from copy import copy
from itemattribute import ItemAttribute
from tqdm.auto import tqdm

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.finetimingsuite.readout_helpers import ReadoutHelpers
import numpy as np


def sinc_envelope(n):
    """sin(pi x)/(pi x) sampled over x in [-6.93, 6.93] -- SEVEN lobes per side,
    truncated where the envelope falls to 1% of peak, matching the simulation's
    sigma_frac = 0.072150. Peak-normalized to 1. Strongly bipolar.
    NOTE: rotation is driven by the SIGNED area, so the negative lobes subtract."""
    n = int(n)
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.ones(1)
    x = np.linspace(-6.93, 6.93, n)
    env = np.sinc(x)  # np.sinc is the NORMALIZED sinc: sin(pi x)/(pi x)
    return env / np.max(np.abs(env))


class RabiFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    Single-point program: one Sinc MW pulse of fixed duration and gain,
    standard init/readout(/reference) cycle, no hardware sweep. Used by
    sweep_rabi_duration() once per duration point.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod",        # 0 for PMOD0_0

        # MW pulse parameters
        "mw_pulse_ftsamp",        # This point's pulse duration (fine-resolution samples)
        "mw_nqz",                 # 1 for f < 2.495 GHz
        "mw_freg",
        "mw_gain",

        # Readout and delays
        "mw_to_laser_delay_treg", # Laser turn-on lag relative to MW
        "relax_delay_treg",       # Spin relaxation delay (post laser reinitialization)

        # Readout
        "laser_on_treg",
        "readout_reference_start_treg",
        "readout_integration_treg",
        "laser_readout_offset_treg",

        # Other
        "reps",
        "pre_init",
        "get_reference",  # Whether to acquire a reference readout with MW gain = 0
    ]

    def initialize(self):
        self.check_cfg()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)

        self.setup_readout()

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Single Sinc envelope at this point's duration
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_ftsamp / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        data[:self.cfg.mw_pulse_ftsamp] = sinc_envelope(self.cfg.mw_pulse_ftsamp)
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        self.pre_init()

    def body(self):
        self.initialize_spin()
        self.program_pulse()
        self.readout_and_reference(self.program_pulse)

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

    iterator = tqdm(durations_ftns, disable=not progress, desc="sinc duration Rabi")
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
    d.pulse_shape = 'sinc'

    return d
