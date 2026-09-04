'''
Duration-sweep Rabi -- Hermite pulse -- ARTIQ+QICK, ON-BOARD SWEEP VERSION
=======================================================================
CHANGE FROM ORIGINAL:
    The original program builds exactly ONE Hermite envelope, sized to
    self.cfg.mw_pulse_tdds, and is meant to be re-*compiled* once per
    duration point from the ARTIQ/Python side. Since the ARTIQ interface
    in use here cannot drive a "compile-time" duration sweep, this
    version instead preprograms EVERY Hermite envelope in the sweep at
    initialize() time (all loaded into gen envelope memory once), and
    selects among them at runtime via an index that ARTIQ writes using
    the same trigger/register path it already uses to drive the
    on-board duration sweep for square pulses.
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


# --- Memory budget -----------------------------------------------------
# Confirmed: 65536 samples of complex envelope memory for the gen channel.
ENVELOPE_MEM_SAMPLES = 65536


def envelope_len_samples(tdds, samps_per_clk):
    """Length (in samples, rounded up to a whole clock) that one Hermite
    pulse of duration `tdds` will actually occupy in envelope memory --
    mirrors the original file's rounding (max(ceil(tdds/spc), 3) * spc)."""
    return max(int(np.ceil(tdds / samps_per_clk)), 3) * samps_per_clk


def total_envelope_memory(durations_tdds, samps_per_clk):
    return sum(envelope_len_samples(t, samps_per_clk) for t in durations_tdds)


def max_sweep_points(t_min_tdds, t_max_tdds, samps_per_clk,
                      mem_budget=ENVELOPE_MEM_SAMPLES):
    """How many linearly-spaced duration points fit in mem_budget samples,
    given the sweep's min/max duration. Rounding overhead (up to
    samps_per_clk - 1 wasted samples per point) is ignored here since it's
    a small correction for realistic point counts -- use
    total_envelope_memory() on your real duration list for an exact check."""
    # sum_{i=0}^{N-1} (t_min + i*step) ~= N*(t_min+t_max)/2  for large N
    if t_min_tdds + t_max_tdds <= 0:
        return 0
    return int(mem_budget // ((t_min_tdds + t_max_tdds) / 2))


class RabiFineResOnboardSweep(NVAveragerProgram):
    '''
    Preloads every Hermite envelope needed for the duration sweep once,
    at compile time, then plays the one selected by an ARTIQ-written
    index each shot -- so ARTIQ can step the sweep by writing an index
    instead of triggering a Python-side recompile.
    '''
    # Matches the dqp's required_cfg exactly. NOTE: this list has no field
    # for where ARTIQ writes the per-shot duration index (see
    # MW_PULSE_INDEX_ADDR below) -- your dqp doesn't appear to pass one
    # through cfg, so confirm how/where that handshake actually happens
    # before trusting program_pulse()'s memri() call.
    required_cfg = [
        "mw_duration_tdds_start",
        "mw_duration_tdds_end",
        "nsweep_points",
        "freq_freg",             # Microwave freq
        "mw_channel",             # MW Channel
        "mw_nqz",                 # 1 at 1405 MHz
        "mw_gain",                # MW Gain
        "reps",
        "pmod_out_pin",                        # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",            # 50ns is reasonable
        "pmod_out_trig_delay_treg",              # delay between trigger and pulse seq start. added to the inherent ~198ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns
        "pulse_seq_delay_treg",                  # delay between pulse seq end and trigger start of next seq
    ]

    MW_PULSE_INDEX_ADDR = 0

    def initialize(self):
        self.check_cfg()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        if self.cfg.nsweep_points < 2:
            raise ValueError("nsweep_points must be >= 2")
        if self.cfg.mw_duration_tdds_end < self.cfg.mw_duration_tdds_start:
            raise ValueError("mw_duration_tdds_end must be >= mw_duration_tdds_start")

        # Build the duration list from start/end/nsweep_points -- ARTIQ
        # only needs to send the corresponding 0..N-1 index at runtime.
        raw_pts = np.linspace(self.cfg.mw_duration_tdds_start,
                               self.cfg.mw_duration_tdds_end,
                               self.cfg.nsweep_points)
        durations = np.rint(raw_pts).astype(int).tolist()


        if len(set(durations)) != len(durations):
            raise ValueError(
                "Duration sweep contains duplicate integer tdds values after "
                "rounding. Reduce nsweep_points or widen the start/end range."
            )

        # --- Hard memory-overflow guard (fail fast, not on hardware) ---
        used = total_envelope_memory(durations, self.samps_per_clk)
        if used > ENVELOPE_MEM_SAMPLES:
            raise ValueError(
                f"Preprogrammed duration sweep needs {used} envelope samples "
                f"on ch {self.cfg.mw_channel}, but only {ENVELOPE_MEM_SAMPLES} "
                f"are available (verify this limit against "
                f"soccfg['gens'][{self.cfg.mw_channel}]['maxlen']). "
                f"Reduce point count, shrink duration range, or coarsen "
                f"spacing at the long-duration end (see max_sweep_points())."
            )
        print(f"Duration sweep envelope memory: {used}/{ENVELOPE_MEM_SAMPLES} "
              f"samples ({100*used/ENVELOPE_MEM_SAMPLES:.1f}%), "
              f"{len(durations)} points.")

        self.pulse_names = []
        for i, tdds in enumerate(durations):
            wf_len = envelope_len_samples(tdds, self.samps_per_clk)
            data = np.zeros(wf_len)
            data[:tdds] = hermite_envelope(tdds)
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            name = f"pulse_{i}"
            self.add_envelope(ch=self.cfg.mw_channel, name=name, idata=data, qdata=data)
            self.pulse_names.append(name)

        # Register a default/first pulse; program_pulse() below is
        # responsible for actually swapping in the right one per shot.
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=self.pulse_names[0])

        # -- tProc v1 branch-table setup --
        self.idx_page = 0
        self.idx_reg = self.new_reg(self.idx_page, name="mw_pulse_idx")
        self.case_reg = self.new_reg(self.idx_page, name="mw_pulse_case")

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.program_pulse()

        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def program_pulse(self):
        # tProc v1 branch table: read this shot's duration index (ARTIQ
        # writes it to tProc data memory at mw_pulse_index_addr, the same
        # way it already drives the on-board square-pulse duration sweep),
        # then branch to the matching preloaded pulse's waveform.
        idx = self.idx_reg.addr
        case = self.case_reg.addr

        self.memri(self.idx_page, idx, self.MW_PULSE_INDEX_ADDR,
                   'mw_pulse_idx <- ARTIQ-written duration index')

        n = len(self.pulse_names)
        for i in range(n):
            self.regwi(self.idx_page, case, i, f'case const {i}')
            self.condj(self.idx_page, idx, "==", case, f"CASE_{i}")

        # tProc v1 has no unconditional jump/goto instruction -- only condj
        # (conditional) and loopnz. `self.goto(...)` doesn't exist and isn't
        # in the instruction set at all, so an unconditional jump is faked
        # with a condj whose condition is always true (a register compared
        # to itself).
        self.label("CASE_DEFAULT")
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=self.pulse_names[0])
        self.condj(self.idx_page, idx, "==", idx, "PULSE_SELECTED")

        for i, name in enumerate(self.pulse_names):
            self.label(f"CASE_{i}")
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=name)
            self.condj(self.idx_page, idx, "==", idx, "PULSE_SELECTED")

        self.label("PULSE_SELECTED")
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()