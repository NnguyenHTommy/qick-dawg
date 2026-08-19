'''
CPMG-XY Fine-Resolution Pulse Sequence -- Hermite pulse
=======================================================================
Identical timing/phase machinery to the verified working square-pulse
CPMGXYFineRes (XY8 phase cycling, sub-ns delay sweeping via waveform
offsetting, register arithmetic); the only change is the pi and pi/2
envelopes: 2nd-order Hermite-Gaussians, (1 - x^2) * exp(-x^2), instead
of flat tops. Each of the samps_per_clk offset variants holds the full
shape shifted by i samples, exactly as the square version holds a
shifted flat window.

As in the square version: n_cpmg = 0 -> Ramsey, n_cpmg = 1 -> Hahn echo.
The pi pulse duration is 2 * mw_pi2_ftsamp (self-similar stretch: at
fixed peak amplitude, pulse area scales with duration for any fixed
shape, so pi = 2x pi/2 duration carries over from the square convention).

NOTE: The interpulse delay (tau/2tau) is measured from the end of one
pulse's waveform window to the start of the next, same as the square
version. For shaped pulses the envelope amplitude near the window edges
is small (the shape's tails), so the effective pulse centers sit at the
window centers; tau retains its meaning as window-edge-to-window-edge
delay, identical convention to the square suite.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.finetimingsuite.readout_helpers import ReadoutHelpers

import numpy as np
from math import floor


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


class CPMGXYFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [
        # Hardware channels
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod",        # 0 for PMOD0_0

        # MW pulse parameters
        "mw_pi2_ftsamp",          # pi/2 pulse duration
        "mw_nqz",                 # 1 for f < 2.495 GHz
        "mw_freg",
        "mw_gain",
        "n_cpmg",

        # Delay sweep (tau)
        "tau_start_ftsamp",
        "tau_end_ftsamp",
        "nsweep_points",

        # Timing delays
        "mw_to_laser_delay_treg", # Laser turn-on lag relative to MW
        "relax_delay_treg",       # Spin relaxation delay (post laser reinitialization)

        # Readout timing
        "laser_on_treg",
        "readout_reference_start_treg",
        "readout_integration_treg",
        "laser_readout_offset_treg",

        # Experiment control
        "reps",
        "pre_init",
        "get_reference",          # If True, collect MW-off reference readouts
    ]


    def initialize(self):
        """
        Prepare waveforms, FPGA registers, and sweep parameters.

        Steps:
          1. Generate shaped pi and pi/2 waveform envelopes for each possible sub-clock offset
          2. Allocate FPGA registers for runtime delay computations.
          3. Configure tau sweep, and the pulse phases.
        """
        # ---------------------
        # General Config
        # ---------------------
        self.check_cfg()

        if self.cfg.mw_gain < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.mw_gain > 32767: # Max DAC value = 2**15 - 1
            assert 0, 'Largest Microwave gain exceeds maximum value'

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)

        self.setup_readout()

        # Samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up
        # ---------------------
        # Waveform buffer length must be >= 3 treg
        self.pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.mw_pi2_ftsamp*2 + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.mw_pi2_ftsamp + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)

        # Hermite envelopes (peak-normalized; scaled to DAC max below)
        pi_env = hermite_envelope(self.cfg.mw_pi2_ftsamp * 2)
        half_pi_env = hermite_envelope(self.cfg.mw_pi2_ftsamp)
        maxv = self.soccfg.get_maxv(self.cfg.mw_channel)

        # Generate waveforms with each offset & pulse type
        # Waveforms are added in order: [pi_0, half_pi_0, ..., pi_15, half_pi_15]
        for i in range(self.samps_per_clk):
            # pi pulses
            data = np.zeros(self.pi_waveform_len_treg * 16)
            data[i: i + self.cfg.mw_pi2_ftsamp*2] = pi_env
            data *= maxv
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=data, qdata=data)

            # pi/2 pulses
            data = np.zeros(self.half_pi_waveform_len_treg * 16)
            data[i : i + self.cfg.mw_pi2_ftsamp] = half_pi_env
            data *= maxv
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=data, qdata=data)

        # Amount of deadtime in the waveforms
        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.cfg.mw_pi2_ftsamp*2
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.mw_pi2_ftsamp

        # ---------------------
        # FPGA Register Setup
        # ---------------------
        # delay_coarse_cycles: coarse-time inter-pulse delay (in FPGA clock cycles, treg).
        self.delay_coarse_cycles = self.new_gen_reg(self.cfg.mw_channel,
                                        name='treg_offset',
                                        init_val=0)

        # delay_fine_samples: fine-time waveform offset (in DAC samples, ftsamp).
        # Initialized to account for difference between pi and pi/2 waveform deadtimes
        self.delay_fine_samples = self.new_gen_reg(self.cfg.mw_channel,
                                        name='ftsamp_offset',
                                        init_val=(self.pi_len_unused-self.half_pi_len_unused))

        # n_cpmg_register: Tracks the pi-pulses (counts down to 0).
        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                        name='ncpmg',
                                        init_val=self.cfg.n_cpmg - 1)

        # Default pulse parameters -- phase is overridden per-pulse by set_waveform()
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     phase=self.deg2reg(90))

        #  tau sweep register (in ftsamp units)
        self.tau = self.new_gen_reg(self.cfg.mw_channel, "tau", init_val=self.cfg.tau_start_ftsamp)
        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self,
                self.tau,
                self.cfg.tau_start_ftsamp,
                self.cfg.tau_end_ftsamp,
                expts=self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor)
                )
        elif self.cfg.scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self,
                self.tau,
                self.cfg.tau_start_ftsamp,
                self.cfg.tau_end_ftsamp,
                self.cfg.nsweep_points)
                )
        else:
            assert 0, 'cfg.scaling_mode must be "linear" or "exponential"'

        # Direct handles to the waveform address and NCO phase registers
        self.address_reg = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        self.phase_reg = self.get_gen_reg(self.cfg.mw_channel, name='phase')

        # Encode the XY8 phase sequence as a bitmask in phase_seq_register (avoids branching)
        self.phase_to_list()

        # Select final pi/2 phase to maintain correct phase projection
        #   N % 8 in {0,1,4,7}  ->  final pi/2 along -Y (270 deg)
        #   N % 8 in {2,3,5,6}  ->  final pi/2 along +Y (90 deg)
        self.ending_half_pi_phase = 270 if self.cfg.n_cpmg % 8 in [0,1,4,7] else 90

        self.pre_init()


    def phase_to_list(self):
        """
        Encode the XY8 pi-pulse phase sequence as a compact integer bitmask.

        XY8 sequence: X Y X Y Y X Y X
        Encoding: X -> 0, Y -> 1, MSB = first pulse.

        At runtime, the phase for pulse k is extracted as:
            bit = (phase_seq_register >> (k % 8)) & 1
            phase = 0 deg if bit == 0 (X), 90 deg if bit == 1 (Y)

        This avoids branching in the inner loop -- the phase is computed
        directly from a register with 3 binary operations.
        """
        self.phase_list = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]
        # Rotate by N_CPMG % 8 to align MSB-first encoding with the countdown register
        rotation = self.cfg.n_cpmg % len(self.phase_list)
        self.phase_list[:] = self.phase_list[rotation:] + self.phase_list[:rotation]
        phase_seq = int(''.join(['0' if val == 'X' else '1' for val in self.phase_list]), 2)
        self.phase_seq_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   "phase_seq_register",
                                                   init_val=phase_seq)


    def body(self):
        """
        One repetition: Initialize spin -> MW-on CPMG -> Readout (also re-initializes).

        When get_reference=True, the signal_and_reference_readout runs
        an equivalent MW-off idle delay for |0> normalization.
        """
        self.initialize_spin()
        self.program_pulses(1)
        self.readout_and_reference(lambda: self.program_pulses(2))

    def program_pulses(self, label_id):
        """
        Program the full CPMG MW pulse train.

        Structure:
            1. pi/2_Y pulse (initial spin rotation)
            2. Loop: N_CPMG x [delay tau -> pi_phi -> delay tau]  (XY8 phase cycling)
            3. pi/2_+/-Y pulse (final readout rotation, phase chosen by parity)

        Args:
            label_id:
                  Used to generate unique loop labels for the assembler.
        """
        # ---------------------
        # Initial pi/2 rotation
        # ---------------------
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0")
        self.phase_reg.set_to(90, physical_unit=True)
        self.sync_all()

        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Skip programming loop if n_cpmg = 0 --
        # Delays are accounted for in set_waveform and initial value of delay_fine_samples
        if self.cfg.n_cpmg > 0:
            # ---------------------
            # CPMG loop init.
            # ---------------------
            self.n_cpmg_register.reset()

            # Subtract tau from delay_fine_samples so the first inter-pulse delay is tau (not 2tau).
            self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                        self.delay_fine_samples.addr, "-", self.tau.addr)

            # Select pi waveform template (used only to set modecode -> pulse length)
            # Phase and fine timing are handled dynamically in set_waveform()
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0")

            # ---------------------
            # CPMG pi-pulse loop
            # ---------------------
            self.label(f"LOOP_ncpmg_{label_id}") # Begin loop

            self.set_waveform(pi_pulse=True) # Compute delay, select offset & phase
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all()

            self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                f"LOOP_ncpmg_{label_id}"
            )

        # ---------------------
        # Final pi/2 projection
        # ---------------------
        # Select pi/2 waveform template (used only to set modecode -> pulse length)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0")
        self.set_waveform(pi_pulse=False, phase=self.ending_half_pi_phase)

        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Reset timing offset for next iteration
        self.delay_fine_samples.reset()


    def set_waveform(self, pi_pulse=True, phase=None):
        """
        Configures timing delays, phase and waveform address for the next pulse.

        Coarse delay -> sync(delay_coarse_cycles)
        Fine delay   -> selects the waveform variant.

        Args:
            pi_pulse: True for pi, False for pi/2.
            phase:    Optional override phase in degrees.
        """
        self.offset_computations(double_tau=pi_pulse)
        self.sync(self.delay_coarse_cycles.page, self.delay_coarse_cycles.addr)

        # Resolve waveform address based on fine offset
        self.address_reg.set_to(self.delay_fine_samples, '*',
            self.pi_waveform_len_treg + self.half_pi_waveform_len_treg,
            physical_unit=False
        )
        if not pi_pulse: # pi/2 Waveforms are stored after pi waveforms
            self.address_reg.set_to(self.address_reg, '+',
                                     self.pi_waveform_len_treg, physical_unit=False)

        self.select_phase(phase)


    def select_phase(self, phase=None):
        """
        Set the NCO phase for the next pulse.

        If `phase` is provided, it is used directly.
        Otherwise, the phase is derived from the XY8 sequence
        using the current CPMG pulse index.

        Args:
            phase: Optional override phase in degrees.
        """
        if phase is not None:
            self.phase_reg.set_to(phase, physical_unit=True)
        else:
            # Step 1: compute pulse index (n_cpmg % 8)
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.n_cpmg_register.addr, "&", int(len(self.phase_list)-1))

            # Step 2: extract corresponding XY8 bit
            # phase_reg = phase_seq_register >> index
            self.bitw(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_seq_register.addr, '>>', self.phase_reg.addr)

            # Step 3: isolate LSB (0 -> X, 1 -> Y)
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_reg.addr, "&", 1)

            # Step 4: Map bit encoding -> NCO phase range
            # (left shift used instead of multiply due to hardware constraint)
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_reg.addr, "<<", 30)


    def offset_computations(self, double_tau=False):
        """
        Computes the delay to the next pulse.

        Timing is referenced to the end of the previous waveform.
        The effective delay is:
            effective_delay = (tau or 2tau) - elapsed_waveform_deadtime

        This delay is decomposed into:
            - delay_coarse_cycles  (integer FPGA cycles, treg)
            - delay_fine_samples   (waveform offset in ftsamp)

        Args:
            double_tau: If True, computes a 2tau delay (after a pi pulse).
        """
        # Step 1: Compute effective delay (fine-sample units)
        self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                  self.delay_fine_samples.addr, "+", self.tau.addr)
        if double_tau:
            self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                      self.delay_fine_samples.addr, "+", self.tau.addr)
        self.mathi(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                    self.delay_fine_samples.addr, "-", self.pi_len_unused)

        # Step 2: extract coarse delay (integer FPGA cycles)
        # Equivalent to: // samps_per_clk
        self.bitwi(self.delay_fine_samples.page, self.delay_coarse_cycles.addr,
                   self.delay_fine_samples.addr, ">>", int(np.log2(self.samps_per_clk)))

        # Step 3: retain remainder (fine waveform offset)
        # Equivalent to: % samps_per_clk
        self.bitwi(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                   self.delay_fine_samples.addr, "&", (self.samps_per_clk-1))


    def acquire(self, raw_data=False, *arg, **kwarg):
        """
        Run the experiment and return processed data.

        Delegates to ReadoutHelpers.acquire(), which in turn calls
        NVAveragerProgram.acquire() via super(), tagging the sweep
        axis as 'tau_ftsamp' for downstream analysis.
        """
        data = super().acquire(raw_data=raw_data, sweep_param='tau_ftsamp', *arg, **kwarg)
        return data
