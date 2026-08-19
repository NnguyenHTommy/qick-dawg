'''
CPMG-XY Fine-Resolution Pulse Sequence -- Hermite pulse -- ARQICK variant
=======================================================================
Same XY8 phase-cycled CPMG machinery as the QICK-only Hermite
CPMGXYFineRes (n_cpmg = total pi-pulse count, any positive integer; tau
swept fully on-chip via waveform-offset addressing -- shaped pulses are
fine here since only the OFFSET changes per point, not the shape). The
only change is the readout mechanism: laser gating and photon counting
are no longer done on this board -- this program triggers ARTIQ's
sequence once per rep and plays the CPMG train; ARTIQ handles
laser/readout externally on that trigger.

n_cpmg = 0 -> Ramsey, n_cpmg = 1 -> Hahn echo, same as before.

TODO: n_cpmg for the actual run may land around 32 depending on
tomorrow's conversations -- update cfg.n_cpmg accordingly when known.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep

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


class CPMGXYFineRes(NVAveragerProgram):
    '''
    CPMG-XY8 sub-nanosecond resolution pulsing program (ARTIQ-triggered readout)
    '''
    required_cfg = [
        "mw_channel",
        "mw_pi2_tdds",           # pi/2 pulse duration
        "mw_nqz",                 # 1 for f < 2.495 GHz
        "freq_freg",
        "mw_gain",
        "n_cpmg",

        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "scaling_mode",
        "scaling_factor",         # only used if scaling_mode is 'exponential'

        "reps",
        "pmod_out_pin",                        # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",            # 50ns is reasonable
        "pmod_out_trig_delay_treg",              # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns
        "pulse_seq_delay_treg",                  # delay between sequence end and next trigger
    ]

    def initialize(self):
        """
        Prepare Hermite pi/pi2 waveforms (one per sub-clock offset), FPGA
        registers, and the tau sweep.
        """
        self.check_cfg()

        if self.cfg.mw_gain < 0:
            assert 0, 'Smallest Microwave gain must be postive'
        elif self.cfg.mw_gain > 32767:
            assert 0, 'Largest Microwave gain exceeds maximum value'

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up
        # ---------------------
        self.pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.mw_pi2_tdds*2 + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = \
            max(int(np.ceil((self.cfg.mw_pi2_tdds + (self.samps_per_clk-1)) / self.samps_per_clk)), 3)

        pi_env = hermite_envelope(self.cfg.mw_pi2_tdds * 2)
        half_pi_env = hermite_envelope(self.cfg.mw_pi2_tdds)
        maxv = self.soccfg.get_maxv(self.cfg.mw_channel)

        for i in range(self.samps_per_clk):
            data = np.zeros(self.pi_waveform_len_treg * 16)
            data[i: i + self.cfg.mw_pi2_tdds*2] = pi_env
            data *= maxv
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=data, qdata=data)

            data = np.zeros(self.half_pi_waveform_len_treg * 16)
            data[i : i + self.cfg.mw_pi2_tdds] = half_pi_env
            data *= maxv
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=data, qdata=data)

        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.cfg.mw_pi2_tdds*2
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.mw_pi2_tdds

        # ---------------------
        # FPGA Register Setup
        # ---------------------
        self.delay_coarse_cycles = self.new_gen_reg(self.cfg.mw_channel,
                                        name='treg_offset',
                                        init_val=0)

        self.delay_fine_samples = self.new_gen_reg(self.cfg.mw_channel,
                                        name='tdds_offset',
                                        init_val=(self.pi_len_unused-self.half_pi_len_unused))

        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                        name='ncpmg',
                                        init_val=self.cfg.n_cpmg - 1)

        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain,
                                     phase=self.deg2reg(90))

        self.tau = self.new_gen_reg(self.cfg.mw_channel, "tau", init_val=self.cfg.delay_tdds_start)
        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self, self.tau, self.cfg.delay_tdds_start, self.cfg.delay_tdds_end,
                expts=self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor))
        elif self.cfg.scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self, self.tau, self.cfg.delay_tdds_start, self.cfg.delay_tdds_end,
                self.cfg.nsweep_points))
        else:
            assert 0, 'cfg.scaling_mode must be "linear" or "exponential"'

        self.address_reg = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        self.phase_reg = self.get_gen_reg(self.cfg.mw_channel, name='phase')

        self.phase_to_list()

        self.ending_half_pi_phase = 270 if self.cfg.n_cpmg % 8 in [0,1,4,7] else 90

        self.synci(200)  # give processor some time to configure pulses

    def phase_to_list(self):
        """
        XY8 sequence X Y X Y Y X Y X encoded as a bitmask (X=0, Y=1, MSB first),
        rotated so index k = n_cpmg_register % 8 reads out the right phase.
        """
        self.phase_list = ["X", "Y", "X", "Y", "Y", "X", "Y", "X"]
        rotation = self.cfg.n_cpmg % len(self.phase_list)
        self.phase_list[:] = self.phase_list[rotation:] + self.phase_list[:rotation]
        phase_seq = int(''.join(['0' if val == 'X' else '1' for val in self.phase_list]), 2)
        self.phase_seq_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   "phase_seq_register",
                                                   init_val=phase_seq)

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.program_pulses(1)

        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def program_pulses(self, label_id):
        """
        pi/2_Y -> N_CPMG x [delay tau -> pi_phi -> delay tau] (XY8) -> pi/2_+/-Y
        """
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0")
        self.phase_reg.set_to(90, physical_unit=True)
        self.sync_all()

        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        if self.cfg.n_cpmg > 0:
            self.n_cpmg_register.reset()

            self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                        self.delay_fine_samples.addr, "-", self.tau.addr)

            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0")

            self.label(f"LOOP_ncpmg_{label_id}")

            self.set_waveform(pi_pulse=True)
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all()

            self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                f"LOOP_ncpmg_{label_id}"
            )

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0")
        self.set_waveform(pi_pulse=False, phase=self.ending_half_pi_phase)

        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.delay_fine_samples.reset()

    def set_waveform(self, pi_pulse=True, phase=None):
        self.offset_computations(double_tau=pi_pulse)
        self.sync(self.delay_coarse_cycles.page, self.delay_coarse_cycles.addr)

        self.address_reg.set_to(self.delay_fine_samples, '*',
            self.pi_waveform_len_treg + self.half_pi_waveform_len_treg,
            physical_unit=False
        )
        if not pi_pulse:
            self.address_reg.set_to(self.address_reg, '+',
                                     self.pi_waveform_len_treg, physical_unit=False)

        self.select_phase(phase)

    def select_phase(self, phase=None):
        if phase is not None:
            self.phase_reg.set_to(phase, physical_unit=True)
        else:
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.n_cpmg_register.addr, "&", int(len(self.phase_list)-1))
            self.bitw(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_seq_register.addr, '>>', self.phase_reg.addr)
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_reg.addr, "&", 1)
            self.bitwi(self.phase_reg.page, self.phase_reg.addr,
                    self.phase_reg.addr, "<<", 30)

    def offset_computations(self, double_tau=False):
        self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                  self.delay_fine_samples.addr, "+", self.tau.addr)
        if double_tau:
            self.math(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                      self.delay_fine_samples.addr, "+", self.tau.addr)
        self.mathi(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                    self.delay_fine_samples.addr, "-", self.pi_len_unused)

        self.bitwi(self.delay_fine_samples.page, self.delay_coarse_cycles.addr,
                   self.delay_fine_samples.addr, ">>", int(np.log2(self.samps_per_clk)))

        self.bitwi(self.delay_fine_samples.page, self.delay_fine_samples.addr,
                   self.delay_fine_samples.addr, "&", (self.samps_per_clk-1))
