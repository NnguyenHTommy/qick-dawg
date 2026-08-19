'''
ODMR Fine Resolution -- Hermite pulse
=======================================================================
Identical protocol to the working square-pulse PODMRFineRes (frequency
sweep, single fixed-duration MW pulse, shared readout helpers); the only
change is the pulse envelope: a 2nd-order Hermite-Gaussian,
(1 - x^2) * exp(-x^2), stretched self-similarly to the pulse duration.

mw_pi_ftsamp still sets the TOTAL pulse duration in fine-timing samples.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.finetimingsuite.readout_helpers import ReadoutHelpers
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


class PODMRFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    Pulsed-ODMR with a Hermite fine-resolution MW pulse.
    Sweeps microwave frequency at fixed pulse duration and gain.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod", # should be 0 for PMOD0_0

        # MW pulse parameters
        "mw_pi_ftsamp",
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain

        # Sweep parameters
        "mw_start_fMHz",
        "mw_end_fMHz",
        "nsweep_points",

        # Readout and delays
        "mw_to_laser_delay_treg", # How long do we need to delay the mw, to ensure the laser pulse is correctly timed before it
        "relax_delay_treg", # delay between laser and next MW pulse

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

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)

        # Setup laser
        self.setup_readout()

        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Configure Hermite pi pulse waveform
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pi_ftsamp / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        data[:self.cfg.mw_pi_ftsamp] = hermite_envelope(self.cfg.mw_pi_ftsamp)
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

        # Configure pulse registers
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_start_freg,
                                     gain=self.cfg.mw_gain,
                                     waveform="pulse",
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        # Setup frequency sweep
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")
        self.add_sweep(NVQickSweep(self,
                                self.mw_frequency_register,
                                self.cfg.mw_start_fMHz,
                                self.cfg.mw_end_fMHz,
                                self.cfg.nsweep_points))

        self.pre_init()

    def body(self):
        self.initialize_spin()
        self.program_pulse()
        self.readout_and_reference(self.program_pulse)

    def program_pulse(self):
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

    def acquire(self, raw_data=False, *arg, **kwarg):
        """
        Delegates to ReadoutHelpers.acquire() -> NVAveragerProgram.acquire(),
        tagging the sweep axis as 'mw_fMHz'.
        """
        data = super().acquire(raw_data=raw_data, sweep_param='mw_fMHz', *arg, **kwarg)
        return data
