'''
Amplitude (gain) Rabi -- Gaussian pulse
=======================================================================
Byte-for-byte the same protocol as the verified working square-pulse
RabiGainFineRes (live gain-register sweep at fixed pulse duration, one
envelope regardless of sweep resolution, scales to >= 1000 points); the
only change is the pulse envelope: a Gaussian,
exp(-x^2 / 2), stretched self-similarly to the pulse duration.

NOTE: built from the CORRECTED square version -- program_pulse() is a
bare pulse()/sync_all() with NO set_pulse_registers(gain=...) call
(setting gain in both default_pulse_registers and set_pulse_registers
raises a RuntimeError; the swept gain register updates the live value
without re-arming).
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.finetimingsuite.readout_helpers import ReadoutHelpers
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


class RabiGainFineRes(ReadoutHelpers, NVAveragerProgram):
    '''
    Amplitude (gain) Rabi with a Gaussian fine-resolution MW pulse.

    Sweeps MW pulse gain (amplitude) at a fixed pulse duration by writing
    directly to the live "gain" pulse-parameter register between pulses --
    the same mechanism PODMRFineRes uses to sweep frequency, via NVQickSweep
    incrementing a register obtained through get_gen_reg(). This needs only
    ONE waveform envelope regardless of sweep resolution, so it scales to
    hundreds or thousands of points.

    TRADEOFF (inherited from the verified square version): the reference
    (MW-off) readout uses the base ReadoutHelpers.readout_and_reference(),
    which restores gain to the static cfg.mw_gain after zeroing it for the
    reference pulse -- not back to the swept value. Keep get_reference=False
    for correct sweep data until that restore logic is reworked.
    '''
    required_cfg = [
        # Channels and pmods
        "mw_channel",
        "adc_channel",
        "laser_gate_pmod",        # 0 for PMOD0_0

        # MW pulse parameters
        "mw_pulse_ftsamp",        # Fixed MW pulse duration (fine-resolution samples)
        "mw_nqz",                 # 1 for f < 2.495 GHz
        "mw_freg",
        "mw_gain",                 # used by the base readout_and_reference() to restore after the reference pulse

        # Sweep parameters
        "mw_gain_start",
        "mw_gain_end",
        "nsweep_points",

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

        if self.cfg.mw_gain_start < 0 or self.cfg.mw_gain_end < 0:
            assert 0, 'Smallest Microwave gain must be positive'
        elif self.cfg.mw_gain_start > 32767 or self.cfg.mw_gain_end > 32767: # 2**15 - 1
            assert 0, 'Largest Microwave gain exceeds maximum value'
        assert self.cfg.nsweep_points >= 2, 'nsweep_points must be >= 2 (NVQickSweep divides by nsweep_points-1)'

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        self.setup_helper_registers(self.cfg.mw_channel)  # creates self.mw_gain_register

        # Setup laser
        self.setup_readout()

        # Samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up (fixed duration, fine-resolution) -- ONE Gaussian
        # envelope regardless of nsweep_points, since amplitude comes from
        # the live gain register rather than per-point envelope data.
        # ---------------------
        self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_ftsamp / self.samps_per_clk)), 3)
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk

        data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
        data[:self.cfg.mw_pulse_ftsamp] = gaussian_envelope(self.cfg.mw_pulse_ftsamp)
        data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

        # ---------------------
        # FPGA Register Setup
        # ---------------------
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain_start,
                                     waveform="pulse",
                                     phase=0)
        self.set_pulse_registers(ch=self.cfg.mw_channel)

        # Amplitude (gain) sweep, reusing the gain register set up in
        # setup_helper_registers -- same mechanism PODMRFineRes uses for its
        # frequency sweep, just on the "gain" field instead of "freq".
        # No other custom registers are allocated in this class.
        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_gain_register,
                                   start=self.cfg.mw_gain_start,
                                   stop=self.cfg.mw_gain_end,
                                   expts=self.cfg.nsweep_points))

        self.pre_init() # Give tproc time to get ahead

    def body(self):
        self.initialize_spin()
        self.program_pulse()
        # Uses the INHERITED ReadoutHelpers.readout_and_reference() -- no
        # override -- exactly like PODMRFineRes/T1FineRes/RabiFineRes.
        self.readout_and_reference(self.program_pulse)

    def program_pulse(self):
        # Bare pulse: the swept gain register IS the live pulse parameter;
        # do NOT re-arm with set_pulse_registers(gain=...) here (RuntimeError:
        # gain would be set in both default and set_pulse_registers).
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

    def acquire(self, raw_data=False, *arg, **kwarg):
        """
        Delegates to ReadoutHelpers.acquire() -> NVAveragerProgram.acquire(),
        tagging the sweep axis as 'mw_gain'.
        """
        data = super().acquire(raw_data=raw_data, sweep_param='mw_gain', *arg, **kwarg)
        return data
