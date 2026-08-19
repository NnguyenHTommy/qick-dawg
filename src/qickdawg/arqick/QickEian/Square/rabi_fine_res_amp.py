'''
Rabi sub-nanosecond resolution pulsing program
=======================================================================
Performs Rabi sweep with a minimum mw pulse step size of 1 ftsamp (200ps)
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
# from qickdawg.nvtestsuite.test_1.readout_helpers import ReadoutHelpers
import numpy as np

class RabiFineRes(NVAveragerProgram): # ReadoutHelpers
    '''
    Rabi sub-nanosecond resolution pulsing program
    '''
    required_cfg = [      
        # Channels and pmods
        "mw_channel",
        # "adc_channel",
        # "laser_gate_pmod",        # 0 for PMOD0_0

        # MW pulse parameters
        "mw_nqz",                 # 1 for f < 2.495 GHz
        "mw_freg",
        "mw_gain",

        # Sweep parameters
        "mw_duration_start_ftsamp",
        "mw_duration_end_ftsamp",
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

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
        # self.setup_helper_registers(self.cfg.mw_channel)

        # # Setup laser
        # self.setup_readout()

        # Samples per clock (16 with current version of QICK-DAWG)
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # ---------------------
        # Waveform Set-up
        # ---------------------
        self.mw_pulse_waveform_len_treg = 4
        self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk  # in ftsamp units
        
        for i in np.arange(0, self.mw_pulse_waveform_len_ftsamp+1, 1):
            data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
            data[:i] = 1
            data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pulse_{i}", idata=data, qdata=data)

        # ---------------------
        # FPGA Register Setup
        # ---------------------
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')
        
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.mw_freg,
                                     gain=self.cfg.mw_gain,
                                     phase = 0)
        
        # mw duration register
        self.mw_duration_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_duration',
                                                   init_val=self.cfg.mw_duration_start_ftsamp)

        # mw coarse and fine pulse loop registers
        self.coarse_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_coarse',
                                                   init_val=0)
        self.fine_mw_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_fine',
                                                   init_val=0)

        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_duration_register,
                                   start=self.cfg.mw_duration_start_ftsamp,
                                   stop=self.cfg.mw_duration_end_ftsamp,
                                   expts=self.cfg.nsweep_points))
        
        self.pre_init() # Give tproc time to get ahead

    def body(self):
        self.initialize_spin()
        self.program_pulses(1)
        # self.readout_and_reference(lambda: self.program_pulses(2))

    def program_pulses(self, num):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)
        
        # set coarse and fine registers based on duration. coarse is x//64 and then multiply by 4 to get treg units
        self.bitwi(self.coarse_mw_register.page, self.coarse_mw_register.addr, self.mw_duration_register.addr, ">>", int(np.log2(self.mw_pulse_waveform_len_ftsamp)))
        self.bitwi(self.coarse_mw_register.page, self.coarse_mw_register.addr, self.coarse_mw_register.addr, "<<", int(np.log2(self.mw_pulse_waveform_len_treg)))
        self.bitwi(self.fine_mw_register.page, self.fine_mw_register.addr, self.mw_duration_register.addr, "&", self.mw_pulse_waveform_len_ftsamp - 1)
        
        # if there is no coarse part just do fine part
        self.condj(self.coarse_mw_register.page, self.coarse_mw_register.addr, "==", 0, f"JUMP_NO_COARSE_{num}")

        # since using sync all (and need to use it for accurate timing), it will always play a pulse so need to subtract onewaveform length
        self.coarse_mw_register.set_to(self.coarse_mw_register, "-", self.mw_pulse_waveform_len_treg, physical_unit=False)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pulse_{self.mw_pulse_waveform_len_ftsamp}", mode = "periodic")
        self.pulse(ch=self.cfg.mw_channel) 
        self.sync_all()
        self.sync(self.coarse_mw_register.page, self.coarse_mw_register.addr)

        self.label(f"JUMP_NO_COARSE_{num}")
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"pulse_{0}", mode = "oneshot")
        self.address_register.set_to(self.fine_mw_register, '*', self.mw_pulse_waveform_len_treg, physical_unit = False)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

    # def acquire(self, raw_data=False, *arg, **kwarg):
    #     """
    #     Delegates to ReadoutHelpers.acquire() → NVAveragerProgram.acquire(), 
    #     tagging the sweep axis as 'mw_duration_ftsamp'.
    #     """
    #     data = super().acquire(raw_data=raw_data, sweep_param='mw_duration_ftsamp', *arg, **kwarg)
    #     return data


# class RabiGainFineRes(ReadoutHelpers, NVAveragerProgram):
#     '''
#     Amplitude (gain) Rabi sub-nanosecond resolution pulsing program

#     Sweeps MW pulse gain (amplitude) at a fixed pulse duration by writing
#     directly to the live "gain" pulse-parameter register between pulses --
#     the same mechanism PODMRFineRes uses to sweep frequency, via NVQickSweep
#     incrementing a register obtained through get_gen_reg(). This needs only
#     ONE waveform envelope regardless of sweep resolution, so it scales to
#     hundreds or thousands of points.

#     This version allocates NOTHING beyond what setup_helper_registers()
#     already provides (self.mw_gain_register) plus the sweep on it -- no
#     extra custom registers, and no override of readout_and_reference(). An
#     earlier version added a spare register to correctly restore the swept
#     gain value across the MW-off reference readout (rather than restoring
#     to a static cfg.mw_gain, which would desync the sweep). But scoping the
#     output showed switching-edge transients with NO RF carrier in between --
#     envelope steps with no oscillation -- suggesting the extra register
#     allocation may have been colliding with (and corrupting) the frequency
#     register's actual hardware slot. Removing it to test that directly.

#     TRADEOFF while this is in place: the reference (MW-off) readout uses the
#     base ReadoutHelpers.readout_and_reference(), which restores gain to the
#     static cfg.mw_gain after zeroing it for the reference pulse -- not back
#     to the swept value. That means every point after the first reference
#     readout in a rep has its gain register reset to cfg.mw_gain instead of
#     continuing the sweep correctly. This is a known, deliberate step backward
#     to isolate the missing-carrier issue; once that's confirmed fixed, the
#     save/restore logic needs to be reintroduced carefully (or reference
#     handling reworked another way) to get correct sweep data with
#     get_reference=True.
#     '''
#     required_cfg = [
#         # Channels and pmods
#         "mw_channel",
#         "adc_channel",
#         "laser_gate_pmod",        # 0 for PMOD0_0

#         # MW pulse parameters
#         "mw_pulse_ftsamp",        # Fixed MW pulse duration (fine-resolution samples)
#         "mw_nqz",                 # 1 for f < 2.495 GHz
#         "mw_freg",
#         "mw_gain",                 # used by the base readout_and_reference() to restore after the reference pulse

#         # Sweep parameters
#         "mw_gain_start",
#         "mw_gain_end",
#         "nsweep_points",

#         # Readout and delays
#         "mw_to_laser_delay_treg", # Laser turn-on lag relative to MW
#         "relax_delay_treg",       # Spin relaxation delay (post laser reinitialization)

#         # Readout
#         "laser_on_treg",
#         "readout_reference_start_treg",
#         "readout_integration_treg",
#         "laser_readout_offset_treg",

#         # Other
#         "reps",
#         "pre_init",
#         "get_reference",  # Whether to acquire a reference readout with MW gain = 0
#     ]

#     def initialize(self):
#         self.check_cfg()

#         if self.cfg.mw_gain_start < 0 or self.cfg.mw_gain_end < 0:
#             assert 0, 'Smallest Microwave gain must be positive'
#         elif self.cfg.mw_gain_start > 32767 or self.cfg.mw_gain_end > 32767: # 2**15 - 1
#             assert 0, 'Largest Microwave gain exceeds maximum value'
#         assert self.cfg.nsweep_points >= 2, 'nsweep_points must be >= 2 (NVQickSweep divides by nsweep_points-1)'

#         # Get mw registers
#         self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)
#         self.setup_helper_registers(self.cfg.mw_channel)  # creates self.mw_gain_register

#         # Setup laser
#         self.setup_readout()

#         # Samples per clock (16 with current version of QICK-DAWG)
#         self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

#         # ---------------------
#         # Waveform Set-up (fixed duration, fine-resolution) -- ONE envelope
#         # regardless of nsweep_points, since amplitude comes from the live
#         # gain register rather than being baked into per-point envelope data.
#         # ---------------------
#         self.mw_pulse_waveform_len_treg = max(int(np.ceil(self.cfg.mw_pulse_ftsamp / self.samps_per_clk)), 3)
#         self.mw_pulse_waveform_len_ftsamp = self.mw_pulse_waveform_len_treg * self.samps_per_clk

#         data = np.zeros(self.mw_pulse_waveform_len_ftsamp)
#         data[:self.cfg.mw_pulse_ftsamp] = 1
#         data *= self.soccfg.get_maxv(self.cfg.mw_channel)
#         self.add_envelope(ch=self.cfg.mw_channel, name="pulse", idata=data, qdata=data)

#         # ---------------------
#         # FPGA Register Setup
#         # ---------------------
#         self.default_pulse_registers(ch=self.cfg.mw_channel,
#                                      style='arb',
#                                      freq=self.cfg.mw_freg,
#                                      gain=self.cfg.mw_gain_start,
#                                      waveform="pulse",
#                                      phase=0)
#         self.set_pulse_registers(ch=self.cfg.mw_channel)

#         # Amplitude (gain) sweep, reusing the gain register set up in
#         # setup_helper_registers -- same mechanism PODMRFineRes uses for its
#         # frequency sweep, just on the "gain" field instead of "freq".
#         # No other custom registers are allocated in this class.
#         self.add_sweep(NVQickSweep(self,
#                                    reg=self.mw_gain_register,
#                                    start=self.cfg.mw_gain_start,
#                                    stop=self.cfg.mw_gain_end,
#                                    expts=self.cfg.nsweep_points))

#         self.pre_init() # Give tproc time to get ahead

#     def body(self):
#         self.initialize_spin()
#         self.program_pulse()
#         # Uses the INHERITED ReadoutHelpers.readout_and_reference() -- no
#         # override -- exactly like PODMRFineRes/T1FineRes/RabiFineRes.
#         self.readout_and_reference(self.program_pulse)

#     def program_pulse(self):
#         # Use the live gain register's value for the pulse gain
#         self.set_pulse_registers(ch=self.cfg.mw_channel, gain=self.mw_gain_register)
#         self.pulse(ch=self.cfg.mw_channel)
#         self.sync_all()

#     def acquire(self, raw_data=False, *arg, **kwarg):
#         """
#         Delegates to ReadoutHelpers.acquire() -> NVAveragerProgram.acquire(),
#         tagging the sweep axis as 'mw_gain'.
#         """
#         data = super().acquire(raw_data=raw_data, sweep_param='mw_gain', *arg, **kwarg)
#         return data
