'''
CPMG XY8 sub-nanosecond resolution pulsing program with arbitrary N
===================================================================
Min resolution of 200 ps for delay steps between pulses in a CPMG XY8
phase sequence. Unlike round-based XY8 implementations, this version
treats n_cpmg as the total number of pi pulses, so N can be any
positive integer. This also works for spin echo with N = 1.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.arqick.standard_ops import StandardOps


class CPMGXY8AnyNFineRes(StandardOps, NVAveragerProgram):
    '''
    CPMG XY8 sub-nanosecond resolution pulsing program where n_cpmg is the
    total number of pi pulses (any positive integer).

    Phase pattern repeats XYXYYXYX continuously, so:
    - n_cpmg=3  -> XYX
    - n_cpmg=10 -> XYXYYXYXXY
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "freq_freg",  # microwave freq
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW gain
        "scaling_mode",  # 'linear' or 'exponential' spacing of delay points in sweep
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start
    
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "n_cpmg",  # total number of pi pulses, can be any positive integer
    ]

    def initialize(self):
        self.init()

        # Precompute the final readout pi/2 phase from N using XY8 periodicity.
        # Mapping: N mod 8 in {0,1,4,7} -> -90, {2,3,5,6} -> +90.
        if self.cfg.n_cpmg % 8 in [0, 1, 4, 7]:
            self.last_pi2_phase_deg = -90
        else:
            self.last_pi2_phase_deg = 90

        self.delay_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='delay',
            init_val=self.cfg.delay_tdds_start,
        )

        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(
                NVQickSweep(
                    self,
                    self.delay_register,
                    self.cfg.delay_tdds_start,
                    self.cfg.delay_tdds_end,
                    self.cfg.nsweep_points,
                    scaling_mode=self.cfg.scaling_mode,
                    scaling_factor=self.cfg.scaling_factor,
                )
            )
        else:
            self.add_sweep(
                NVQickSweep(
                    self,
                    self.delay_register,
                    self.cfg.delay_tdds_start,
                    self.cfg.delay_tdds_end,
                    self.cfg.nsweep_points,
                )
            )

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.pmod_trigger_sequence()

        self.tdds_offset_register.reset()
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.delay_register, n_cpmg_pulses=self.cfg.n_cpmg, vary_n=False)

        # Final tau and readout pi/2 pulse.
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(self.last_pi2_phase_deg))
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.delay_register)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)

