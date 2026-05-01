'''
CPMG XY8 sub-nanosecond resolution pulsing program with to sweep N pulses
===================================================================
Min resolution of 200 ps for delay steps between pulses in a CPMG XY8
phase sequence. Unlike round-based XY8 implementations, this version
treats n_cpmg as the total number of pi pulses, so N can be any
positive integer.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np
from qickdawg.arqick.standard_ops import StandardOps

class CPMGXY8SweepNFineRes(StandardOps, NVAveragerProgram):
    '''
    CPMG XY8 sub-nanosecond resolution pulsing program where n_cpmg is the
    total number of pi pulses (any positive integer).

    Phase pattern repeats XYXYYXYX continuously, so:
    - n_cpmg=3  -> XYX
    - n_cpmg=10 -> XYXYYXYXXY
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "n_cpmg_start",
        "n_cpmg_end",
        "nsweep_points",
        "freq_freg",  # microwave freq
        "delay_tdds",  # fixed interpulse delay
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW gain
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start
    ]

    def initialize(self):
        self.init()
        if self.cfg.n_cpmg_start < 1:
            raise ValueError("n_cpmg_start must be >= 1 for CPMG XY8 variable-N sequence.")
        if self.cfg.n_cpmg_end < 1:
            raise ValueError("n_cpmg_end must be >= 1 for CPMG XY8 variable-N sequence.")

        # want ending pi/2 pulse to be -90 for n pulses mod 8: 1,4,7,8 
        self.end_phase_sequence_string_int = int("1101011101011111", 2) # 01 is 90 and 11 is -90 degrees for the final pi/2 pulse encoded in bits 

        self.n_cpmg_sweep_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg_sweep',
            init_val=0,
        )

        self.add_sweep(
            NVQickSweep(
                self,
                self.n_cpmg_sweep_register,
                self.cfg.n_cpmg_start,
                self.cfg.n_cpmg_end,
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

        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds, n_cpmg_pulses=self.n_cpmg_sweep_register, vary_n=True)

        # Final tau and readout pi/2 pulse.
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            waveform="half_pi_0",
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
            phase=self.deg2reg(-90),
        )
        self.offset_computations(pi2_after=True)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def offset_computations(self, pi2_after=False, delay_tau_tdds=None):
        """
        Computes waveform address, phase, and coarse/fine delay correction.
        """
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', delay_tau_tdds)

        if not pi2_after:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '+', delay_tau_tdds)
            self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi_to_pi_correction_tdds)
        else:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi_to_pi2_correction_tdds)
            # set the phase register for last pi/2 pulse based on N mod 8 to get correct final readout phase.
            self.phase_register.set_to(self.end_phase_sequence_string_int, physical_unit=False)
            self.bitwi(self.phase_step_register.page, self.phase_step_register.addr, self.n_cpmg_sweep_register.addr, '&', 7)
            self.bitwi(self.phase_step_register.page, self.phase_step_register.addr, self.phase_step_register.addr, "<<", 1,) # multiply by 2
            self.bitw(self.phase_register.page, self.phase_register.addr, self.phase_register.addr, ">>", self.phase_step_register.addr,)
            self.bitwi(
                self.phase_register.page,
                self.phase_register.addr,
                self.phase_register.addr,
                "&",
                3,
            ) # mod 4 to get the correct 2-bit phase for the final pi/2 pulse
            self.bitwi(
                self.phase_register.page,
                self.phase_register.addr,
                self.phase_register.addr,
                "<<",
                30,
            )
            
        self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi_len_unused_tdds)

        # Coarse delay in treg units.
        self.bitwi(
            self.tdds_offset_register.page,
            self.treg_offset_register.addr,
            self.tdds_offset_register.addr,
            ">>",
            int(np.log2(self.samps_per_clk)),
        )

        # Fine offset in tdds units.
        self.bitwi(
            self.tdds_offset_register.page,
            self.tdds_offset_register.addr,
            self.tdds_offset_register.addr,
            "&",
            self.samps_per_clk - 1,
        )

        # Waveform select from fine offset.
        self.address_register.set_to(
            self.tdds_offset_register,
            '*',
            self.pi_waveform_len_treg + self.half_pi_waveform_len_treg,
            physical_unit=False,
        )

        if not pi2_after:
            self.address_register.set_to(
                self.address_register,
                '+',
                self.half_pi_waveform_len_treg,
                physical_unit=False,
            )

            # phase_step_register is already maintained modulo 8 in the pulse loop.
            self.phase_register.set_to(self.phase_sequence_string_int, physical_unit=False)
            self.bitw(
                self.phase_register.page,
                self.phase_register.addr,
                self.phase_register.addr,
                ">>",
                self.phase_step_register.addr,
            )
            self.bitwi(
                self.phase_register.page,
                self.phase_register.addr,
                self.phase_register.addr,
                "&",
                1,
            )
            self.bitwi(
                self.phase_register.page,
                self.phase_register.addr,
                self.phase_register.addr,
                "<<",
                30,
            )
