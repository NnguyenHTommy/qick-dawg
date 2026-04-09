'''
CPMG XY8 sub-nanosecond resolution pulsing program with arbitrary N
===================================================================
Min resolution of 200 ps for delay steps between pulses in a CPMG XY8
phase sequence. Unlike round-based XY8 implementations, this version
treats n_cpmg as the total number of pi pulses, so N can be any
positive integer.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np


class CPMGXY8AnyNFineRes(NVAveragerProgram):
    '''
    CPMG XY8 sub-nanosecond resolution pulsing program where n_cpmg is the
    total number of pi pulses (any positive integer).

    Phase pattern repeats XYXYYXYX continuously, so:
    - n_cpmg=3  -> XYX
    - n_cpmg=10 -> XYXYYXYXXY
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg",  # microwave freq
        "n_cpmg",  # total number of pi pulses, can be any positive integer
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
    ]

    def initialize(self):
        self.check_cfg()
        if self.cfg.n_cpmg < 1:
            raise ValueError("n_cpmg must be >= 1 for CPMG XY8 variable-N sequence.")

        # Precompute the final readout pi/2 phase from N using XY8 periodicity.
        # Mapping: N mod 8 in {0,1,4,7} -> -90, {2,3,5,6} -> +90.
        if self.cfg.n_cpmg % 8 in [0, 1, 4, 7]:
            self.last_pi2_phase_deg = -90
        else:
            self.last_pi2_phase_deg = 90

        # Get mw registers.
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version.
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Configure waveforms for all fine-resolution start offsets.
        self.pi_len_tdds = self.cfg.mw_pi2_tdds * 2
        self.pi_waveform_len_treg = max(
            int(np.ceil((self.pi_len_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3
        )
        self.half_pi_waveform_len_treg = max(
            int(np.ceil((self.cfg.mw_pi2_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3
        )

        for i in range(16):
            # pi/2 pulse
            i_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            q_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            i_data[i: i + self.cfg.mw_pi2_tdds] = 1
            q_data[i: i + self.cfg.mw_pi2_tdds] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=i_data, qdata=q_data)

            # pi pulse
            i_data = np.zeros(self.pi_waveform_len_treg * self.samps_per_clk)
            q_data = np.zeros(self.pi_waveform_len_treg * self.samps_per_clk)
            i_data[i: i + self.pi_len_tdds] = 1
            q_data[i: i + self.pi_len_tdds] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=i_data, qdata=q_data)

        # default special registers but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr')  # waveform select
        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name='phase')  # NCO phase

        # Account for waveform dead time due to treg granularity.
        self.pi_len_unused_tdds = self.pi_waveform_len_treg * self.samps_per_clk - self.pi_len_tdds
        self.half_pi_len_unused_tdds = (
            self.half_pi_waveform_len_treg * self.samps_per_clk - self.cfg.mw_pi2_tdds
        )

        # Delay correction registers.
        self.tdds_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='tdds_offset',
            init_val=self.pi_len_unused_tdds - self.half_pi_len_unused_tdds,
        )
        self.treg_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='treg_offset',
            init_val=0,
        )

        # Main loop register: total number of pi pulses.
        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg',
            init_val=self.cfg.n_cpmg - 1,
        )

        # Tracks pulse index so phase follows XYXYYXYX cyclically for any N.
        self.phase_step_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='phase_step',
            init_val=0,
        )

        # Bit encoding X=0, Y=1 for XYXYYXYX over bits [0..7].
        self.phase_sequence_string_int = int("01011010", 2)

        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='arb',
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
        )

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
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.tdds_offset_register.reset()
        self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.delay_register)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Set pi pulse waveform once; phase and address are updated per pulse.
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", phase=self.deg2reg(0))

        self.n_cpmg_register.reset()
        self.phase_step_register.reset()
        self.label("LOOP_ncpmg")

        self.offset_computations(last_pi2=False)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Advance phase index for the next pi pulse.
        self.mathi(
            self.phase_step_register.page,
            self.phase_step_register.addr,
            self.phase_step_register.addr,
            "+",
            1,
        )
        self.bitwi(
            self.phase_step_register.page,
            self.phase_step_register.addr,
            self.phase_step_register.addr,
            "&",
            7,
        )

        self.loopnz(
            self.n_cpmg_register.page,
            self.n_cpmg_register.addr,
            'LOOP_ncpmg',
        )

        # Final tau and readout pi/2 pulse.
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            waveform="half_pi_0",
            phase=self.deg2reg(self.last_pi2_phase_deg),
        )
        self.offset_computations(last_pi2=True)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def offset_computations(self, last_pi2=False):
        """
        Computes waveform address, phase, and coarse/fine delay correction.
        Uses 1*tau delay for the final pi/2 pulse and 2*tau delay for pi pulses.
        """

        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)

        if not last_pi2:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)

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

        if not last_pi2:
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
