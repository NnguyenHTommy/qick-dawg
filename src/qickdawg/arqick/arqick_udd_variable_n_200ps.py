'''
UDD sub-nanosecond resolution pulsing program with arbitrary N
==============================================================
Min resolution of 200 ps for the delays between pulses in a Uhrig
dynamical decoupling sequence. The basic unit is a UDDn of unit time t
whose n pi pulse centers sit at t_j = t*sin^2(pi*j/(2n+2)), and that
unit is repeated N_udd times, i.e. (UDDn)^N. The n+1 intervals are all
different and are read one at a time out of a host loaded table in the
tProc data memory. N_udd = 1 is a single UDDn sequence, and n_udd = 1
with N_udd = 1 is a spin echo.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.arqick.standard_ops_udd import StandardOpsUDD


class UDDAnyNFineRes(StandardOpsUDD, NVAveragerProgram):
    '''
    UDD pulsing program where n_udd is the number of pi pulses in one unit
    and N_udd is the number of times that unit is repeated. Both can be any
    positive integer, and the total number of pi pulses is n_udd*N_udd.

    Phase pattern repeats XYXYYXYX continuously across all pulses when
    phase_mode is 'xy8', and is all X when phase_mode is 'x', so:
    - n_udd=2, N_udd=2, 'xy8' -> XYXY
    - n_udd=2, N_udd=2, 'x'   -> XXXX
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "freq_freg",  # microwave freq
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW gain
        "phase_mode",  # 'xy8' or 'x' pattern for the pi pulses
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start

        "table_addr_start",
        "table_addr_end",
        "nsweep_points",
        "n_udd",  # pi pulses in one unit, can be any positive integer
        "N_udd",  # number of times the unit is repeated, can be any positive integer
    ]

    def initialize(self):
        self.init()

        # Precompute the final readout pi/2 phase from the total pulse count using XY8 periodicity.
        # Mapping: total mod 8 in {0,1,4,7} -> -90, {2,3,5,6} -> +90. All X never flips, so always -90.
        if self.cfg.phase_mode == 'xy8' and (self.cfg.n_udd * self.cfg.N_udd) % 8 not in [0, 1, 4, 7]:
            self.last_pi2_phase_deg = 90
        else:
            self.last_pi2_phase_deg = -90

        self.table_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='table',
            init_val=self.cfg.table_addr_start,
        )

        self.add_sweep(
            NVQickSweep(
                self,
                self.table_register,
                self.cfg.table_addr_start,
                self.cfg.table_addr_end,
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

        self.udd_gate(gate_index=0, pi_2_pulse_before=True, table_pointer=self.table_register, n_udd_pulses=self.cfg.n_udd, n_udd_units=self.cfg.N_udd, vary_n=False)

        # Final interval and readout pi/2 pulse. Delta_(n+1) equals Delta_1, which is table index 0.
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(self.last_pi2_phase_deg))
        self.pointer_register.set_to(self.table_register, '+', 0, physical_unit=False)
        self.memr(self.spacing_register.page, self.spacing_register.addr, self.pointer_register.addr)
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.spacing_register)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)