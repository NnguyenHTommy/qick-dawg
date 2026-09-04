# This file contains the UDD version of the standard operations, where the delay before and after each pi pulse is different.
import numpy as np
from qickdawg.arqick.standard_ops import StandardOps


class StandardOpsUDD(StandardOps):
    def init(self):
        super().init()

        # All X is the same bit extraction with every bit cleared.
        if self.cfg.phase_mode == 'x':
            self.phase_sequence_string_int = 0

        # Pointer into the host loaded spacing table, and the spacing it reads back.
        self.pointer_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='pointer',
            init_val=0,
        )
        self.spacing_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='spacing',
            init_val=0,
        )

        # Address of the spacing that opens the current unit. Delta_1 for the first unit, 2*Delta_1 after that.
        self.entry_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='entry',
            init_val=0,
        )

        # Inner loop register: pi pulses per unit. Outer loop register: number of units.
        self.n_udd_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='nudd',
            init_val=0,
        )
        self.N_udd_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='Nudd',
            init_val=0,
        )

    def place_pi_pulse(self):
        self.memr(self.spacing_register.page, self.spacing_register.addr, self.pointer_register.addr)
        self.offset_computations(pi2_after=False, delay_tau_tdds=self.spacing_register)
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

    # table_pointer is the swept register holding the first table address of this sweep point
    # a table row is [Delta_1, Delta_2, ..., Delta_n, 2*Delta_1], the last entry being the gap across a unit boundary
    def udd_gate(self, gate_index, pi_2_pulse_before, table_pointer, n_udd_pulses, n_udd_units, vary_n = False):
        if pi_2_pulse_before:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.pi_len_unused_tdds - self.half_pi_len_unused_tdds - self.pi_to_pi2_correction_tdds + self.pi_to_pi_correction_tdds)
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))

        self.entry_register.set_to(table_pointer, '+', 0, physical_unit=False)
        if vary_n:
            self.N_udd_register.set_to(self.N_udd_sweep_register, '-', 1, physical_unit=False)
        else:
            self.N_udd_register.set_to(n_udd_units - 1, physical_unit=False)
        self.phase_step_register.reset()

        self.label("LOOP_unit"+str(gate_index))
        self.pointer_register.set_to(self.entry_register, '+', 0, physical_unit=False)
        self.place_pi_pulse()

        if n_udd_pulses > 1:
            self.n_udd_register.set_to(n_udd_pulses - 2, physical_unit=False)
            self.pointer_register.set_to(table_pointer, '+', 1, physical_unit=False)
            self.label("LOOP_pulse"+str(gate_index))
            self.place_pi_pulse()
            self.pointer_register.set_to(self.pointer_register, '+', 1, physical_unit=False)
            self.loopnz(
                self.n_udd_register.page,
                self.n_udd_register.addr,
                'LOOP_pulse'+str(gate_index),
            )

        self.entry_register.set_to(table_pointer, '+', n_udd_pulses, physical_unit=False)
        self.loopnz(
            self.N_udd_register.page,
            self.N_udd_register.addr,
            'LOOP_unit'+str(gate_index),
        )


    def offset_computations(self, pi2_after=False, delay_tau_tdds=None):
        """
        Computes waveform address, phase, and coarse/fine delay correction.
        The spacing is added once, not twice, because UDD intervals are not symmetric about each pi pulse.
        """

        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', delay_tau_tdds)
        if pi2_after:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi_to_pi2_correction_tdds)
        else:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '-', self.pi_to_pi_correction_tdds)

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