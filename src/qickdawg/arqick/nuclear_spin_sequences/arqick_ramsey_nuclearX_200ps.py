'''
Nuclear Ramsey pulsing program
===================================================================
Needs to have init, ramsey, then tomography
https://www.nature.com/articles/nnano.2014.2#Sec2
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np


class NuclearRamseyFineResX(NVAveragerProgram):
    '''
    Nuclear Ramsey pulsing program with sub-nanosecond resolution for <X> basis readout
    '''

    required_cfg = [
        "mw_pi2_tdds",  # length of pi/2 pulse
        "n_cpmg_ramsey_start",
        "n_cpmg_ramsey_end",
        "nsweep_points",
        "delay_tdds_ramsey",
        "freq_freg",  # microwave freq
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "mw_gain",  # MW gain
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start

        "delay_tdds_gate_crxpi2",
        "n_cpmg_gate_crxpi2",
        "delay_tdds_gate_rzpi2",
        "n_cpmg_gate_rzpi2",
        "reinit_tdds_time" # tricky to compute because want to init after the tau delay. will do this on artiq/client side
    ]

    def initialize(self):
        self.check_cfg()
        
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
        self.half_pi_len_unused_tdds = self.half_pi_waveform_len_treg * self.samps_per_clk - self.cfg.mw_pi2_tdds

        # Delay correction registers.
        self.tdds_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='tdds_offset',
            init_val=0,
        )
        self.treg_offset_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='treg_offset',
            init_val=0,
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

        # Main loop register: total number of pi pulses.
        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg',
            init_val=0,
        )

        self.n_cpmg_sweep_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg_sweep',
            init_val=self.cfg.n_cpmg_ramsey_start,
        )

        self.add_sweep(
            NVQickSweep(
                self,
                self.n_cpmg_sweep_register,
                self.cfg.n_cpmg_ramsey_start,
                self.cfg.n_cpmg_ramsey_end,
                self.cfg.nsweep_points,
            )
        )

        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        # reset things that need to be reset each sequence iteration
        self.tdds_offset_register.reset()

        # NUCLEAR SPIN INITIALIZATION 
        # e RY(pi/2) 
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0",phase=self.deg2reg(0))
        self.offset_computations(after_pi2=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # n RZ(pi/2)
        self.cpmg_xy8_gate(gate_index=1, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_rzpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_rzpi2)

        # e-n CROTX(pi/2)
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.delay_tdds_gate_rzpi2 - self.cfg.delay_tdds_gate_crxpi2)
        self.cpmg_xy8_gate(gate_index=2, pi_2_pulse_before=False, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # would do self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.delay_tdds_gate_crxpi2 - self.cfg.delay_tdds_gate_crxpi2) but its 0 so can omit
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.reinit_tdds_time)

        # NUCLEAR SPIN RAMSEY SEQUENCE
        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=3, pi_2_pulse_before=False, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # n RZ(wt) for variable time
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.cfg.delay_tdds_gate_crxpi2 - self.cfg.delay_tdds_ramsey)
        self.cpmg_xy8_gate(gate_index=4, pi_2_pulse_before=False, delay_tau_tdds=self.cfg.delay_tdds_ramsey, n_cpmg_pulses=None, vary_n=True)

        # TOMOGRAPHY <X> basis 
        # diverging from universal control paper to follow 10 qubit paper tomography since looks better and makes more sense

        # e RY(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0",phase=self.deg2reg(90))
        self.offset_computations(after_pi2=True, delay_tau_tdds=self.cfg.delay_tdds_ramsey)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # e-n CROTX(pi/2)
        self.cpmg_xy8_gate(gate_index=5, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2, n_cpmg_pulses=self.cfg.n_cpmg_gate_crxpi2)

        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0",phase=self.deg2reg(0))
        self.offset_computations(after_pi2=True, delay_tau_tdds=self.cfg.delay_tdds_gate_crxpi2)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)

    def cpmg_xy8_gate(self, gate_index, pi_2_pulse_before, delay_tau_tdds, n_cpmg_pulses, vary_n = False):
        if pi_2_pulse_before:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.pi_len_unused_tdds - self.half_pi_len_unused_tdds)
            self.tdds_offset_register.set_to(self.tdds_offset_register, '-', delay_tau_tdds) 
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", phase=self.deg2reg(0))
        
        if vary_n:
            self.n_cpmg_register.set_to(self.n_cpmg_sweep_register, '-', 1, physical_unit=False)
        else:
            self.n_cpmg_register.set_to(n_cpmg_pulses - 1, physical_unit=False)
        self.phase_step_register.reset()  
        self.label("LOOP_ncpmg"+str(gate_index)) # TODO: this might not work so check. also maybe more efficient way
        self.offset_computations(after_pi2=False, delay_tau_tdds=delay_tau_tdds)
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
            'LOOP_ncpmg'+str(gate_index),
        )


    def offset_computations(self, after_pi2=False, delay_tau_tdds=None):
        """
        Computes waveform address, phase, and coarse/fine delay correction.
        """

        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', delay_tau_tdds)

        if not after_pi2:
            self.tdds_offset_register.set_to(self.tdds_offset_register, '+', delay_tau_tdds)

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

        if not after_pi2:
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
