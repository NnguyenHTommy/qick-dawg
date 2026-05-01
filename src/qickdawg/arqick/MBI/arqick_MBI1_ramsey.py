'''
N14 MBI with 1 MBI and then electron spin ramsey
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.arqick.standard_ops import StandardOps
import numpy as np

class N14MBI_electron_ramsey(StandardOps, NVAveragerProgram):

    required_cfg = [
        # params that usually won't change
        "mw_channel",  # MW channel
        "mw_nqz",  # 1 at 1405 MHz
        "reps",
        "pmod_out_pin",  # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg",  # 50ns is reasonable
        "pmod_out_trig_delay_treg",  # delay between trigger and pulse seq start
        "inherent_trigger_to_pulses_delay_treg",  # should be 209.27ns
        "pulse_seq_delay_treg",  # delay between sequence end and next trigger start
        
        # for the normal pulse part
        "mw_gain",  # MW gain
        "freq_freg",  # microwave freq
        "mw_pi2_tdds",  # length of pi/2 pulse

        # for the MBI part
        "mBI_pi_tdds",  # length of pi pulse for MBI
        "mBI_mw_gain",  # MW gain for MBI, can be different from the gain for the CPMG part
        "mBI_freq_freg",  # microwave freq for MBI, can be different from the freq for the CPMG part

        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "ramsey_freq_freg",
        "inital_delay_tdds", # should be > 100ns, maybe 200ns to be safe
        "N14_measurement_delay_tdds",
    ]

    def initialize(self):
        self.init()
        if self.cfg.mBI_pi_tdds % 2 != 0:
            raise ValueError("For this sequence, we require the MBI pi pulse to have even number of tdds for easier timing alignment. Please adjust")
        self.mBI_pi_waveform_len_treg = max(
            int(np.ceil((self.cfg.mBI_pi_tdds + self.samps_per_clk - 1) / self.samps_per_clk)), 3
        )
        self.mBI_offset_tdds = int(self.cfg.inital_delay_tdds - self.pi_len_unused_tdds - (self.pi_len_tdds/2 + self.cfg.mBI_pi_tdds/2))
        self.mBI_offset_mod_16_tdds = self.mBI_offset_tdds % self.samps_per_clk
        i_data = np.zeros(self.mBI_pi_waveform_len_treg * self.samps_per_clk)
        q_data = np.zeros(self.mBI_pi_waveform_len_treg * self.samps_per_clk)
        i_data[self.mBI_offset_mod_16_tdds:self.mBI_offset_mod_16_tdds + self.cfg.mBI_pi_tdds] = 1
        q_data[self.mBI_offset_mod_16_tdds:self.mBI_offset_mod_16_tdds + self.cfg.mBI_pi_tdds] = 1
        i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
        self.add_envelope(ch=self.cfg.mw_channel, name="mBI_pi", idata=i_data, qdata=q_data)
        self.mBI_pi_len_unused_tdds = self.mBI_pi_waveform_len_treg * self.samps_per_clk - self.cfg.mBI_pi_tdds

        self.mBI_readout_tdds = int(self.cfg.N14_measurement_delay_tdds - self.mBI_pi_len_unused_tdds - (self.cfg.mBI_pi_tdds/2 + self.cfg.mw_pi2_tdds/2))

        self.delay_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='delay',
            init_val=0,
        )

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

        # electron pi pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # N14 specific pi pulse for MBI
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="mBI_pi", freq=self.cfg.mBI_freq_freg, gain=self.cfg.mBI_mw_gain, phase=self.deg2reg(0))
        self.tdds_offset_register.set_to(self.mBI_offset_tdds)
        self.bitwi(
            self.tdds_offset_register.page,
            self.treg_offset_register.addr,
            self.tdds_offset_register.addr,
            ">>",
            int(np.log2(self.samps_per_clk)),
        )
        self.bitwi(
            self.tdds_offset_register.page,
            self.tdds_offset_register.addr,
            self.tdds_offset_register.addr,
            "&",
            self.samps_per_clk - 1,
        )
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # wait for readout
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.mBI_readout_tdds)
        
        # electron Ramsey sequence
        # e RX(pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.ramsey_freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(0))
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
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # wait 
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.delay_register)
        self.tdds_offset_register.set_to(self.tdds_offset_register, '+', self.pi_len_unused_tdds - self.half_pi_len_unused_tdds + self.pi_to_pi2_correction_tdds - self.pi2_to_pi2_correction_tdds)
        # e RX(-pi/2)
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.ramsey_freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(180))
        self.offset_computations(pi2_after=True, delay_tau_tdds=0)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)