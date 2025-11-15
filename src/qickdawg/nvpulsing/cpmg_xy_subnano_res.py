'''
RFTest CPMG-XY
=======================================================================
RFTest Envelope class used to test the shape of RF envelopes.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np

class CPMGXY8FineRes(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [        
        "mw_pi2_tdds", # length of pi/2 pulse
        "delay_tdds_start",
        "delay_tdds_end",
        "nsweep_points",
        "freq_freg", # Microwave freq 
        "n_cpmg", # number of cpmgxy8 rounds
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
    ]

    def initialize(self):
        self.check_cfg()

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Get samps per clk for later calculations. should be 16 for mw with current version 11/14/2025
        self.samps_per_clk = self.soccfg['gens'][self.cfg.mw_channel]['samps_per_clk']

        # Configure the waveforms for different fine resolution delay steps
        # must be 16 of them for pi and pi/2 pulses. Need to do all if sweeping at this fine resolution
        self.pi_len_tdds = self.cfg.mw_pi2_tdds * 2
        # Waveforms must have at least a length of 3 treg units
        self.pi_waveform_len_treg = max(int(np.ceil((self.pi_len_tdds + self.samps_per_clk-1) / self.samps_per_clk)), 3)
        self.half_pi_waveform_len_treg = max(int(np.ceil((self.cfg.mw_pi2_tdds + self.samps_per_clk-1) / self.samps_per_clk)), 3)
        
        for i in range(16):
            # pi/2 pulse
            i_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            q_data = np.zeros(self.half_pi_waveform_len_treg * self.samps_per_clk)
            i_data[i : i + self.cfg.mw_pi2_tdds] = 1
            q_data[i : i + self.cfg.mw_pi2_tdds] = 1
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
        #print(self.envelopes[self.cfg.mw_channel]['envs'])
        # default special register but need to manually modify them later
        self.address_register = self.get_gen_reg(self.cfg.mw_channel, name='addr') # for specifying which waveform
        self.phase_register = self.get_gen_reg(self.cfg.mw_channel, name='phase') # for specifying phase

        # there is dead time in the waveforms due to the waveform length being in treg units
        # need to account for this in delays so need the values 
        self.pi_len_unused_tdds = self.pi_waveform_len_treg*self.samps_per_clk - self.pi_len_tdds
        self.half_pi_len_unused_tdds = self.half_pi_waveform_len_treg*self.samps_per_clk - self.cfg.mw_pi2_tdds

        # two registers needed to account for unused time for a half pi pulse and pi pulse
        # one for coarse adjustment (treg_offset) and one for fine adjustment (tdds_offset)
        # we can initialize tdds_offset to already account for the first half_pi_pulse
        self.tdds_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='tdds_offset',
                                                    init_val=self.pi_len_unused_tdds-self.half_pi_len_unused_tdds)

        self.treg_offset_register = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
        # CPMG loop register
        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='ncpmg',
                                                    init_val=self.cfg.n_cpmg - 1)
        
        # phase sequence loop register
        # Sequence is XYXYYXYX -> 0b10100101 = 165. Here did X to be 1 since sequence starts with X.
        self.phase_sequence_string_int = int("10100101", 2)
        self.phase_sequence_register = self.new_gen_reg(self.cfg.mw_channel,
                                            name='phase_sequence',
                                            init_val=7)
        
        # mw pulse register
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain)
        
        # Set up register for storing and sweeping delays
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                            name='delay',
                                            init_val=self.cfg.delay_tdds_start - self.pi_len_unused_tdds)

        # Set up a delay factor register if needed for scaling between 1*tau and 2*tau
        self.delay_factor_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='delay_factor',
                                                    init_val=1) 
        
        self.add_sweep(NVQickSweep(
            self, 
            self.delay_register,
            self.cfg.delay_tdds_start - self.pi_len_unused_tdds,
            self.cfg.delay_tdds_end - self.pi_len_unused_tdds, #end time 
            self.cfg.nsweep_points)) # nsweep points
        
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.tdds_offset_register.reset() # reset the dds_offset adjustment
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=90)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # Loop n_cpmg times
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg")

        # loop phase sequence
        self.phase_sequence_register.reset()
        self.label("LOOP_phase_sequence")
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="pi_0", phase=0)

        self.offset_computations(last_pi2=False) # compute offsets and set waveform and phase
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.loopnz(
                self.phase_sequence_register.page,
                self.phase_sequence_register.addr,
                'LOOP_phase_sequence')
        self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                'LOOP_ncpmg')
        
        # last tau and pi/2 pulse of -Y
        self.delay_factor_register.reset()  # set delay factor to 1 for 1*tau
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=-90)
        self.offset_computations(last_pi2=True) # compute offsets and set waveform and phase
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)
    
    def offset_computations(self, last_pi2 = False):
        # Computes the total delay needed until the next pulse from the end of this waveform
        # by adding amount of samples to wait + current sample offset
        self.delay_factor_register.set_to(self.delay_register, '*', self.delay_factor_register)
        self.tdds_offset_register.set_to(self.delay_factor_register, '+', self.tdds_offset_register)
        if last_pi2:
            self.delay_factor_register.set_to(1)  # set delay factor to 1 for 1*tau
        else:
            self.delay_factor_register.set_to(2)  # set delay factor to 2 for 2*tau
        # Computes how long to stall the FPGA output in tproc cycles from the total delay.
        # This operation also converts from samples (200ps) to treg (3.2ns)
        self.bitwi(self.tdds_offset_register.page, self.treg_offset_register.addr, self.tdds_offset_register.addr, ">>", int(np.log2(self.samps_per_clk)))
        # Computes the remaining samples that the pulse should be delayed by
        # This is equivalent to: total delay (in samples) - fpga delay (in samples)
        self.bitwi(self.tdds_offset_register.page, self.tdds_offset_register.addr, self.tdds_offset_register.addr, "&", self.samps_per_clk - 1)

        # updating address register to select correct waveform based on the current offset
        self.address_register.set_to(self.tdds_offset_register, '*', self.pi_waveform_len_treg+self.half_pi_waveform_len_treg, physical_unit = False)
        if last_pi2:
            self.phase_register.set_to(-90, physical_unit=True) # set last pi/2 pulse to -Y
        else:
            self.address_register.set_to(self.address_register, '+', self.half_pi_waveform_len_treg, physical_unit = False)

            # setting the phase. We do this in binary where X is 1 and Y is 0 so XYXYYXYX = 0b10100101
            self.phase_register.set_to(self.phase_sequence_string_int, physical_unit=False)
            self.bitw(self.phase_register.page, self.phase_register.addr, self.phase_register.addr, ">>", self.phase_sequence_register.addr)
            self.bitwi(self.phase_register.page, self.phase_register.addr, self.phase_register.addr, "&", 1)
            self.phase_register.set_to(self.phase_register, '*', 90, physical_unit=True) # set phase to 0 or 90 based on LSB