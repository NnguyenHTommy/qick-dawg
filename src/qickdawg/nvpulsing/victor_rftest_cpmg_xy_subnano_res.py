'''
RFTest CPMG-XY
=======================================================================
RFTest Envelope class used to test the shape of RF envelopes.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
import numpy as np

class RFTest_CPMG(NVAveragerProgram):
    '''
    An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
    '''
    required_cfg = [        
        "mw_pi2_samples", # length of pi/2 pulse
        "tau_len_samples", # length of tau delay
        "freq_freg", # Microwave freq # ~1405 MHz per Tommy
        "n_cpmg", # number of cpmgxy8 rounds
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", #MW Gain
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
    ]

    def initialize(self):
        # NVConfiguration class does not have Gain units unlike freq, time, or phase
        self.check_cfg()

        # check for min tau length has to be greater than some ns
        #maybe put a limit on gain later

        # Get mw registers
        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        # Configure the waveforms for different sample offsets
        # Waveforms must have at least a length of 3 treg
        self.pi_len_samples = self.cfg.mw_pi2_samples * 2
        self.pi_waveform_len_treg = max(int(np.ceil((self.pi_len_samples + 15) / 16)), 3)
        self.half_pi_waveform_len_treg = max(int(np.ceil((self.cfg.mw_pi2_samples + 15) / 16)), 3)

        for i in range(16):
            # half pi
            i_data = np.zeros(self.half_pi_waveform_len_treg * 16)
            q_data = np.zeros(self.half_pi_waveform_len_treg * 16)
            i_data[i : i + self.cfg.mw_pi2_samples] = 1
            q_data[i : i + self.cfg.mw_pi2_samples] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"half_pi_{i}", idata=i_data, qdata=q_data)

            # pi pulse
            i_data = np.zeros(self.pi_waveform_len_treg * 16)
            q_data = np.zeros(self.pi_waveform_len_treg * 16)
            i_data[i: i + self.pi_len_samples] = 1
            q_data[i: i + self.pi_len_samples] = 1
            i_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            q_data *= self.soccfg.get_maxv(self.cfg.mw_channel)
            self.add_envelope(ch=self.cfg.mw_channel, name=f"pi_{i}", idata=i_data, qdata=q_data)

        # Compute how much delay is in the waveforms
        self.pi_len_unused = self.pi_waveform_len_treg*16 - self.pi_len_samples
        self.half_pi_len_unused = self.half_pi_waveform_len_treg*16 - self.cfg.mw_pi2_samples

        # Set up registers for storing tau treg and sample offsets
        self.tau_samples = self.new_gen_reg(self.cfg.mw_channel,
                                            name='tau_step',
                                            init_val=self.cfg.tau_len_samples - self.pi_len_unused)
        
        # we can initialize sample_offset to already account for the first half_pi_pulse
        self.sample_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='sample_offset',
                                                    init_val=(self.pi_len_unused-self.half_pi_len_unused))
        self.treg_offset = self.new_gen_reg(self.cfg.mw_channel,
                                                name='treg_offset',
                                                init_val=0)
        
        # CPMG loop register
        self.n_cpmg_register = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='ncpmg',
                                                    init_val=self.cfg.n_cpmg - 1)
        
        # comparison register
        self.comparison = self.new_gen_reg(self.cfg.mw_channel,
                                                    name='comparison_val',
                                                    init_val=0)
        
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='arb',
                                     freq=self.cfg.freq_freg,
                                     gain=self.cfg.mw_gain)
        s = self.cfg.tau_len_samples - self.pi_len_unused
        self.add_sweep(NVQickSweep(
            self, 
            self.tau_samples,
            s,
            s + 45,
            10))
        
       

        # CPMG waveform
        self.synci(200)  # give processor some time to configure pulses


    def body(self):
        # Set first half pi x
        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", phase=0)
        self.offset_computations() # offset comp is for the very next sync but the one after pulse
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.sync(self.treg_offset.page, self.treg_offset.addr)

        # Loop pi-X, tau, pi-Y, tau
        self.n_cpmg_register.reset()
        self.label("LOOP_ncpmg")
        
        # X pulse
        # Configures assembly code for picking the waveform
        self.set_waveform("Execute_X_Pi_Pulse", "pi_", phase=0)
        self.offset_computations()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.sync(self.treg_offset.page, self.treg_offset.addr)

        # Y pulse
        self.set_waveform("Execute_Y_Pi_Pulse", "pi_", phase=90)
        self.offset_computations()
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        self.sync(self.treg_offset.page, self.treg_offset.addr)

        self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                'LOOP_ncpmg')
        
        # Pi/2 X pulse
        self.set_waveform("Execute_Last_Pulse", "half_pi_", phase=0)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.sample_offset.reset() # reset the sample_offset adjustment 
        self.sync_all(self.cfg.pulse_seq_delay_treg)
  
    def set_waveform(self, label, pulse_type="pi_", phase=0):
        """
        Configures the assembly code necessary for setting the waveform
        """
        self.select_waveform(4, 8, 16, label, pulse_type, phase)
        self.label(label)
    
    def offset_computations(self):
        """
        Compute the sample_offset and treg_offset for the next pulse
        """
        # sample_offset = sample_offset + sample_step
        self.math(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "+", self.tau_samples.addr)
        # treg_offset = sample_offset >> 4 (global offset)
        self.mathi(self.sample_offset.page, self.treg_offset.addr, self.sample_offset.addr, ">>", 4)
        # sample_offset = sample_offset & 15
        self.mathi(self.sample_offset.page, self.sample_offset.addr, self.sample_offset.addr, "&", 15)

    def select_waveform(self, depth, center, span, label, pulse_type="pi_", phase=0):
        """
        A binary search tree to select the correct waveform for a given sample_offset
        """
        center = int(center)
        if (depth==0):
            self.set_pulse_registers(ch=self.cfg.mw_channel, waveform=f"{pulse_type}{center}", phase=self.deg2reg(phase))
            self.condj(self.sample_offset.page, self.sample_offset.addr, "==", self.sample_offset.addr, label)
            return

        self.regwi(self.comparison.page, self.comparison.addr, center)
        self.condj(
                self.sample_offset.page,
                self.sample_offset.addr,
                ">=",
                self.comparison.addr,
                f"{pulse_type}{phase}_pulse_offset_{center}")
        
        self.select_waveform(depth-1, center-span/4, span/2, label, pulse_type, phase)

        self.label(f"{pulse_type}{phase}_pulse_offset_{center}")
        self.select_waveform(depth-1, center+span/4, span/2, label, pulse_type, phase)