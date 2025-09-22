from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qick.averager_program import QickSweep


class Rabi(NVAveragerProgram):

    required_cfg = [
        "mw_channel",
        "mw_nqz",
        "counting_duration_treg", 
        "mw_gain",
        "freq_freg",
        "mw_duration_start_treg", # units of freg are Mhz
        "mw_duration_end_treg",
        "nsweep_points",
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
        "reps", # needed but not used
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        ]

    def initialize(self):
        # Get registers for mw
        self.check_cfg()
        
        self.declare_gen(ch=self.cfg.mw_channel,
                         nqz=self.cfg.mw_nqz)

        # # Setup pulse defaults microwave
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='const',
                                     gain=self.cfg.mw_gain,
                                     freq=self.cfg.freq_freg,
                                     phase=0)

        # things that might change for the MW 
        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 length=self.cfg.mw_duration_start_treg,
                                 )
    

        self.mw_length_register = self.new_gen_reg(self.cfg.mw_channel,
                                                   name='mw_length',
                                                   init_val=self.cfg.mw_duration_start_treg)

        self.add_sweep(NVQickSweep(self,
                                   reg=self.mw_length_register,
                                   start=self.cfg.mw_duration_start_treg,
                                   stop=self.cfg.mw_duration_end_treg,
                                   expts=self.cfg.nsweep_points,
                                   label='length', 
                                   mw_channel=self.cfg.mw_channel))

        
        self.synci(100)  # give processor some time to configure pulses

    def body(self):
        self.synci(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.synci(self.cfg.pmod_out_trig_delay_treg)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync(self.mw_length_register.page, self.mw_length_register.addr)
        self.synci(self.cfg.pulse_seq_delay_treg)
