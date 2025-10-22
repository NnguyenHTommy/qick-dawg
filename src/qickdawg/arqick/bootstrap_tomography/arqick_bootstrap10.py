from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qick.averager_program import QickSweep


class Bootstrap10(NVAveragerProgram):

    required_cfg = [
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "mw_pi2_treg",
        "mw_delay_treg", # time between mw pulses
        "freq_freg",
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
                                     length=self.cfg.mw_pi2_treg,
                                     freq=self.cfg.freq_freg)

        # things that might change for the MW 
        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 phase = 0
                                 )        
        
        self.synci(100)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.mw_delay_treg)
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.mw_delay_treg)   
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)