from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qick.averager_program import QickSweep


class CWODMR(NVAveragerProgram):

    required_cfg = [
        "mw_channel",
        "mw_nqz",
        "counting_duration_treg", # rf and readout are on at the same time and same duration
        "mw_gain",
        "freq_start_freg", # units of freg are Mhz
        "freq_end_freg",
        "nsweep_points",
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start
        "reps", # needed but not used
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 198ns 
        ]

    def initialize(self):
        # Get registers for mw
        self.check_cfg()
        
        self.declare_gen(ch=self.cfg.mw_channel,
                         nqz=self.cfg.mw_nqz)

        # # Setup pulse defaults microwave
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='const',
                                     length=self.cfg.counting_duration_treg,
                                     gain=self.cfg.mw_gain)

        # things that might change for the MW 
        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 freq=self.cfg.freq_start_freg,
                                 phase=0)
    

        # Make frequency register and convert frequency values to integers
        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")
        # # # note this is different from NVQickSweep
        self.add_sweep(QickSweep(self,
                                 self.mw_frequency_register,
                                 self.cfg.freq_start_fMHz,
                                 self.cfg.freq_end_fMHz,
                                 self.cfg.nsweep_points))

        
        self.synci(100)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)
