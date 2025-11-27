from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qick.averager_program import QickSweep


class CPMG(NVAveragerProgram):

    required_cfg = [
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "freq_freg",
        "mw_pi2_treg",
        "delay_start_treg",
        "delay_end_treg",
        "nsweep_points",
        "n_cpmg", # number of pi pulses in the cpmg sequence
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
                                     length=self.cfg.mw_pi2_treg
                                     )

        # things that might change for the MW 
        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 phase = 0
                                 )
    

        # Addd loops
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                               name='delay', 
                                               init_val=self.cfg.delay_start_treg)


        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg',
            init_val=self.cfg.n_cpmg - 1) # -1 because the first pi pulse is taken care of already

        self.add_sweep(NVQickSweep(
            self, 
            self.delay_register,
            self.cfg.delay_start_treg, 
            self.cfg.delay_end_treg,
            self.cfg.nsweep_points))
        
        self.synci(100)  # give processor some time to configure pulses

    def body(self):
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.n_cpmg_register.reset()            
        self.label("LOOP_ncpmg{}".format(0))  
        self.sync(self.delay_register.page, self.delay_register.addr)
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()
        self.sync(self.delay_register.page, self.delay_register.addr)
        self.loopnz(
                self.n_cpmg_register.page,
                self.n_cpmg_register.addr,
                'LOOP_ncpmg{}'.format(0))
        
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(-90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)