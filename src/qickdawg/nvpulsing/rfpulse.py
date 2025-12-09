from .nvaverageprogram import NVAveragerProgram

class RFPulse(NVAveragerProgram):
   required_cfg = [
       "pulse_len_treg",
       "relax_delay_treg",
       "mw_channel",
       "mw_nqz",
       "mw_freg",
       "mw_off_freg",
       "reps",
       "init_delay_treg",
       "gain",
       "num_pulses",
       "pmod_out_pin",
       "pmod_out_pulse_width_treg",
       "pmod_out_trig_delay_treg"]
  
   def initialize(self):
       # NVConfiguration class does not have Gain units unlike freq, time, or phase
       # Get mw registers
       self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

       self.default_pulse_registers(ch=self.cfg.mw_channel,
                                        style='const',
                                        freq=self.cfg.mw_freg,
                                        length=self.cfg.pulse_len_treg,
                                        gain=self.cfg.gain)

       self.set_pulse_registers(ch=self.cfg.mw_channel,
                                    phase=0)

       self.synci(100)  # give processor some time to configure pulses


   def body(self):
        self.trigger(pins = [self.cfg.pmod_out_pin], width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel) 
        self.sync_all(self.cfg.relax_delay_treg)


         