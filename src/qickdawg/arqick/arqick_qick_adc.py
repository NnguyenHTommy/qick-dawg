from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qick.averager_program import QickSweep

class QickAdcTesting(NVAveragerProgram):
   '''
   An NVAveragerProgram class that generates RF gain and frequency stepping sequences.
   '''
   required_cfg = [
       "pmod_out_pin",
       "pmod_out_pulse_width_treg",
       "adc_channel",
       "adc_trig_offset_treg",
       "relax_delay_treg",
       "readout_integration_treg",
       "readout_threshold",
       "mw_channel",
       "mw_freg",
       "mw_nqz",
       "mw_pulse_len_treg",
       "reps",
       "gain",
       ]
  
   def initialize(self):
       # NVConfiguration class does not have Gain units unlike freq, time, or phase
       # need to call: cfg.add_unitless_linear_sweep(gain, start, stop, delta, nsweep_points)
       self.check_cfg()
       self.mathi(0, 2, 2, "==", 0)
       self.r_thresh = 6                                               # register 6 I believe
       self.regwi(0, self.r_thresh, self.cfg.readout_threshold)  # store threshdold into 6
       self.setup_readout()
       self.wait_all()

       # Get mw registers
       self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

       self.default_pulse_registers(ch=self.cfg.mw_channel,
                                        style='const',
                                        freq=self.cfg.mw_freg,
                                        length=self.cfg.mw_pulse_len_treg)

       self.set_pulse_registers(ch=self.cfg.mw_channel,
                                    phase=0, gain=0)

       self.synci(500)  # give processor some time to configure pulses

   def body(self):
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(90), gain=self.cfg.gain)        
        self.trigger(
            pins=[self.cfg.pmod_out_pin],
            adcs=[self.cfg.adc_channel],
            width=self.cfg.readout_integration_treg)
        
        self.wait_all(200)        
        self.read(0,0,"lower",2)
        self.condj(0,2,'>',self.r_thresh,'label_play_b')
        
        self.pulse(ch=self.cfg.mw_channel)
        self.condj(0, 0, ">=", 0, 'end_of_logic') # Skip Case B
        
        self.label('label_play_b')
        self.sync_all(self.us2cycles(1))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.relax_delay_treg)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.relax_delay_treg)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.relax_delay_treg)

        self.label('end_of_logic')
        self.sync_all()
        self.wait_all()
