'''
Sweeping end phase for DDRF spectroscopy
=======================================================================
Min resolution of 200ps for delay steps between pulses in CPMG XY8 sequence
using fine control of waveform start address and phase.
'''

from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing.nvqicksweep import NVQickSweep
from qickdawg.arqick.standard_ops import StandardOps

class CPMGXY8EndPhaseSweepMbpFineRes(StandardOps, NVAveragerProgram):
    '''
    CPMG XY8 sub-nanosecond resolution pulsing program with end phase sweep
    '''
    required_cfg = [        
        "mw_pi2_tdds", # length of pi/2 pulse
        "freq_freg", # Microwave freq 
        "mw_channel", # MW Channel
        "mw_nqz", # 1 at 1405 MHz
        "mw_gain", # MW Gain
        "reps",
        "pmod_out_pin", # should be 0 for PMOD0_0
        "pmod_out_pulse_width_treg", # 50ns is reasonable
        "pmod_out_trig_delay_treg", # delay between trigger and pulse seq start. this is added to the already 198 inherent ns delay so putting 300 means 198+300=498ns delay
        "inherent_trigger_to_pulses_delay_treg", # should be 209.27ns 
        "pulse_seq_delay_treg", # delay between pulse seq end and trigger start of next seq
        
        "delay_tdds",
        "end_phase_start_preg",
        "end_phase_end_preg",
        "nsweep_points",
        "n_cpmg",

        "adc_channel",
        "adc_trig_offset_treg",
        "readout_threshold",
        "readout_integration_treg",
        "qick_processing_time_after_qick_readout_treg",
        "delay_before_charge_check_repeats_treg",
        "pmod_out_trig_to_mw_delay_treg",
    ]

    def initialize(self):
        self.init()
        self.setup_readout()
        self.mathi(0, 2, 2, "==", 0)
        self.r_thresh = 6
        self.regwi(0, self.r_thresh, self.cfg.readout_threshold)

        self.end_phase_register = self.new_gen_reg(self.cfg.mw_channel,
                                            name='end_phase',
                                            init_val=0)

        self.add_sweep(NVQickSweep(
            self, 
            self.end_phase_register,
            self.cfg.end_phase_start_preg,
            self.cfg.end_phase_end_preg,
            self.cfg.nsweep_points))
        
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.mathi(0, 2, 2, "==", 0)
        self.label("wait_for_trigger")

        # self.pmod_trigger_sequence()
        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)
        self.trigger(pins=[self.cfg.pmod_out_pin], width=self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_delay_treg)

        self.tdds_offset_register.reset()

        self.sync_all(self.cfg.inherent_trigger_to_pulses_delay_treg)   #  +209ns
        self.trigger(pins=[self.cfg.pmod_out_pin],
                     adcs=[self.cfg.adc_channel],
                     width=self.cfg.readout_integration_treg)
        self.wait_all(200) # pause until 200 clocks past the end of the readout window
        self.read(0,0,"lower",2)
        self.condj(0,2,'>',self.r_thresh,"skip_to_mw_pulse")
        self.sync_all(self.cfg.delay_before_charge_check_repeats_treg)  # w/ -209ns
        self.condj(0,2,'<',self.r_thresh,"wait_for_trigger")

        self.label("skip_to_mw_pulse")
        self.sync_all(self.cfg.qick_processing_time_after_qick_readout_treg)
        self.trigger(pins = [self.cfg.pmod_out_pin],
                     width = self.cfg.pmod_out_pulse_width_treg)
        self.sync_all(self.cfg.pmod_out_trig_to_mw_delay_treg)          # w/ -209ns

        self.set_pulse_registers(ch=self.cfg.mw_channel, waveform="half_pi_0", freq=self.cfg.freq_freg, gain=self.cfg.mw_gain, phase=self.deg2reg(90))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        self.cpmg_xy8_gate(gate_index=0, pi_2_pulse_before=True, delay_tau_tdds=self.cfg.delay_tdds, n_cpmg_pulses=self.cfg.n_cpmg)

        # Final tau and readout pi/2 pulse.
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            waveform="half_pi_0",
            freq=self.cfg.freq_freg,
            gain=self.cfg.mw_gain,
            phase=self.deg2reg(0),
        )
        self.offset_computations(pi2_after=True, delay_tau_tdds=self.cfg.delay_tdds)
        self.phase_register.set_to(self.end_phase_register, physical_unit=False)
        self.sync(self.treg_offset_register.page, self.treg_offset_register.addr)
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.pulse_seq_delay_treg)