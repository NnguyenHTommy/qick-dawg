from .nvaverageprogram import NVAveragerProgram
from .nvqicksweep import NVQickSweep


class CPMGXY8nDelaySweepInBody(NVAveragerProgram):

    required_cfg = [
        "mw_channel",
        "mw_nqz",
        "mw_pi2_treg",
        "mw_gain",
        "scaling_mode",
        "delay_start_treg",
        "delay_end_treg",
        "nsweep_points",
        "relax_delay_treg",
        "reps",
        "n_cpmg"]

    def initialize(self):
        # Get registers for mw

        self.declare_gen(ch=self.cfg.mw_channel,
                         nqz=self.cfg.mw_nqz)

        # Setup pulse defaults microwave
        self.default_pulse_registers(ch=self.cfg.mw_channel,
                                     style='const',
                                     freq=self.cfg.mw_freg,
                                     length=self.cfg.mw_pi2_treg,
                                     gain=self.cfg.mw_gain)

        self.set_pulse_registers(ch=self.cfg.mw_channel,
                                 phase=0)

        # Addd loops to loop tau 
        self.delay_register = self.new_gen_reg(self.cfg.mw_channel,
                                               name='delay',
                                               init_val=self.cfg.delay_start_treg)

        # for how many N CPMG pulses
        self.n_cpmg_register = self.new_gen_reg(
            self.cfg.mw_channel,
            name='ncpmg',
            init_val=self.cfg.n_cpmg - 1)

        if self.cfg.scaling_mode == 'exponential':
            self.add_sweep(NVQickSweep(
                self,
                self.delay_register,
                self.cfg.delay_start_treg,
                self.cfg.delay_end_treg,
                expts=self.cfg.nsweep_points,
                scaling_mode=self.cfg.scaling_mode,
                scaling_factor=self.cfg.scaling_factor))

        elif self.cfg.scaling_mode == 'linear':
            self.add_sweep(NVQickSweep(
                self,
                self.delay_register,
                self.cfg.delay_start_treg,
                self.cfg.delay_end_treg,
                self.cfg.nsweep_points))
        else:
            assert 0, 'cfg.scaling_mode must be "linear" or "exponential"'

        self.synci(100)  # give processor some time to configure pulses

    def body(self):

        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))
        ## Pulse Sequence 1
        # first reset phase to x

        # pi/2 - x
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all()

        # unit of XY8
        self.n_cpmg_register.reset()            
        self.label("LOOP_ncpmg{}".format(0))            
        for phase in [0, 90, 0, 90, 90, 0, 90, 0]:
            #  delay
            self.sync(self.delay_register.page, self.delay_register.addr)
            # set phase
            self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(phase))
            # pi - pulse
            self.pulse(ch=self.cfg.mw_channel)
            self.pulse(ch=self.cfg.mw_channel)
            self.sync_all()
            # delay
            self.sync(self.delay_register.page, self.delay_register.addr)
        self.loopnz(
            self.n_cpmg_register.page,
            self.n_cpmg_register.addr,
            'LOOP_ncpmg{}'.format(0))
        # pi-2 projection pulse
        self.set_pulse_registers(ch=self.cfg.mw_channel, phase=self.deg2reg(0))
        self.pulse(ch=self.cfg.mw_channel)
        self.sync_all(self.cfg.relax_delay_treg)


