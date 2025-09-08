from qickdawg.nvpulsing.cpmgxy_sweep_outside import CPMGXY8nDelaySweepOutside
import qickdawg as qd
from copy import copy


def pulse_cpmg(delay):

    default_config = qd.NVConfiguration()

    default_config.adc_channel = 0
    default_config.edge_counting = True
    default_config.high_threshold = 2000
    default_config.low_threshold = 500


    default_config.mw_channel = 0
    default_config.mw_nqz = 1
    default_config.mw_gain = 5000

    default_config.laser_gate_pmod = 0

    default_config.relax_delay_tns = 50 # between each rep, wait for everything to catch up, mostly aom

    soc = qd.soc
    config = copy(default_config)

    config.mw_gain = 30000
    config.mw_fMHz = 500
    config.mw_pi2_tns = 100

    config.relax_delay_tus = 0.5
    config.delay_tns = delay

    config.reps=1
    config.n_cpmg = 1
    prog = CPMGXY8nDelaySweepOutside(config)
    prog.run_rounds(soc, rounds=0, start_src="external")