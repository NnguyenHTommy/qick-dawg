'''
Hermite-pulse fine timing suite -- fully self-contained file set.

Every experiment file carries its own hermite_envelope() definition and
imports only qickdawg.nvpulsing.* and the shared
qickdawg.finetimingsuite.readout_helpers, so individual files can also be
dropped flat into finetimingsuite/ if preferred.

Install as a package: copy this folder into your qickdawg package as
    qickdawg/finetimingsuite/hermite/
so the notebook imports
    from qickdawg.finetimingsuite.hermite.podmr_fine_res import PODMRFineRes
resolve.

Contents:
    podmr_fine_res.py      -- PODMRFineRes (frequency sweep, hermite pulse)
    rabi_fine_res.py       -- RabiFineRes + sweep_rabi_duration()
                              (Python-side shape-faithful duration sweep)
    rabi_fine_res_amp.py   -- RabiGainFineRes (hardware gain sweep,
                              built from the CORRECTED square version)
    cpmg_xy_fine_res.py    -- CPMGXYFineRes (n_cpmg=0 Ramsey, 1 Hahn, N CPMG-XY8)
    ramsey_fine_res.py     -- RamseyFineRes (standalone; latent square bugs fixed)
    hahn_echo_fine_res.py  -- HahnEchoFineRes (standalone; latent square bug fixed)
'''
