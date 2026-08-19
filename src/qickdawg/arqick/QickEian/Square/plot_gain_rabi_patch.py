'''
plot_gain_rabi -- Visualizer method for amplitude (gain) Rabi sweeps
=======================================================================
PASTE the method below into the Visualizer class in
src/qickdawg/finetimingsuite/visualization.py (same level as plot_rabi).

It follows the same spec/plot_experiment pattern as every other plot_*
method. Differences from plot_rabi:
  - x_key is 'mw_gain' (DAC units), not 'mw_duration_ftns'
  - fit model is a cosine in GAIN: at fixed pulse duration, rotation angle
    scales linearly with drive amplitude, so signal ~ A*cos(2*pi*x/G) + C
    where G is the gain per full rotation. The pi-pulse gain is G/2.
  - no exponential decay term: decoherence depends on elapsed time, which
    is constant here (fixed duration), unlike a duration sweep.

With get_reference=False, only the 'MW On' traces exist; plot_experiment
already warn-skips missing traces, and view='raw' + the fit (which uses
'MW On - Signal') work fine. view='contrast' requires get_reference=True.
'''

# ---- paste into Visualizer (visualization.py) from here ----

    @staticmethod
    def plot_gain_rabi(data, cfg=None, fit=True, contrast_mode="signal_over_off", view="raw"):
        """Plot amplitude (gain) Rabi data with optional cosine fit.

        Parameters
        ----------
            data : object
                Data object containing signal1_cts_s (and signal2_cts_s etc.
                if acquired with get_reference=True) and mw_gain.
            cfg : object, optional
                Configuration object used for annotations.
            fit : bool, default True
                Whether to fit a cosine (in gain) to the MW On signal.
            contrast_mode : str, default "signal_over_off"
                Contrast method if view="contrast" (requires get_reference=True).
            view : str, default "raw"
                "raw" or "contrast".
        """

        def gain_rabi_model(x, A, G, C):
            # Rotation angle is linear in gain at fixed duration:
            # signal ~ A * cos(2*pi * x / G) + C ; pi-pulse gain = G/2
            return A * np.cos(2 * np.pi * x / G) + C

        def gain_rabi_initial_guess(x, y):
            A0 = (np.max(y) - np.min(y)) / 2
            C0 = float(np.mean(y))
            dx = max(float(np.median(np.diff(x))), 1e-12)
            freqs = np.fft.rfftfreq(len(x), d=dx)
            spectrum = np.abs(np.fft.rfft(y - np.mean(y)))
            if len(spectrum) > 1 and np.argmax(spectrum[1:]) >= 0:
                f0 = float(freqs[1 + np.argmax(spectrum[1:])])
            else:
                f0 = 0.0
            # If no clear oscillation is resolved, assume roughly a half
            # period across the sweep span (monotonic decrease).
            G0 = 1.0 / f0 if f0 > 0 else 2.0 * float(np.max(x) - np.min(x))
            return [A0, G0, C0]

        def format_gain_rabi_params(p):
            A, G, C = p
            return [
                f"A = {A:.2f}",
                rf"$\pi$-pulse gain = {G/2:.0f} DAC",
                f"G (full period) = {G:.0f} DAC",
                f"C = {C:.2f}",
            ]

        # plot_experiment computes contrast eagerly whenever contrast_mode is
        # set (even for view="raw"), and that needs the MW Off traces -- which
        # only exist when the data was acquired with get_reference=True.
        # Detect their absence and drop contrast_mode so raw view still works.
        if not hasattr(data, "signal2_cts_s"):
            contrast_mode = None
            if view == "contrast":
                raise ValueError(
                    "view='contrast' needs MW Off traces: re-acquire with get_reference=True")

        GAIN_RABI_SPEC = {
            "name": "Amplitude (Gain) Rabi",
            "x_key": "mw_gain",
            "x_label": "MW Gain (DAC units)",
            "contrast_mode": contrast_mode,
            "traces": Visualizer.DATA_TRACES,
            "fit": {
                "model": gain_rabi_model,
                "trace": "MW On - Signal",
                "initial_guess": gain_rabi_initial_guess,
                "format_params": format_gain_rabi_params,
                "label": "Gain Rabi",
                "equation": r"$y = A \cos(2\pi x / G) + C$",
            }
        }

        Visualizer.plot_experiment(data, GAIN_RABI_SPEC, cfg=cfg, fit=fit, view=view)

# ---- end paste ----
