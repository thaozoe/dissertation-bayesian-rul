"""
realtime_utils.py
==================
Support code for NB5 (Real-Time RUL Prediction). Generalizes the single-file
feature extraction used throughout NB2/NB4 (previously test1-only, in
robustness_test1_utils.py) so it works for ANY test run, and adds a
`RealTimeRULPredictor` that simulates a live deployment: snapshots arrive one
at a time, features are extracted on the fly, a rolling window is maintained
for the LSTM, and all 3 trained models produce a prediction + uncertainty
with wall-clock latency measured per snapshot.

Usage (see NB5):
    from realtime_utils import extract_features_from_file, RealTimeRULPredictor

    predictor = RealTimeRULPredictor(
        feature_cols=FEATURE_COLS, window_size=WINDOW_SIZE, max_rul=MAX_RUL_TEST3,
        flat_scaler=flat_scaler_test3,
        bridge_model=bridge_model,
        gp_model=gp_model, gp_scaler=gp_scaler,
        lstm_model=lstm_model, seq_scaler=seq_scaler, device=device,
        lowcut=lowcut, highcut=highcut,   # MUST match metadata['bandpass_band_per_test'][test_name] (NB2)
        n_mc_samples=50,
    )
    record = predictor.ingest(filepath, ch_map, n_channels, failure_idx)
"""
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, kurtosis
from scipy.signal import butter, hilbert
from scipy import signal as sp_signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HI_COLS = ('rms', 'kurtosis', 'bpfo_amp', 'env_kurtosis', 'crest_factor')

# ── Constants (must match NB2) ───────────────────────────────────────────────
FS       = 20_000
N_POINTS = 20_480
BPFO, BPFI, BSF, FTF = 236.4, 296.9, 139.9, 14.78


# ── Feature extraction (identical logic to NB2, generalized to any test) ────
#
# BUG FIX (1 Sept): NB2 was updated to select the bandpass range per test run
# via spectral kurtosis (kurtogram) instead of a fixed band -- see NB2 Section
# 2b, `select_bandpass_band_for_test()`. That band is saved to
# metadata.pkl['bandpass_band_per_test'][test_name] and is what NB3/NB4's
# training features were computed with. This file was never updated to match:
# every function below silently fell back to the OLD fixed 1000-5000 Hz
# default, so real-time features were computed in the wrong band -- mismatched
# against what the models were trained on. lowcut/highcut are now threaded
# through every call in this chain; the 1000/5000 defaults are kept only as a
# fallback for standalone/ad-hoc use, NOT for real deployment via
# RealTimeRULPredictor, which now requires them explicitly (see below).
def bandpass_filter(data, lowcut=1000, highcut=5000, fs=FS, order=4):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sp_signal.sosfilt(sos, data)


def get_envelope_spectrum(sig, lowcut=1000, highcut=5000, fs=FS):
    filtered  = bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=fs)
    envelope  = np.abs(hilbert(filtered))
    envelope -= envelope.mean()
    n   = len(envelope)
    win = np.hanning(n)
    yf  = np.fft.fft(envelope * win)
    xf  = np.fft.fftfreq(n, 1 / fs)[:n // 2]
    amp = 2.0 / n * np.abs(yf[:n // 2])
    return xf, amp


def extract_time_features(sig):
    rms      = np.sqrt(np.mean(sig ** 2))
    peak     = np.max(np.abs(sig))
    mean_abs = np.mean(np.abs(sig))
    return {
        'rms': rms, 'peak': peak, 'kurtosis': kurtosis(sig), 'skew': skew(sig),
        'crest_factor': peak / (rms + 1e-12), 'std': np.std(sig),
        'shape_factor': rms / (mean_abs + 1e-12),
        'impulse_factor': peak / (mean_abs + 1e-12),
    }


def extract_frequency_features(sig, lowcut=1000, highcut=5000, fs=FS):
    xf, amp = get_envelope_spectrum(sig, lowcut=lowcut, highcut=highcut, fs=fs)
    env = np.abs(hilbert(bandpass_filter(sig, lowcut=lowcut, highcut=highcut, fs=fs)))
    return {
        'bpfo_amp': amp[(np.abs(xf - BPFO)).argmin()],
        'bpfi_amp': amp[(np.abs(xf - BPFI)).argmin()],
        'bsf_amp':  amp[(np.abs(xf - BSF)).argmin()],
        'ftf_amp':  amp[(np.abs(xf - FTF)).argmin()],
        'spectral_entropy': -np.sum((amp / (amp.sum() + 1e-12)) *
                             np.log2(amp / (amp.sum() + 1e-12) + 1e-12)),
        'env_kurtosis': kurtosis(env),
    }


def parse_filename_to_datetime(filename):
    for fmt in ('%Y.%m.%d.%H.%M.%S', '%Y_%m_%d_%H_%M_%S'):
        try:
            return datetime.strptime(filename, fmt)
        except ValueError:
            continue
    return None


def extract_features_from_file(filepath, ch_map, n_channels, failure_idx,
                                lowcut=1000, highcut=5000):
    """
    Parse ONE raw snapshot file (the unit a real sensor system would deliver
    every ~10 min) into the 14 failing-bearing features NB2 uses everywhere
    else. Returns (timestamp, feature_dict) or (None, None) if unreadable.

    lowcut, highcut: bandpass range (Hz) for the frequency-domain features.
    MUST be the same band NB2 selected for this test run via the kurtogram
    (metadata.pkl['bandpass_band_per_test'][test_name]) -- see module
    docstring / RealTimeRULPredictor.
    """
    path = Path(filepath)
    ts = parse_filename_to_datetime(path.name)
    if ts is None:
        return None, None
    try:
        data = pd.read_csv(path, sep=r'\s+', header=None, usecols=range(n_channels)).values
        if data.shape[0] != N_POINTS:
            return None, None
        sig = data[:, ch_map[failure_idx]]
        feats = {}
        feats.update(extract_time_features(sig))
        feats.update(extract_frequency_features(sig, lowcut=lowcut, highcut=highcut))
        return ts, feats
    except Exception as e:
        print(f'  [skip] {path.name}: {e}')
        return None, None


# ── Causal smoothing (streaming-safe approximation of NB2's Hampel+EWMA) ────
class CausalSmoother:
    """
    NB2 applies Hampel(window=5, center=True) then EWMA(alpha=0.05) to the FULL
    offline dataframe. A live system only has the past, so:
      - Hampel: use a TRAILING window of the same size instead of centered.
        This shifts detection by ~2 samples (~20 min) but is otherwise identical.
      - EWMA: already a purely causal recursive filter (y_t = a*x_t + (1-a)*y_{t-1}),
        so no approximation needed there -- just carry the previous state forward.
    """
    def __init__(self, feature_names, hampel_window=5, n_sigmas=3, ewma_alpha=0.05):
        self.feature_names = list(feature_names)
        self.window    = hampel_window
        self.n_sigmas  = n_sigmas
        self.alpha     = ewma_alpha
        self.raw_hist  = {f: deque(maxlen=hampel_window) for f in self.feature_names}
        self.ewma_prev = {f: None for f in self.feature_names}

    def update(self, raw_feats):
        """raw_feats: dict {feature_name: value} for ONE new snapshot. Returns the
        smoothed dict (Hampel-cleaned + EWMA), same keys."""
        smoothed = {}
        for f in self.feature_names:
            self.raw_hist[f].append(raw_feats[f])
            buf = np.array(self.raw_hist[f])
            median = np.median(buf)
            mad = 1.4826 * np.median(np.abs(buf - median))
            x = raw_feats[f]
            x_clean = median if (mad > 0 and abs(x - median) > self.n_sigmas * mad) else x

            prev = self.ewma_prev[f]
            sm = x_clean if prev is None else self.alpha * x_clean + (1 - self.alpha) * prev
            self.ewma_prev[f] = sm
            smoothed[f] = sm
        return smoothed


# ── Health Index: fit offline once, apply per-snapshot online ──────────────
def fit_health_index_transform(df_smoothed_healthy_region, healthy_n, hi_cols=HI_COLS):
    """
    Mirrors NB2's construct_health_index() exactly, but returns the FITTED
    parameters instead of a dataframe, so the same transform can be applied to
    one new smoothed feature vector at a time in real-time.

    df_smoothed_healthy_region must be the SAME smoothed dataframe NB2 built
    (e.g. df_test.parquet) so the fitted scaler/PCA/sign/mu/sd are identical to
    what NB2 used -- this is an offline fit (needs the historical run), done
    ONCE at predictor start-up, exactly like fitting flat_scaler on the
    healthy baseline.
    """
    cols = [c for c in hi_cols if c in df_smoothed_healthy_region.columns]
    X = df_smoothed_healthy_region[cols].values
    X_healthy = X[:healthy_n]

    scaler = StandardScaler().fit(X_healthy)
    pca    = PCA(n_components=1).fit(scaler.transform(X_healthy))
    hi_all = pca.transform(scaler.transform(X)).flatten()

    sign = -1.0 if hi_all[-healthy_n:].mean() < hi_all[:healthy_n].mean() else 1.0
    hi_signed = hi_all * sign
    mu, sd = hi_signed[:healthy_n].mean(), hi_signed[:healthy_n].std() + 1e-12

    return {'cols': cols, 'scaler': scaler, 'pca': pca, 'sign': sign, 'mu': mu, 'sd': sd}


def transform_health_index(smoothed_feats, hi_params):
    """Apply the pre-fitted HI transform to ONE new smoothed feature dict."""
    x = np.array([[smoothed_feats[c] for c in hi_params['cols']]])
    hi = hi_params['pca'].transform(hi_params['scaler'].transform(x))[0, 0] * hi_params['sign']
    return (hi - hi_params['mu']) / hi_params['sd']


# ── Maintenance decision logic ───────────────────────────────────────────────
def maintenance_status(rul_lower_min, critical_min=60.0, warning_min=180.0):
    """
    Conservative decision uses the LOWER bound of the uncertainty interval
    (not the mean) -- i.e. "how bad could this be", not "what's most likely".
    That's the point of having calibrated uncertainty at all: a wide interval
    with a low mean should trigger action even if the mean alone looks okay.
    """
    if rul_lower_min < critical_min:
        return 'CRITICAL — schedule maintenance now'
    elif rul_lower_min < warning_min:
        return 'WARNING — monitor closely, plan maintenance window'
    return 'NORMAL — no action'


# ── Real-time predictor ───────────────────────────────────────────────────────
class RealTimeRULPredictor:
    def __init__(self, feature_cols, window_size, max_rul, flat_scaler,
                 bridge_model, gp_model, gp_scaler,
                 lstm_model, seq_scaler, device,
                 df_for_hi_fit, healthy_n,
                 lowcut, highcut,
                 n_mc_samples=50, critical_min=60.0, warning_min=180.0):
        """
        df_for_hi_fit: the offline smoothed dataframe (e.g. df_test.parquet) used
        to fit the Health Index transform ONCE at start-up -- exactly like
        flat_scaler is fit once on the healthy baseline. Needed because 'hi' is a
        derived feature (PCA over history) that a single incoming snapshot alone
        cannot produce; see fit_health_index_transform().

        lowcut, highcut: bandpass range (Hz) used for every frequency-domain
        feature extracted during streaming. REQUIRED (no default) -- pass
        metadata.pkl['bandpass_band_per_test'][test_name] from NB2's kurtogram
        band selection. Passing the wrong band here silently produces features
        that don't match what the models were trained on (this was the 1 Sept
        bug: this class used to hardcode the old fixed 1000-5000 Hz band).
        """
        self.feature_cols = feature_cols
        self.window_size  = window_size
        self.max_rul      = max_rul
        self.flat_scaler  = flat_scaler
        self.bridge_model = bridge_model
        self.gp_model     = gp_model
        self.gp_scaler    = gp_scaler
        self.lstm_model   = lstm_model
        self.seq_scaler   = seq_scaler
        self.device       = device
        self.lowcut       = lowcut
        self.highcut      = highcut
        self.n_mc_samples = n_mc_samples
        self.critical_min = critical_min
        self.warning_min  = warning_min

        raw_feature_names = [c for c in feature_cols if c != 'hi']
        self.smoother  = CausalSmoother(raw_feature_names)
        self.hi_params = (fit_health_index_transform(df_for_hi_fit, healthy_n)
                           if 'hi' in feature_cols else None)

        self.buffer  = deque(maxlen=window_size)   # scaled flat feature vectors, for LSTM
        self.history = []

    def _predict_lstm(self):
        if len(self.buffer) < self.window_size:
            return None, None   # still warming up -- not enough history for a window yet
        X_seq = np.stack(self.buffer, axis=0)[None, ...]              # (1, window, n_feat)
        X_seq_scaled = self.seq_scaler.transform(
            X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape).astype(np.float32)
        Xt = torch.tensor(X_seq_scaled, dtype=torch.float32).to(self.device)

        self.lstm_model.train()   # keep dropout active (MC Dropout)
        preds = np.zeros(self.n_mc_samples, dtype=np.float32)
        with torch.no_grad():
            for i in range(self.n_mc_samples):
                preds[i] = self.lstm_model(Xt).cpu().numpy().ravel()[0]
        return float(preds.mean()), float(preds.std())

    def ingest(self, filepath, ch_map, n_channels, failure_idx, y_true_norm=None):
        """Process ONE incoming snapshot file end-to-end. Returns a result dict."""
        t0 = time.perf_counter()

        ts, raw_feats = extract_features_from_file(
            filepath, ch_map, n_channels, failure_idx,
            lowcut=self.lowcut, highcut=self.highcut)
        if raw_feats is None:
            return None

        # Causal Hampel+EWMA smoothing (streaming-safe -- see CausalSmoother)
        feats = self.smoother.update(raw_feats)
        # Health Index is a derived feature (PCA fit offline); compute it from
        # the just-smoothed values using the pre-fitted transform
        if self.hi_params is not None:
            feats = dict(feats)
            feats['hi'] = transform_health_index(feats, self.hi_params)

        x_raw    = np.array([feats[c] for c in self.feature_cols]).reshape(1, -1)
        x_scaled = self.flat_scaler.transform(x_raw)
        self.buffer.append(x_scaled[0])

        bridge_mu, bridge_sigma = self.bridge_model.predict(x_scaled, return_std=True)
        x_scaled_gp = self.gp_scaler.transform(x_scaled)
        gp_mu, gp_sigma = self.gp_model.predict(x_scaled_gp, return_std=True)
        lstm_mu, lstm_sigma = self._predict_lstm()

        latency_ms = (time.perf_counter() - t0) * 1000

        gp_mu_c    = float(np.clip(gp_mu[0], 0, 1))
        gp_lower   = max(gp_mu_c - 1.645 * float(gp_sigma[0]), 0.0)   # 90% one-sided lower bound
        status     = maintenance_status(gp_lower * self.max_rul, self.critical_min, self.warning_min)

        record = {
            'timestamp': ts,
            'y_true_norm': y_true_norm,
            'bridge_mu': float(np.clip(bridge_mu[0], 0, 1)), 'bridge_sigma': float(bridge_sigma[0]),
            'gp_mu': gp_mu_c, 'gp_sigma': float(gp_sigma[0]),
            'gp_rul_min': gp_mu_c * self.max_rul, 'gp_lower_rul_min': gp_lower * self.max_rul,
            'lstm_mu': lstm_mu, 'lstm_sigma': lstm_sigma,
            'status': status,
            'latency_ms': latency_ms,
        }
        self.history.append(record)
        return record

    def history_df(self):
        return pd.DataFrame(self.history)
