"""
robustness_test1_utils.py
==========================
Self-contained loader for the IMS test1 (1st_test) run, used for OUT-OF-DISTRIBUTION
robustness evaluation of the NB4 Bayesian models (Bayesian Ridge, Gaussian Process,
MC Dropout LSTM).

This bundles the exact same feature-extraction / RUL-labeling / smoothing / Health-Index
/ scaling logic used in NB2 (2_Preprocessing_v10_kurtogram.ipynb), so test1 goes through
an IDENTICAL pipeline to test2/test3 — including per-test-run kurtogram bandpass band
selection (Section 2b) and its own scaler fit on its own healthy baseline (no leakage
from test2/test3), matching NB2's "per-test-run healthy baseline" scaler strategy.

BUG FIX (1 Sept): this file previously mirrored NB2 v8 (fixed 1000-5000 Hz band) and
was never updated when NB2 moved to v10's kurtogram-based band selection -- see the
comment above select_bandpass_band_for_test1() below for details on the mismatch this
caused in the Chapter 4.2 robustness/OOD numbers.

Difference vs NB2's own `process_test1_for_robustness()`:
    NB2's version only returns the WINDOWED sequence arrays (X_seq, y_seq) — fine for
    the LSTM, but Bayesian Ridge / GP need the FLAT (unwindowed) arrays. This version
    returns BOTH, plus the raw df, so one call feeds all 3 NB4 models.

Usage (from NB4, Step 9):
    from robustness_test1_utils import load_test1_for_robustness
    rob = load_test1_for_robustness(BASE_DIR, FEATURE_COLS, window_size=30)
    # rob['X_flat'], rob['y_flat']   -> for Bayesian Ridge / GP
    # rob['X_seq'],  rob['y_seq']    -> for MC Dropout LSTM
    # rob['df']                      -> raw dataframe w/ timestamps, RUL, features
"""
import os
import glob
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, hilbert
from scipy import signal as sp_signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Constants (must match NB2 exactly) ───────────────────────────────────────
FS       = 20_000
N_POINTS = 20_480
BPFO, BPFI, BSF, FTF = 236.4, 296.9, 139.9, 14.78
HEALTHY_N = 200

TEST1_META = {'failure_bearings': [2, 3], 'failure_modes': ['inner_race', 'roller'],
              'n_channels': 8, 'ch_map': [0, 2, 4, 6]}
TEST1_FAILURE_IDX = 2   # same convention as NB2's FAILURE_IDX['test1']


# ── Bandpass band selection via spectral kurtosis (ported from NB2 Section 2b)
#
# BUG FIX (1 Sept): this file was written against NB2 v8, which used a fixed
# 1000-5000 Hz band everywhere. NB2 was since upgraded to v10 (kurtogram-based
# per-test-run band selection -- see select_bandpass_band_for_test1() below),
# but this file was never updated to match, so test1 robustness/OOD features
# were still being extracted in the OLD fixed band while test2/test3
# (train/test) features used the NEW kurtogram-selected band -- a mismatch
# between what the models were trained on and what they were evaluated
# against on test1. That directly affects the OOD robustness numbers (PICP
# etc.) reported in Chapter 4.2. Ported the same kurtogram logic here so
# test1 goes through an identical band-selection methodology.
def compute_spectral_kurtosis(sig, fs=FS, nperseg=256, noverlap=None):
    """Spectral Kurtosis (SK) via STFT -- simplified single-level kurtogram.
    Reference: Antoni, J. & Randall, R.B. (2006)."""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = sp_signal.stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    sk = kurtosis(mag, axis=1, fisher=True)
    return f, sk


def find_optimal_band(sig, fs=FS, nperseg=256, bandwidth=None,
                       min_bandwidth=300, max_bandwidth=3000,
                       search_range=(300, None)):
    """Scans for the sub-band with highest spectral kurtosis (the resonance
    band most excited by fault-induced impacts). Identical to NB2's version."""
    lo_search, hi_search = search_range
    if hi_search is None:
        hi_search = fs / 2 - 1
    f, sk = compute_spectral_kurtosis(sig, fs=fs, nperseg=nperseg)
    mask = (f >= lo_search) & (f <= hi_search)
    f_valid, sk_valid = f[mask], sk[mask]
    best_idx = np.argmax(sk_valid)
    center_freq = f_valid[best_idx]
    peak_sk = sk_valid[best_idx]

    if bandwidth is None:
        half_max = peak_sk / 2
        lo_idx = best_idx
        while lo_idx > 0 and sk_valid[lo_idx - 1] >= half_max:
            lo_idx -= 1
        hi_idx = best_idx
        while hi_idx < len(sk_valid) - 1 and sk_valid[hi_idx + 1] >= half_max:
            hi_idx += 1
        bandwidth = f_valid[hi_idx] - f_valid[lo_idx]
        bandwidth = np.clip(bandwidth, min_bandwidth, max_bandwidth)

    lowcut = max(center_freq - bandwidth / 2, 1)
    highcut = min(center_freq + bandwidth / 2, fs / 2 - 1)
    return lowcut, highcut, center_freq, peak_sk


def select_bandpass_band_for_test1(test1_dir, ch_map=None, failure_idx=None, fs=FS):
    """Selects the bandpass range for test1, on a near-failure snapshot (second-
    to-last file) of its own failing-bearing channel -- same convention NB2 uses
    for test2/test3 (select_bandpass_band_for_test)."""
    ch_map = TEST1_META['ch_map'] if ch_map is None else ch_map
    failure_idx = TEST1_FAILURE_IDX if failure_idx is None else failure_idx
    files = sorted(glob.glob(os.path.join(test1_dir, '*.*.*.*.*.*')))
    if not files:
        files = sorted(glob.glob(os.path.join(test1_dir, '*_*_*_*_*_*')))
    near_failure_file = files[-2]
    data = pd.read_csv(near_failure_file, sep=r'\s+', header=None).values
    sig = data[:, ch_map[failure_idx]]
    lowcut, highcut, center, peak_sk = find_optimal_band(sig, fs=fs)
    print(f'  [test1] bandpass range (kurtogram, failing bearing, near-failure '
          f'snapshot): {lowcut:.0f}-{highcut:.0f} Hz (center={center:.0f} Hz, peak SK={peak_sk:.2f})')
    return lowcut, highcut


# ── Feature extraction (identical to NB2) ────────────────────────────────────
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


def extract_failing_bearing_features(data, ch_map, failure_idx, lowcut=1000, highcut=5000):
    sig = data[:, ch_map[failure_idx]]
    feats = {}
    feats.update(extract_time_features(sig))
    feats.update(extract_frequency_features(sig, lowcut=lowcut, highcut=highcut))
    return feats


def parse_filename_to_datetime(filename):
    for fmt in ('%Y.%m.%d.%H.%M.%S', '%Y_%m_%d_%H_%M_%S'):
        try:
            return datetime.strptime(filename, fmt)
        except ValueError:
            continue
    return None


def _process_single_file(filepath, ch_map, n_channels, failure_idx, lowcut=1000, highcut=5000):
    path = Path(filepath)
    ts = parse_filename_to_datetime(path.name)
    if ts is None:
        return None
    try:
        data = pd.read_csv(path, sep=r'\s+', header=None, usecols=range(n_channels)).values
        if data.shape[0] != N_POINTS:
            return None
        row = {'timestamp': ts, 'filename': path.name, 'test_run': 'test1'}
        row.update(extract_failing_bearing_features(data, ch_map, failure_idx, lowcut, highcut))
        return row
    except Exception as e:
        print(f'  [skip] {path.name}: {e}')
        return None


def build_test1_feature_dataset(test1_dir, lowcut=1000, highcut=5000, max_workers=8):
    files = sorted(glob.glob(os.path.join(test1_dir, '*.*.*.*.*.*')))
    if not files:
        files = sorted(glob.glob(os.path.join(test1_dir, '*_*_*_*_*_*')))
    print(f'  [test1] {len(files)} files | {TEST1_META["n_channels"]} channels | '
          f'ch_map={TEST1_META["ch_map"]}')

    records = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_file, f, TEST1_META['ch_map'],
                                    TEST1_META['n_channels'], TEST1_FAILURE_IDX,
                                    lowcut, highcut): f
                   for f in files}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res:
                records.append(res)
            if (i + 1) % 200 == 0:
                print(f'    {i + 1}/{len(files)} done...')

    df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
    dur = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    print(f'  [test1] ✓ {len(df)} valid samples | {dur:.1f} hours')
    return df


# ── RUL labeling (identical to NB2) ──────────────────────────────────────────
def compute_fpt(df, bearing_idx=0, sigma_multiplier=3.0, window=10):
    col = 'kurtosis' if 'kurtosis' in df.columns else f'kurtosis_{bearing_idx}'
    baseline = df[col].iloc[:HEALTHY_N]
    threshold = baseline.mean() + sigma_multiplier * baseline.std()
    rolling = df[col].rolling(window=window, center=False).mean()
    exceed = rolling[rolling > threshold]
    if exceed.empty:
        print('  [FPT] No exceedance found — using start')
        return df['timestamp'].iloc[0], 0
    fpt_idx = exceed.index[0]
    return df['timestamp'].iloc[fpt_idx], fpt_idx


def add_rul_labels(df, test_name='test1', failure_bearing_idx=TEST1_FAILURE_IDX):
    t_end = df['timestamp'].max()
    t_start = df['timestamp'].min()
    fpt_ts, fpt_idx = compute_fpt(df, bearing_idx=failure_bearing_idx)
    T_degradation = (t_end - fpt_ts).total_seconds() / 60
    MAX_RUL = round(T_degradation, 0)

    df['rul_linear'] = df['timestamp'].apply(
        lambda t: min((t_end - t).total_seconds() / 60, MAX_RUL))

    def piecewise_rul(row):
        t = row['timestamp']
        if t <= fpt_ts:
            return float(MAX_RUL)
        return min((t_end - t).total_seconds() / 60, MAX_RUL)

    df['rul_piecewise'] = df.apply(piecewise_rul, axis=1)
    df['rul_normalized'] = df['rul_piecewise'] / MAX_RUL
    df['max_rul'] = MAX_RUL

    total_h = (t_end - t_start).total_seconds() / 3600
    print(f'  [{test_name}] Total={total_h:.1f}h | '
          f'FPT idx={fpt_idx} ({(fpt_ts - t_start).total_seconds() / 3600:.1f}h) | '
          f'MAX_RUL={MAX_RUL:.0f} min (data-driven)')
    return df, fpt_ts, MAX_RUL


# ── Smoothing + Health Index (identical to NB2) ─────────────────────────────
def apply_hampel_filter(series, window_size=5, n_sigmas=3):
    rolling_median = series.rolling(window=window_size, center=True).median()
    rolling_mad = (1.4826 * (series - rolling_median).abs()
                   .rolling(window=window_size, center=True).median())
    upper = rolling_median + n_sigmas * rolling_mad
    lower = rolling_median - n_sigmas * rolling_mad
    filtered = series.copy()
    mask = (series > upper) | (series < lower)
    filtered[mask] = rolling_median[mask]
    return filtered.bfill().ffill()


def construct_health_index(df, healthy_n=HEALTHY_N):
    feature_cols = ['rms', 'kurtosis', 'bpfo_amp', 'env_kurtosis', 'crest_factor']
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].values
    X_healthy = X[:healthy_n]

    scaler = StandardScaler().fit(X_healthy)
    pca = PCA(n_components=1).fit(scaler.transform(X_healthy))
    hi = pca.transform(scaler.transform(X)).flatten()

    if hi[-healthy_n:].mean() < hi[:healthy_n].mean():
        hi = -hi

    mu, sd = hi[:healthy_n].mean(), hi[:healthy_n].std() + 1e-12
    df['hi'] = (hi - mu) / sd
    return df


def fit_scaler_on_healthy(df, feature_cols, healthy_n=HEALTHY_N):
    sc = StandardScaler()
    sc.fit(df[feature_cols].iloc[:healthy_n].values)
    return sc


# ── Sequencing (identical to NB2) ────────────────────────────────────────────
def create_sequences(X, y, window_size=30, stride=1):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - window_size + 1, stride):
        X_seq.append(X[i: i + window_size])
        y_seq.append(y[i + window_size - 1])
    if not X_seq:
        return (np.array([]).reshape(0, window_size, X.shape[1] if len(X) else 0),
                np.array([]))
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


# ── Main entry point ─────────────────────────────────────────────────────────
def load_test1_for_robustness(test1_dir, feature_cols, window_size=30, healthy_n=HEALTHY_N,
                               lowcut=None, highcut=None):
    """
    Full test1 pipeline, mirroring NB2 exactly, returning BOTH flat and sequence
    arrays (NB2's own process_test1_for_robustness() only returns sequences).

    lowcut, highcut: bandpass range (Hz) for frequency-domain features. If not
    given, selected automatically via kurtogram on a near-failure snapshot of
    test1's own failing-bearing channel (select_bandpass_band_for_test1) --
    the same per-test-run methodology NB2 uses for test2/test3. Pass explicit
    values instead if you already have them (e.g. re-running against a cached
    band) to skip the extra file read.

    Returns a dict:
        X_flat, y_flat   -> (N, n_features)          for Bayesian Ridge / GP
        X_seq,  y_seq    -> (N', window_size, n_feat) for MC Dropout LSTM
        df               -> full smoothed dataframe (timestamps, RUL, features, hi)
        scaler           -> StandardScaler fit on test1's own healthy baseline
        fpt_ts, max_rul  -> for reference / plotting
        lowcut, highcut  -> the band actually used (log this / save it alongside
                             your other results so the robustness numbers are reproducible)
    """
    print('=== Loading test1 for robustness testing (flat + sequence) ===')
    if not os.path.exists(test1_dir):
        raise FileNotFoundError(f'test1 not found at {test1_dir}')

    if lowcut is None or highcut is None:
        lowcut, highcut = select_bandpass_band_for_test1(test1_dir)

    df_r = build_test1_feature_dataset(test1_dir, lowcut=lowcut, highcut=highcut)
    df_r, fpt_ts, max_rul = add_rul_labels(df_r)

    # Smooth (Hampel + EWMA), same as NB2 dfs_smooth step
    rul_cols  = ['rul_linear', 'rul_piecewise', 'rul_normalized', 'max_rul']
    meta_cols = ['timestamp', 'filename', 'test_run'] + rul_cols
    df_s = df_r.copy()
    for col in [c for c in df_s.columns if c not in meta_cols]:
        df_s[col] = apply_hampel_filter(df_s[col])
        df_s[col] = df_s[col].ewm(alpha=0.05, adjust=False).mean()

    # Health Index (failing bearing, oriented)
    df_s = construct_health_index(df_s)

    # Scale with test1's OWN healthy-baseline scaler (no leakage from test2/3)
    sc1 = fit_scaler_on_healthy(df_s, feature_cols, healthy_n=healthy_n)
    X_flat = sc1.transform(df_s[feature_cols].values)
    y_flat = df_s['rul_normalized'].values

    X_seq, y_seq = create_sequences(X_flat, y_flat, window_size=window_size)

    print(f'  test1 flat : X{X_flat.shape}  y{y_flat.shape}')
    print(f'  test1 seq  : X{X_seq.shape}  y{y_seq.shape}  (window_size={window_size})')

    return {
        'X_flat': X_flat, 'y_flat': y_flat,
        'X_seq': X_seq, 'y_seq': y_seq,
        'df': df_s, 'scaler': sc1,
        'fpt_ts': fpt_ts, 'max_rul': max_rul,
        'lowcut': lowcut, 'highcut': highcut,
    }
