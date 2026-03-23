# scripts/debug_twave3.py
"""
Verify T-wave fix produces valid windows.
"""

from src.config import CFG

def simulate_t_window(median_rr_ms, fs=100):
    """Simulate what the fixed function will produce."""
    pre  = int(CFG.beat_window_pre_ms  * fs / 1000)   # 20
    post = int(CFG.beat_window_post_ms * fs / 1000)   # 50
    signal_length = pre + post                          # 70
    r_sample      = pre                                 # 20
    MIN_T          = 10

    # HR-adjusted
    if median_rr_ms > 400:
        rr_s    = int(median_rr_ms * fs / 1000)
        t_s_hr  = r_sample + int(CFG.t_wave_start_fraction * rr_s)
        t_e_hr  = r_sample + int(CFG.t_wave_end_fraction   * rr_s)
        t_s_hr  = min(t_s_hr, signal_length - MIN_T - 1)
        t_e_hr  = min(t_e_hr, signal_length)

        if t_s_hr >= 0 and t_e_hr > t_s_hr and (t_e_hr - t_s_hr) >= MIN_T:
            return t_s_hr, t_e_hr, 'HR_ADJUSTED'

    # Fallback
    t_s = r_sample + int(CFG.t_wave_fallback_start_ms * fs / 1000)
    t_e = r_sample + int(CFG.t_wave_fallback_end_ms   * fs / 1000)
    t_s = max(0, min(t_s, signal_length - MIN_T - 1))
    t_e = max(t_s + MIN_T, min(t_e, signal_length))
    return t_s, t_e, 'FALLBACK'


print("=== T-WAVE WINDOW SIMULATION ===\n")
print(f"beat_window_pre_ms         : {CFG.beat_window_pre_ms}")
print(f"beat_window_post_ms        : {CFG.beat_window_post_ms}")
print(f"t_wave_fallback_start_ms   : {CFG.t_wave_fallback_start_ms}")
print(f"t_wave_fallback_end_ms     : {CFG.t_wave_fallback_end_ms}")

pre  = int(CFG.beat_window_pre_ms  * 100 / 1000)
post = int(CFG.beat_window_post_ms * 100 / 1000)
print(f"\nBeat array: {pre+post} samples ({pre} pre + {post} post)\n")

print(f"{'HR (BPM)':<12} {'RR (ms)':<12} {'T_start':<10} {'T_end':<10} {'Length':<10} {'Method'}")
print("-" * 64)

for hr in [45, 50, 55, 60, 70, 80, 90, 100]:
    rr = round(60000 / hr, 1)
    t_s, t_e, method = simulate_t_window(rr)
    length = t_e - t_s
    ok = "OK" if length >= 10 else "TOO SHORT"
    print(f"{hr:<12} {rr:<12} {t_s:<10} {t_e:<10} {length:<10} {method} {ok}")