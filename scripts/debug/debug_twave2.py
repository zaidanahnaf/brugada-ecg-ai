# scripts/debug_twave2.py
"""
Diagnose exactly why T-wave window = [69, 70]
"""

from src.config.__init__ import CFG

fs = 100
median_rr_ms = 1125.0
r_sample = 20
signal_length = 70  # beat array length

rr_samples = int(median_rr_ms * fs / 1000)  # = 112 samples
t_start_rr = r_sample + int(CFG.preprocessing.t_wave_start_fraction * rr_samples)
t_end_rr   = r_sample + int(CFG.preprocessing.t_wave_end_fraction   * rr_samples)

print(f"CFG.preprocessing.beat_window_pre_ms   : {CFG.preprocessing.beat_window_pre_ms}")
print(f"CFG.preprocessing.beat_window_post_ms  : {CFG.preprocessing.beat_window_post_ms}")
print(f"CFG.preprocessing.t_wave_start_fraction: {CFG.preprocessing.t_wave_start_fraction}")
print(f"CFG.preprocessing.t_wave_end_fraction  : {CFG.preprocessing.t_wave_end_fraction}")
print(f"CFG.preprocessing.t_wave_fallback_start_ms: {CFG.preprocessing.t_wave_fallback_start_ms}")
print(f"CFG.preprocessing.t_wave_fallback_end_ms  : {CFG.preprocessing.t_wave_fallback_end_ms}")
print()
print(f"median_rr_ms  = {median_rr_ms}")
print(f"rr_samples    = {rr_samples}")
print(f"r_sample      = {r_sample}")
print(f"signal_length = {signal_length}")
print()
print(f"HR-adjusted T_start = {t_start_rr}  (ms: {(t_start_rr-r_sample)*10})")
print(f"HR-adjusted T_end   = {t_end_rr}    (ms: {(t_end_rr-r_sample)*10})")
print(f"After clamping to [0, {signal_length}]:")
print(f"  t_start = {max(0, min(t_start_rr, signal_length-1))}")
print(f"  t_end   = {max(max(0, min(t_start_rr, signal_length-1))+1, min(t_end_rr, signal_length))}")
print()

# Fallback
t_fb_start = r_sample + int(CFG.preprocessing.t_wave_fallback_start_ms * fs / 1000)
t_fb_end   = r_sample + int(CFG.preprocessing.t_wave_fallback_end_ms   * fs / 1000)
print(f"Fallback T_start = {t_fb_start}  (ms: {(t_fb_start-r_sample)*10})")
print(f"Fallback T_end   = {t_fb_end}    (ms: {(t_fb_end-r_sample)*10})")
print(f"Fallback fits in beat window ({signal_length} samples)? {t_fb_end <= signal_length}")