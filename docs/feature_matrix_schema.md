SCHEMA: features/feature_matrix.csv

Row = one subject (patient_id is unique)
Column definitions below.

MANDATORY COLUMNS (always present):
┌──────────────────────────┬──────────┬─────────────────────────────────────┐
│ Column Name │ Type │ Description │
├──────────────────────────┼──────────┼─────────────────────────────────────┤
│ patient_id │ string │ Unique subject identifier │
│ │ │ Must match metadata.csv exactly │
├──────────────────────────┼──────────┼─────────────────────────────────────┤
│ brugada │ int │ Target label: 0=Normal, 1=Brugada │
│ │ │ INCLUDED for convenience ONLY │
│ │ │ Must NEVER be used as model input │
├──────────────────────────┼──────────┼─────────────────────────────────────┤
│ pipeline_status │ string │ 'OK' | 'PARTIAL' | 'FAILED' │
│ │ │ Exclude FAILED rows before ML │
├──────────────────────────┼──────────┼─────────────────────────────────────┤
│ n_valid_beats │ int │ Number of clean beats used │
│ │ │ Quality indicator — not ML feature │
└──────────────────────────┴──────────┴─────────────────────────────────────┘

FEATURE COLUMNS (see feature*manifest.json for full definitions):
┌──────────────────────────────────────┬─────────┬──────────────────────────┐
│ Pattern │ Count │ Example │
├──────────────────────────────────────┼─────────┼──────────────────────────┤
│ sq*{lead}_{stat} │ ~60 │ sq_V1_std_mean │
│ st_{lead}_{feature}_{stat} │ ~240 │ st*V2_st_j40_median │
│ morph*{lead}_{feature}_{stat} │ ~210 │ morph*V1_covedness_score_median │
│ cl*{feature} │ ~11 │ cl*max_st_j40_v1v3 │
│ hr_estimate_bpm │ 1 │ hr_estimate_bpm │
│ rr*{stat} │ ~4 │ rr_std_ms │
│ median_rr_ms │ 1 │ median_rr_ms │
│ n_valid_beats (already listed above) │ 1 │ — │
└──────────────────────────────────────┴─────────┴──────────────────────────┘

STAT SUFFIXES (per aggregation strategy):
\_mean : mean across valid beats
\_median : median across valid beats ← PRIMARY for ML
\_std : std across valid beats
\_min : min across valid beats
\_max : max across valid beats
\_n_valid: count of non-NaN beats for this feature

EXCLUDED FROM THIS FILE (never written):
basal_pattern → metadata only, loaded separately
sudden_death → metadata only, loaded separately
Any computed from test fold statistics

EXAMPLE ROW (truncated):
patient_id, brugada, pipeline_status, n_valid_beats, \
 sq_V1_mean_mean, sq_V1_std_mean, ..., \
 st_V1_j_point_amplitude_median, st_V1_st_j40_median, \
 st_V1_st_slope_j0_j40_median, st_V1_st_convexity_median, \
 morph_V1_covedness_score_median, morph_V1_t_inversion_indicator_median, \
 cl_max_st_j40_v1v3, cl_n_leads_t_inverted_v1v3, \
 hr_estimate_bpm, rr_std_ms
BRU_001, 1, OK, 12, -0.002, 0.041, ..., 0.31, 0.24, -0.003, 0.12, \
 0.71, 1.0, 0.31, 2.0, 68.2, 42.1
