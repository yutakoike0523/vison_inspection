"""Canonical local-FFT settings used by the image11 comparison figure."""

WINDOW_PX = 128
STRIDE_PX = 16
REFERENCE_DISTANCE_PX = 48
REFERENCE_COUNT = 8
REFERENCE_AGGREGATION = "per-frequency arithmetic mean"
FREQUENCY_RANGE_CYCLES_PER_PIXEL = (0.01, 0.50)
RADIAL_BIN_WIDTH = 0.01
WINDOW_FUNCTION = "2D Hann"
REMOVE_WINDOW_MEAN = True
POWER_NORMALISATION = "divide by total 2D power in 0.01-0.50"
SCORE = "RMS of target-minus-reference radial log-power spectrum"
DISPLAY_PERCENTILES = (5, 99)
DISPLAY_SCALE_SOURCE = "original/no-grid-removal auxiliary normal region"
DISPLAY_SCALE_SHARING = "shared across compared conditions within each spectrum type"

