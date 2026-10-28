# Tag and FFT-window similarity audit

Pairwise distance = radial-spectrum RMS difference (dB). Four-neighbour classes use a per-image/window/spacing robust threshold: median pair distance + 0.5 × IQR. This is a diagnostic rule, not a trained classifier.

The 256 px / 128 px condition is the primary tag audit. Tag and background labels are used only to explain observed classes, never to select references.

Rows: 1,021,600; 256 px tag-free rows: 59,777.

See tag_group_rates.csv for the requested tag strata and 3+1 singleton-tag rate; tag_free_geometry_background_rates.csv separates image geometry from annotated background.
