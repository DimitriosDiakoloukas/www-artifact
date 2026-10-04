# Phase 1: hyperparameter selection (development)

Written 2026-10-03 before any run of this study. Design: NEW_CAMPAIGN_PLAN Section O (Phase 1 design).

Grid per (architecture, network): learning rate in {1e-3, 5e-3, 1e-2} x weight decay in {0, 1e-5};
T = 8; seeds {0, 1}; spectral features; everything else as in Phase 0 (Adam, clipping 1.0, validation
every 5 epochs, patience 20, at most 300 epochs; SIDNET restart as released per dataset; published
architecture settings). Slashdot and Epinions run with --memory-efficient (identical results).

Selection rule: for each (architecture, network), the configuration with the highest mean validation
AUC over the two seeds; ties broken toward the smaller learning rate, then the smaller weight decay.
Runs use --train-only, which never reads test labels. The selected configurations are written to
selection.json by select.py and are frozen into the confirmatory protocol for every T.
