# ML Predictor Summary

- Training structures: **20**
- Features: **19** structural descriptors
- Target: **log₁₀(D_tracer)** averaged across 500/600/700/800/900 K
- Validation: **Leave-one-structure-out**

## Cross-Validation Results

| Model | MAE (log₁₀D) | RMSE (log₁₀D) |
| --- | ---: | ---: |
| Mean | 0.1586 | 0.1904 |
| Ridge | 0.0525 | 0.0653 |
| KNN | 0.0534 | 0.0650 |
| RandomForest | 0.0555 | 0.0649 |
| GradientBoosting | 0.0617 | 0.0719 |

## Top Features (RandomForest)

- `li_s_coord_3p0_mean`: 0.1273
- `natoms`: 0.1050
- `li_li_nn_std_A`: 0.0847
- `li_li_nn_mean_A`: 0.0835
- `li_cl_nn_std_A`: 0.0790
- `is_bulk`: 0.0776
- `is_gb`: 0.0754
- `li_s_coord_3p0_std`: 0.0712

## Interpretation

- This ML model predicts tracer diffusion coefficients from static structural features.
- The model is trained entirely on UMA-MD simulation labels, not experimental data.
- Feature importance reveals which structural descriptors most influence Li transport.
- This constitutes the 'structure → transport' ML framework of the thesis.
