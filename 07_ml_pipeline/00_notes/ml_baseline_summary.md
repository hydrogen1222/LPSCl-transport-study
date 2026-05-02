# ML Baseline Summary

## Interpretation

- This baseline validates the data pipeline, not the final thesis claim.
- The direct ML target is `log10(D_tracer)`, not experimental-equivalent room-temperature conductivity.
- Formal conductivity remains a derived upper-bound quantity.

## Results

| Task | Model | Rows | Structures | MAE (log10D) | RMSE (log10D) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| screening_1000steps_600_700K_log10D | mean | 16 | 8 | 0.1385 | 0.1772 | Leave-one-structure-out validation on available labeled rows. |
| screening_1000steps_600_700K_log10D | ridge | 16 | 8 | 0.2175 | 0.2633 | Leave-one-structure-out validation on available labeled rows. |
| screening_1000steps_600_700K_log10D | knn | 16 | 8 | 0.1625 | 0.1959 | Leave-one-structure-out validation on available labeled rows. |
| formal_20000steps_600_700_800K_log10D | insufficient_data | 9 | 3 | NA | NA | Need at least 5 unique structures for meaningful leave-one-structure-out validation. |

## Current recommendation

- Screening-level MD labels are sufficient to validate the ML plumbing.
- Formal `20000`-step labels are still too few for a serious conductivity predictor.
- The next GPU batch should expand formal labels before any stronger ML claim is made.
