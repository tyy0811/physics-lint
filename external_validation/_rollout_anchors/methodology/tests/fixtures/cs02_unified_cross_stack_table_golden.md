| Rule | gns-tgv2d | segnn-tgv2d | modulus_ns_meshgraphnet-vortex_shedding_2d |
|---|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) | 5.881e-02 (x1 identical) |
| `energy_drift` | SKIP (x20, D0-18) | SKIP (x20, D0-18) | SKIP (x1, D0-22 (amendment 1)) |
| `dissipation_sign_violation` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) | SKIP (x1, D0-22) |

**Provenance (D0-19 three-sha):**

- **gns-tgv2d**: pkl_inference=f48dd3f376, npz_conversion=f48dd3f376, sarif_emission=8e49339469
- **segnn-tgv2d**: pkl_inference=8c3d080397, npz_conversion=5857144, sarif_emission=8e49339469
- **modulus_ns_meshgraphnet-vortex_shedding_2d**: pkl_inference=n/a_cs02_no_pkl_stage, npz_conversion=n/a_cs02_no_conversion_stage, sarif_emission=a6fbd14

**Inference run status (rung-4c §9 review-gate fold-in):**

- **gns-tgv2d**: n/a (pre-salvage-tag-schema)
- **segnn-tgv2d**: n/a (pre-salvage-tag-schema)
- **modulus_ns_meshgraphnet-vortex_shedding_2d**: from_completed_inference
