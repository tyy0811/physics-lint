"""Release criterion 4: broken CNN shows PH-SYM-001 > 2x baseline."""

from physics_lint.validation.broken_cnn import run_criterion_4_validation


def test_criterion_4_broken_cnn_shows_c4_violation():
    outcome = run_criterion_4_validation(n_training_steps=200, seed=42)
    baseline = outcome["baseline_c4_error"]
    broken = outcome["broken_c4_error"]
    print(f"baseline C4 error: {baseline:.3e}")
    print(f"broken   C4 error: {broken:.3e}")
    assert broken > 2.0 * baseline, (
        f"criterion 4 failure: broken {broken} is not > 2x baseline {baseline}"
    )
