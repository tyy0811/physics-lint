"""Shared test fixtures and hypothesis profile registration."""

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci-quick",
    max_examples=25,
    deadline=500,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("dev")
