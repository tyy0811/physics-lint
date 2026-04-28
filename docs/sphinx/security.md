# Security notes

physics-lint loads adapter files via Python `exec` — the same trust model
as `pytest` loading `conftest.py`.

## Local use

For local development on your own machine, adapters are no more dangerous
than any other Python code you run. No special precautions beyond not
running adapters from untrusted sources.

## CI use

In CI contexts (GitHub Actions, GitLab CI, etc.), physics-lint runs adapter
Python with the same permissions as the CI job itself. **Always set the
minimum permissions** needed for SARIF upload:

```yaml
permissions:
  contents: read
  security-events: write
```

Do not grant `contents: write` or `pull-requests: write` unless you have a
specific need.

## Public-contribution workflows

For workflows where the PR author and the repository owner differ (public
model zoos, OSS projects accepting external contributions), use
`pull_request_target` with branch restrictions per
[GitHub's guidance on that trigger](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#pull_request_target).
The `pull_request_target` event gives adapter code access to repo secrets,
so restrict which branches can trigger the workflow and gate on branch-
protection rules.
