# Contributing

Contributions are welcome — bugs, features, or documentation improvements.
 
## ⏣ Getting Started

Fork the repo and create a feature branch:

```bash
git checkout -b feature/your-feature
```

For setup and local installation, see [SETUP.md](SETUP.md).

## ⏣ Before You Submit

Format and lint your code:

```bash
black src tests
isort src tests
flake8 src tests
```

Run the test suite:

```bash
pytest -q
```

All checks must pass before opening a pull request. GitHub Actions will verify this automatically on your PR.

## ⏣ Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Write a clear title and description explaining what changed and why
- If your PR changes model behaviour or metrics, include before/after numbers
- Reference any related issues with `Fixes #issue-number`

## ⏣ Reporting Bugs

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Python version, GPU/CPU)

## ⏣ License

By contributing, you agree that your contributions will be licensed under the MIT License.
