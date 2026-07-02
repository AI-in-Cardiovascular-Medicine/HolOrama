#  Contributing to the project

First off, thank you for considering contributing to HolOrama software!
Whether you’re filing a bug, proposing a new feature, improving documentation, or refactoring code, your help makes this project better for you and the whole community.

## Code of Conduct
This project adheres to the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).  
By participating, you agree to respect everyone in this community.

## Table of Contents

1. [Getting Started](#getting-started)  
2. [I Have a Question](#i-have-a-question)  
3. [Reporting Bugs](#reporting-bugs)  
4. [Suggesting Enhancements](#suggesting-enhancements)  
5. [Your First Code Contribution](#your-first-code-contribution)  
6. [Pull Request Process](#pull-request-process)  
7. [Coding Style & Tests](#coding-style--tests)  
8. [Writing Documentation](#writing-documentation)  
9. [Where to Get Help](#where-to-get-help)

## Getting started
1. Fork the repo and clone your fork
```bash
git clone https://github.com/AI-in-Cardiovascular-Medicine/HolOrama.git
cd HolOrama
```
2. Create a new branch
```bash
git checkout -b feature/new-feature
```
3. install dependencies and run tests (test not yet implemented!)
```bash
pip install -e
pytest
```
## I have a Question 
Before opening an issue:
- Read the [Documentation](https://aivus-caa.readthedocs.io/en/latest/index.html)
- Search existing [Issues](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues?q=is%3Aissue) 

If you still need help:
- Open a new issue: [Click here](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new/choose)
- Provide:
    - A clear, descriptive title
    - Context: what you're trying to do, expected vs. actual behaviour
    - Project verion, OS/platform, Python version, and any relevant logs

## Reporting Bugs
If you find a bug, please help us fix it by opening a [a new issue](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new/choose) and providing:
- **Title**: A short descriptive title
- **Steps to reproduce**: Minimal code snippet or sequence to trigger the bug
- **Expected behaviour** vs **Actual behaviour**
- **Environment**:
    - `HolOrama` version
    - Python version
    - OS and architecture

## Suggesting Enhancements

Open a [feature request issue](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new/choose) and describe:
- The problem you are trying to solve
- Your proposed solution and why it would be useful
- Any alternatives you considered

## Your First Code Contribution

- We currently need help improving the documentation and writing tests for the existing functionality.
- If you feel up for a challenge, we want to implement the possibility of additional contours for e.g. EEM.
- Pick an open issue labelled `good first issue` or `help wanted`, comment on it, and open a pull request referencing the issue number.

## Pull Request Process

1. Make sure all tests pass (`pytest`)
2. Update documentation if your change affects behaviour
3. Keep the pull request focused — one logical change per PR
4. A maintainer will review and merge once approved

## Coding Style & Tests

- Format with `ruff` / `black` (configured in `pyproject.toml`)
- Type-annotate new functions and check with `mypy`
- Add or update tests in `tests/` for any changed behaviour
- Run the full suite with `pytest` before opening a PR

## Writing Documentation

Documentation lives in `docs/` and is built with Sphinx.
- Edit `.rst` files under `docs/contents/` for narrative docs
- Run `.\make.bat html` (Windows) or `make html` (Linux/macOS) to preview locally

## Where to Get Help

- Open a [GitHub Discussion](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/discussions) for general questions
- File an [issue](https://github.com/AI-in-Cardiovascular-Medicine/HolOrama/issues/new/choose) for bugs or feature requests