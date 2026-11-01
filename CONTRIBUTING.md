# Contributing to RehabPose

RehabPose is a research prototype, not a medical device. Contributions that improve its evidence,
reliability or clarity are welcome, and so are reports of things that don't work.

## Set up
You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and Node 24. The project doesn't support
CUDA or NVIDIA packages.

```bash
uv sync
uv run pre-commit install
cd frontend
npm ci
```

## Before opening a pull request
Run these from the repo root:
```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```
Run these from `frontend/` if you changed it:
```bash
npx tsc -b
npm run lint
npm test
npm run build
```

Tests use synthetic fixtures only, with no network, camera or real data. Tests that need pose models skip
when the models are missing. Every public function in `rehabpose.core` needs a unit test.

## Project rules
- **Never invent results.** Every number in docs and reports must come from a command that was run, with
  its output path and MLflow run id for ML. Otherwise write `NOT YET MEASURED`. Numbers from generated
  data must be labelled SYNTHETIC.
- **Update the results source.** README and model card metrics are checked against
  `reports/results_summary.json` by `tests/unit/test_results_summary.py`. Regenerate it with
  `scripts/summarize_results.py` rather than editing it by hand.
- **Never commit data.** That covers videos, personal data and weights over 5 MB. The only exception is
  tiny synthetic fixtures under `tests/fixtures/`, each under 1 MB.
- **Respect dataset licences.** Download scripts are never run automatically. REHAB24-6, for example, is
  non-commercial only.
- **Keep the licence clean.** Core code never imports Ultralytics (AGPL-3.0). It may appear only in the
  `[ablation]` extra, behind a lazy import.
- **Keep the disclaimer.** `rehabpose.DISCLAIMER` must stay verbatim in the README, the first-launch modal
  and every report.
- **Don't make clinical claims.** Thresholds are engineering defaults, so describe them that way.

## Code style
- Type hints everywhere, with `numpy.typing.NDArray` for arrays. mypy is strict for `core` and `api`.
- `core` stays pure where possible. State lives only in engine, filter and state-machine classes.
- Thresholds go in YAML under `configs/`, validated by pydantic, rather than in code.
- Use `logging` in library code, not `print`.
- Docstrings are one line plus units (degrees, seconds, normalized units).
- Line length is 100, using ruff rules E, F, I, UP, B, SIM and N.

## Commits and pull requests
- Use [Conventional Commits](https://www.conventionalcommits.org/), such as `fix(api): reject empty uploads`.
- Keep each pull request small and focused, and say how you verified it.
- If a library's API differs from what the docs say, record the change as an ADR in `docs/adr/`.

## Reporting a problem
Open an issue with what you ran, what you expected and what happened. **Never attach real recordings or
personal data.** If you have a health concern, stop exercising and talk to a healthcare professional. This
project can't give medical advice.

## License
Contributions are accepted under [Apache-2.0](LICENSE).
