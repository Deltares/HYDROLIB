# Contributing

## Development setup

HYDROLIB uses [Pixi](https://pixi.sh) to manage development environments, tasks, and
the conda-forge-based GDAL/geospatial stack. After installing Pixi, set up a
working copy with:

``` bash
pixi install
pixi run test
pixi run lint
pixi run typecheck
pixi run --environment docs docs-build
```

## Tooling

### Ruff

We use [Ruff](https://docs.astral.sh/ruff/) for formatting and import sorting,
replacing Black and isort. Run `pixi run lint` locally to check formatting,
imports, and lint rules. Run `pixi run fix` to apply formatting and Ruff's
safe fixes. Linting is not enforced in CI yet.

### Commitizen
We use `commitizen` to automatically bump the version number.
If you use [conventional commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary), the [`changelog.md`](../changelog.md) is generated automatically. More details below under ["Merging"](#merging).

### Conda environment exports

`environment-win-64.yml` and `environment-linux-64.yml` at the repository root
are frozen Conda environment files, generated from `pixi.lock`, for
Miniforge/Conda users who don't want to install Pixi. They let a user recreate
the exact `test-py312` environment with:

``` bash
conda env create -f environment-win-64.yml
conda activate hydrolib
```

Pixi remains the source of truth; these files are a generated export, not a
second dependency set to maintain by hand. Whenever `pixi.lock` changes, run

``` bash
pixi run --environment test-py312 export-conda-environments
```

and commit the regenerated files alongside `pixi.lock`.

Install the [pre-commit](https://pre-commit.com) hook once with
`pixi run --environment test-py312 pre-commit install` and it will regenerate
both files automatically whenever `pixi.lock` is staged; because the hook
modifies files, the first `git commit` attempt after a lockfile change stops
so you can `git add` the regenerated files and commit again. CI also
regenerates both files in a clean checkout and fails the `check-conda-exports`
job if they differ from what's committed, so a PR can't merge with stale
exports even if the hook was skipped or never installed.

## Development

### Branches
For each issue or feature, a separate branch should be created from the main. To keep the branches organized each branch should be created with a prefix in the name:
* `feat/` for new features and feature improvements;
* `fix/` for bugfixes;
* `docs/` for documentation;
* `chore/` for tasks, tool changes, configuration work, everything not relevant for external users.

After this prefix, preferrably add the issue number, followed by a brief title using underscores. For example: `feat/160_obsfile` or, `fix/197_validation_pump_stages`.

### Pull requests
When starting development on a branch, a pull request should be created for reviews and continous integration.
In the description text area on GitHub, use a [closing keyword](https://docs.github.com/articles/closing-issues-using-keywords) such that this PR will be automatically linked to the issue.
For example: `Fixes #160`.

During continuous integration, the test suite runs with several Python versions on Windows and Ubuntu. Formatting and lint checks remain local developer tasks for now.
We advise to use a draft pull request, to prevent the branch to be merged back before developement is finished. When the branch is ready for review, you can update the status of the pull request to "ready for review".

### Reviews
When an issue is ready for review, it should be moved to the "Ready for review" column on the GitHub board for visibility.

### Merging
Merging a branch can only happen when a pull request is accepted through review. When a pull request is accepted the changes should be merged back with the "squash and merge" option.
The merge commit message should adhere to the [conventional commit guidelines](https://www.conventionalcommits.org/en/v1.0.0/#summary).
* In the first textfield of the GitHub commit form, use for example: `feat: Support 3D timeseries in .bc file`, *without* any PR/issue references.
* In the text area of the GitHub commit form, optionally add some more description details on the commit.
* In the same text area, add footer line `Refs: #<issuenr>`, and if needed an extra line `BREAKING CHANGE: explanation`. Don't forget a blank line between footer lines and the preceding description lines (if present).
