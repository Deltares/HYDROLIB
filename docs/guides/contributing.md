# Contributing


## Tooling
### Black
We use `black` as an autoformatter. It is also run during CI and will fail if it's not formatted beforehand.

### Isort
We use `isort` as an autoformatter.

### Commitizen
We use `commitizen` to automatically bump the version number.
If you use [conventional commit messages](https://www.conventionalcommits.org/en/v1.0.0/#summary), the [`changelog.md`](../changelog.md) is generated automatically. More details below under ["Merging"](#merging).

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

During continuous integration, the checks will be run with several Python versions on Windows, Ubuntu and MacOS. The checks consist of running the tests, checking the code formatting and running SonarCloud.
We advise to use a draft pull request, to prevent the branch to be merged back before developement is finished. When the branch is ready for review, you can update the status of the pull request to "ready for review".

### Reviews
When an issue is ready for review, it should be moved to the "Ready for review" column on the GitHub board for visibility.

### Merging
Merging a branch can only happen when a pull request is accepted through review. When a pull request is accepted the changes should be merged back with the "squash and merge" option.
The merge commit message should adhere to the [conventional commit guidelines](https://www.conventionalcommits.org/en/v1.0.0/#summary).
* In the first textfield of the GitHub commit form, use for example: `feat: Support 3D timeseries in .bc file`, *without* any PR/issue references.
* In the text area of the GitHub commit form, optionally add some more description details on the commit.
* In the same text area, add footer line `Refs: #<issuenr>`, and if needed an extra line `BREAKING CHANGE: explanation`. Don't forget a blank line between footer lines and the preceding description lines (if present).