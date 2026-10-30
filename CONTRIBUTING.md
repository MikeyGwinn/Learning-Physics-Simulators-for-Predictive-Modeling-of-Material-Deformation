# Contributing Guide

This guide describes how our undergraduate capstone team will collaborate on the research project **Learning Physics Simulators for Predictive Modeling of Material Deformation**. The our contributions to the project are just getting started, and a existing repository is not in place for what we will be working on. We will update the check and setup commands once we get started with development.

## Code of Conduct

Be respectful, constructive, and inclusive. Raise team-process or conduct concerns privately with project-manager or instruction team.

## Getting Started

1. Clone the repository and create a branch from `main`:

   ```text
   git clone https://github.com/MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation.git
   cd Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation
   git switch -c feat/short-description
   ```

4. Install project dependencies and run the checks listed in **Testing** and **Code Style, Linting & Formatting**. Those commands will be added when the stack and tooling are established.
5. Do not commit credentials, private information, or raw project data unless the team has confirmed it is permitted and appropriate to store in this repository. Use the team's approved storage location for restricted or large data, and document how to access it without embedding secrets.

## Issues & Planning

We have yet to decide what to use to record work, questions, bugs, and decisions. Before starting substantial work, agree on an issue or sprint task with the team and identify what will count as acceptance. Keep work small enough to review in a timely way; split larger tasks into independently reviewable pieces.

If work happens in a simulator, data platform, or another system outside this repository, record the task and its outcome in the tracker and link to the external work where access permits. Do not copy restricted data into an issue or pull request.

## Branching & Workflow

- `main` is the shared integration branch. Do not commit directly to it.
- Create a branch for each task, based on the latest `main`. Use short names such as `feat/add-baseline`, `fix/data-loader`, or `docs/setup-guide`; include an issue number when available.
- Open a pull request (PR) to merge work into `main`. Keep the PR focused and explain its purpose, approach, and validation.
- If `main` advances while you are working, update your branch using the team's agreed Git procedure. Ask before resolving a conflict that could change scientific meaning, data interpretation, or another contributor's work.

## Commit Messages

Write concise, imperative commit subjects that describe the change, for example:

- `Add simulator data loader`
- `Document local setup`

Include the issue number in the PR title or description when one exists. The team may adopt a stricter commit convention later if its tools require one.

## Code Style, Linting & Formatting

The language, formatter, linter, and their configuration files are not established yet. The team will choose and document them when the starter code and project stack have been determined. Until then, keep changes readable, consistent with nearby code, and free of unrelated formatting changes.

**Local check/fix commands:** to be added with the selected tools.

## Testing

Component-specific correctness checks are documented with each component; see
[pore_generator/README.md](pore_generator/README.md#development-and-checks).
Run `git diff --check` for whitespace errors. Add or update relevant checks when
changing behavior, and document checks that cannot reasonably be automated.
No project-wide test runner or CI test job is configured yet.

## Definition of Done

A change is done when it is reviewed, validated, documented where needed, and merged. This is a shared team commitment, not just the author's checklist. The table distinguishes current manual expectations from automation the team still needs to build.

| Check | Required before merge | Enforcement and owner |
|---|---|---|
| Review | At least one teammate reviews the PR, understands the change, and approves it. Authors do not approve their own PRs. | Manual PR review now. Configure a required approval rule when repository settings permit; **owner: Michael Gwinn**. |
| Tests | Relevant tests pass; add or update tests for changed behavior. | A local standard-library suite covers the pore generator. No CI test job exists yet; proposed CI owner: Michael Gwinn in **Sprint 2**. |
| Lint and formatting | The configured linter and formatter checks pass. | No tools or checks exist yet. Select and configure them, then add a CI lint job in **Sprint 2**; **owner: Michael Gwinn**. |
| Security | No secrets or unauthorized data are committed; dependency/code security checks pass once configured. | Authors and reviewers check changes manually now. Add an appropriate automated dependency/security scan in **Sprint 2**; **owner: Michael Gwinn**. |
| Documentation | Update the README or relevant documentation when setup, usage, results, assumptions, or behavior changes. | Author checklist and reviewer check on every PR. |
| CI | All required CI checks pass before merge; do not bypass a failing or missing required check without explicit team agreement. | CI is not configured yet. Add a CI workflow that runs tests, lint/format, and the selected security scan in **Sprint 2**; **Michael Gwinn**. |

**Sprint 2 is a proposed target** because the project tools and code are not available yet. At sprint 1 we will formalize details.
## Contribution Norm

Every student is responsible for at least one piece of accepted work in every sprint, consistent with the course's [Contribution Lines](https://capstone.alexulbrich.com/assignments/sprint-notes/#contribution-lines) and [Sprint Notes](https://capstone.alexulbrich.com/assignments/sprint-notes/) expectations.

- **Where work lives:** Track tasks and decisions in this repository's issue tracker. Code, tests, configuration, and project documentation belong in this repository. For work in the simulator, a data platform, or a partner's system, create or update a tracker item here and include a link or a concise, non-sensitive record of the work and result.
- **What accepted means:** A code, test, configuration, or documentation contribution is accepted when its PR is approved and merged. An external-system or research contribution is accepted when the agreed deliverable is completed, recorded in its tracker item with enough evidence or explanation to review, and acknowledged by a teammate or supervisor. Agree on acceptance criteria before the work starts.
- **Make the norm workable:** Size tasks so each can be completed and reviewed within the sprint. Pair on unfamiliar or high-risk work, and make each person's contribution visible in the issue/PR record. A paired task can count for each student only when the record makes each person's distinct contribution clear.
- **Review turnaround:** Acknowledge review requests within two school days when possible. If you cannot review by then, tell the author and team when you can; the team should arrange another reviewer rather than leave work blocked.

## Pull Requests & Reviews

Before requesting review, make sure the PR:

- links the issue or explains why there is none;
- describes what changed and why;
- lists tests, checks, or manual validation performed (and any not yet available);
- updates relevant documentation; and
- calls out assumptions, data constraints, or known limitations.

Reviewers should check correctness, clarity, test coverage, reproducibility where relevant, data handling, and documentation. Give specific, respectful feedback. The author should respond to comments and request re-review after changes. Do not merge until the Definition of Done is met; if a required gate is not yet available, explicitly verify the manual expectation instead.

## CI/CD

There is no CI/CD workflow yet. The Sprint 2 target in the Definition of Done is to add CI for tests, lint/format checks, and the selected security scan. Until then, run available checks locally and state clearly in the PR which checks do not exist yet. Add the workflow location and instructions for viewing or rerunning jobs here when it is created.

## Security & Secrets

Never commit credentials, access tokens, private student information, or data that the team is not authorized to publish. Do not put secrets in source code, issues, PRs, logs, or screenshots. Report a suspected vulnerability or accidental secret exposure privately to **project-manager or project partners**, not in a public issue. The team should promptly follow the course or institution process for rotating exposed credentials and handling sensitive data.

## Documentation Expectations

Update this guide when setup commands, tools, or team workflow change. Update the README or relevant project documentation when a change affects installation, usage, experiment steps, data provenance, assumptions, results, or limitations. Document enough for another team member to reproduce important results without disclosing restricted data.

## Release Process

No release or packaging process has been established. For now, merge reviewed work to `main` and record notable milestones and reproducible results in the project documentation. Define versioning, tagging, packaging, and rollback steps if and when the project needs releases.

## Support & Contact

Ask routine questions in **team discord channel** or the relevant issue/PR so answers are visible to the team. For private, sensitive, or time-critical concerns, contact **project-manager, or project parters**. Add expected response times once the team agrees on them.
