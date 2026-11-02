# Risk Register

This is the project's only risk register. Last reviewed: 2026-11-01.

**Objective.** By fall week 10, the walking skeleton (MOOSE simulation → data format v0 → loader → model → rollout) runs end to end on sample data on the OSU HPC. Over the year, the model predicts porous graphite deformation 10 load steps ahead better than the extrapolation baselines. Every impact rating is judged against this objective.

| Rank | ID | Risk | Category | Likelihood | Impact | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | R-3 | Simulations may not match real graphite | Technical | High | High | Tiago |
| 2 | R-1 | Baseline model doesn't fit our problem | Technical | High | High | Ellie |
| 3 | R-6 | Production target not agreed with the partner | Business | Medium | High | Tiago |
| 4 | R-4 | Confidential SiC/SiC data could leak | Business | Medium | High | Gunnar |
| 5 | R-5 | Simulation data can't be produced fast enough | Technical | Medium | Medium | John |
| 6 | R-2 | SiC/SiC data may arrive late | Business | Medium | Low | Gunnar |

### R-3: Simulations may not match real graphite
Given that the model learns only from our MOOSE simulations, those simulations may not reproduce Chakraborty et al. (2016) or Dr. Chen's experiments. The cause could be the solver setup, the mesh resolution, or a graphite grade that differs from the paper's. The model would then learn the wrong physics.
- **Likelihood: High.** Validation thresholds aren't agreed.
- **Impact: High.** The model can be no more accurate than its training data.
- **Response (mitigate):** Before generating a full dataset, reproduce the paper's notched-specimen benchmark, show mesh convergence, and confirm the graphite grade with Dr. Chen.
- **Trigger:** Peak load differs from the paper by more than 15%, or changes by more than 5% between the two finest meshes.

### R-1: Baseline model doesn't fit our problem
Given that Dr. Kim's model steps through time, takes 3D point clouds, and fits one rigid transform per object, adapting it to load-driven, 2D, continuously deforming material may not work within the term.
- **Likelihood: High.** Load conditioning and the 2D changes (issues #9 and #10) are open questions in `docs/design.md`.
- **Impact: High.** The model is the last stage of the walking skeleton, and the success metric is defined per *load* step.
- **Response (mitigate):** Make the smallest change first: set D = 2, drop the rigid-transform fit, and train on a few samples. Settle the load-conditioning approach in its RFC.
- **Trigger:** The 2D model can't overfit one simulated sample by 2026-11-22, or doesn't beat the load-scaled-increment baseline on held-out samples.
- **Asked:** Dr. Kim, 2026-10-20, about load conditioning.

### R-6: Production target not agreed with the partner
Given that no supervisor has confirmed the OSU HPC and sample data as the week-10 target, we may build toward a target the partner doesn't accept.
- **Likelihood: Medium.** We already run there, but nobody has asked for the target to be confirmed.
- **Impact: High.** The target is a checkpoint milestone item.
- **Response (mitigate):** Get Dr. Chen and Dr. Kim to confirm the deployment section of `docs/design.md`, and record their answer in an issue.
- **Trigger:** No confirmation by 2026-11-08.
- **Start by:** 2026-11-06.

### R-4: Confidential SiC/SiC data could leak
Given that the repo is public, the team uses AI tools, and no private storage location has been chosen, SiC/SiC data could be committed or sent to an AI service when it arrives.
- **Likelihood: Medium.** Nothing automatically checks for data files.
- **Impact: High.** A leak breaks the partner company's approval, and a committed file stays in Git history.
- **Response (mitigate):** Choose a private storage location before the data arrives, add data directories to `.gitignore`, and get written sponsor approval of which AI tools may see which data. The rules are already in `AGENTS.md`.
- **Trigger:** The data arrives and no storage location is written down.
- **Asked:** Dr. Kim, for AI tool and data approval.

### R-5: Simulation data can't be produced fast enough
Given that each MOOSE run takes hours, runs from one container image in one member's HPC directory, and only one or two people can set it up, we may not produce enough training data in time.
- **Likelihood: Medium.** INL's conda server failed on 2026-10-26, and the cost per sample at our target resolution isn't measured yet.
- **Impact: Medium.** A smaller dataset weakens the model but doesn't block the skeleton.
- **Response (mitigate):** Measure the cost per sample and size the dataset to fit. Move the container image to a group directory, and have a second teammate run a job from the documented steps.
- **Trigger:** The modeling team goes a full sprint without new simulated data.

### R-2: SiC/SiC data may arrive late
Given that the SiC/SiC data needs approval from the partner company, it may arrive too late for this term.
- **Likelihood: Medium.** Dr. Chen estimated 1–2 weeks, but the approval is outside his control.
- **Impact: Low.** The pipeline doesn't depend on the material, so the graphite work continues.
- **Response (accept):** Continue on graphite. If the trigger fires, the SiC/SiC work moves to winter term.
- **Trigger:** No access by 2026-11-13.
- **Asked:** Dr. Chen, during sprint 1.

## External prerequisites

| Prerequisite | Risk | Asked | Status |
| --- | --- | --- | --- |
| SiC/SiC data from the partner company | R-2 | Gunnar → Dr. Chen, sprint 1 | Waiting |
| Production target agreement | R-6 | Tiago → Dr. Chen and Dr. Kim, start by 2026-11-06 | Not asked |
| Validation thresholds | R-3 | Tiago → Dr. Chen  | Waiting |
| Load-conditioning approach | R-1 | Ellie → Dr. Kim, 2026-10-20 | Follow-up planned |
| ET-10 graphite data shared | — | Dr. Chen | Approved |
| OSU HPC accounts | — | — | Granted |

## Staging and rollback plan

| What | Plan | Lands by | Owner |
| --- | --- | --- | --- |
| Where changes are tested before use | CI runs on every PR. Changes to the simulation or model then run on a small sample batch on the OSU HPC before any full run. | CI 2026-11-08; sample batch 2026-11-15 | Michael |
| Undoing a bad change | Revert the merge commit with a new PR. Never force-push `main`. | In place | Michael |

## Closed

