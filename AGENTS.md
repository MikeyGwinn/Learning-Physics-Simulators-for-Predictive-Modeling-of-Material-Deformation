# AI Agent Instructions


## Project Context


This CS 461 capstone project develops a particle-based neural physics simulator to predict future material deformation from in-situ scanning electron microscopy (SEM) observations.


The planned work includes:
- Tracking points across SEM image sequences to produce particle trajectories.
- Training and evaluating a neural physics model on observed trajectories.
- Comparing predicted deformation with experimental observations.


## Stack and Repository Guidance


- The project uses PyTorch and NumPy; Python is the anticipated primary language. Confirm exact versions and dependencies before assuming them.
- Read `README.md` for project setup and `CONTRIBUTING.md` for the current branch/PR workflow, testing, review requirements, and Definition of Done.
- Follow existing repository structure and conventions. Do not assume directories, scripts, tests, or commands exist without checking.
- Keep changes focused and explain any new dependency or significant design decision.
- Document important tensor shapes, units, coordinate conventions, and model input/output assumptions.
- Do not silently change data preprocessing, temporal train/evaluation boundaries, or scientific assumptions.
- Do not fabricate experiment results or claim tests/training runs were performed unless they actually were.
- Ask a team member before launching expensive training jobs or using remote GPU resources.


## AI Use and Confidentiality


- **Until the project sponsors explicitly approve tools and data categories**, submit only public information, synthetic examples, and otherwise explicitly approved material to external AI services.
- Do not submit nonpublic sponsor code, experimental SEM data, unpublished results, restricted meeting notes, or other partner-provided materials to external AI tools without explicit authorization for the specific tool and material.
- Never include credentials, API tokens, SSH keys, passwords, or secrets in prompts, generated files, commits, or logs.
- Do not search for, open, copy, or transmit restricted files
- If a data classification or tool authorization is uncertain, stop and ask before accessing or sharing the material.
- If a suspected disclosure occurs, stop the affected workflow and notify the user about what occurred and how; follow their incident-response instructions.


## Human Review


- AI-generated contributions must be understood and reviewed by the submitting teammate.
- Follow `CONTRIBUTING.md` for tests, code review, and merge requirements. Report any checks that could not be run.
- Do not merge changes or treat scientific predictions as validated solely because generated code executes.


## Planned Updates (Week 2)


As implementation decisions are made, add the following details. The AI Coordinator will keep file current and coordinate with the listed owners.


| Missing guidance | Source / responsible owner | Update trigger |
| --- | --- | --- |
| Exact environment, dependency versions, and setup commands | Quality Owner (Michael) | Once environment is established |
| Repository layout and module interfaces | AI Coordinator (Gunnar), with subsystem owners | Once initial modules are created |
| Test, lint, and CI commands | Quality Owner (Michael) | Once CI and tests are configured |
| Point-tracking data formats and model tensor contracts | AI Coordinator (Gunnar), with subsystem owners | Once interfaces are agreed upon |
| Training and evaluation commands and reproducibility conventions | AI Coordinator (Gunnar), with ML pipeline owners | Once pipelines are implemented |
| Approved AI tools and permitted categories of partner data | AI Coordinator (Gunnar), with project sponsors | Before any restricted material is used with AI |


Do not present planned features or commands as already implemented.
