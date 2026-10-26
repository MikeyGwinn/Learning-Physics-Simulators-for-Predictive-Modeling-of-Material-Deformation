<!-- docs/rfcs/2026-10-25-MOOSEvFenics.md -->

# RFC: MOOSE vs. Fenics

| Field | Value |
| --- | --- |
| Author | John Scumniotales |
| Date | 2026-10-25 |
| Status | Proposed |
| Decision Date | Pending |
| Deciders | To be agreed with the simulation team |
| Superseded By | Not applicable |

## Problem and Context

There are many open source finite element analysis (FEA) tools out there that are appropriate for different use cases. Among them, MOOSE and Fenics are a good fit because they are both pythonic, support GPU acceleration and distributed computing strategies. Deciding on which tool is very important because one of the main challenge of this project is implementing a highly-accurate framework centered around the phasefield model for material fracture and deformation. 

The math behind this model can be quite hard to understand, and implementing it in multiple frameworks within the scope of this project could lead to divergence between the results under the two frameworks. If the team continues to use multiple FEA packages it will be harder for new & future team members to understand the code and it will expand the complexity of environment setup.

## Options

### Option A: Adopt the MOOSE framework

Instead of maintaining multiple programs with different FEA packages, we centralize all programs around the MOOSE framework. This will help new and current project members to learn and understand the code, and prevent numerical inconsistencies between physical solutions. Cost: Currently one of our programs (in the SEM2Mesh branch) uses Fenics, if the team decides on this option the tool will either need to be re-written for the MOOSE framework or have it's simulation feature depreciated. 

### Option B: Adopt the Fenics framework

Like option A, centralizing around the Fenics framework will be beneficial for understanding the codebase. Tiago has been working on testing & validating the MOOSE framework on the HPC so if the team moves forward with this option, the programs he has written will also require a re-write.

### Option C: Do Nothing

Allow for different contributors to write programs with various FEA frameworks. This option increases the amount of work required to maintain the codebase with the benefit of allowing more flexibility and for contributors to explore different features and implementations.

### Option D: Adopt a Co-simulation Framework
Instead of making the decision to centralize around a single framework or ignore the problem itself we could adopt a co-simulation framework that uses both frameworks to generate the synthetic data required to train the pointcloud model. This would increase the complexity of the code greatly but could improve the computational speed of generating synthetic data and potentially improve the performance of the pointcloud model.

## Analysis

### Summary
Fenics is a stronger choice for rapid development of custom FE methods & experimentation, however MOOSE is often the stronger choice for developing mature Multiphysics applications that benefit from established physics methods. 

### Performance
Fenics generally outperforms MOOSE on traditional physics benchmarks, sometimes greatly. However universal & phasefield specific performance evidence is insufficient to establish an advantage over the two frameworks. See this repo for more details: https://github.com/WieraB/compare-fem?.com.

### Complexity and Maintainability
The MOOSE framework divides applications into configurable objects for residual contributions, physics models, boundary conditions and more. This architecture is attractive because applications and physics are highly portable and easy to share with other developers. This object oriented approach brings many conveniences at the cost of customizability, implementing new physics models may require learning it's C++ object hierarchy and registration system.

Fenics on the other hand, puts more responsibility on the maintainers and developers but lets them express weak contributions directly in code. Furthermore it compiles these codes into executable, optimized kernels that avoid repetitive calls; improving the speed of the framework. This architectural difference is particularly useful when equations change frequently or are unknowns.

On maintainability, Fenics is likely better than the MOOSE framework because of its separation of mathematical definitions and numerical implementation. This concise, variational form will likely be much easier to review and modify than a collection of custom MOOSE implementation classes. Because my experience with both of these frameworks are limited this maintainability analysis is purely based on the different software architectures and requires the implementation of the physics solver under both programs for a final decision to be made.

Sources for this analysis: 
MOOSE Framework System Design: https://mooseframework.inl.gov/sqa/framework_sdd.html?
FEniCS project repo: https://github.com/FEniCS?

## Recommendation and Verification Plan

I propose that the team adopts the Fenics framework because modifying constitutive equations is far easier and faster than doing so under the MOOSE framework. In previous meetings we have discussed using ML techniques to include new terms in the phasefield model to reduce inconsistencies between real-world and simulated results which I think will be difficult (but no impossible) to do with the MOOSE framework. Furthermore, its general performance edge over the MOOSE framework is relevant because of the large amount of data to train the point-cloud model. Other members of the modeling team and our project partner Dr. Chen have proposed using the MOOSE framework so there is a clear need for more discussion and experimentation. 

### Proposed Verification Plan
Since the project is still in its early/exploratory stage, a verification plan is difficult to define concretely. Here is the current plan I have come up with (subject to change):

Constitutive Model Extensibility and Synthetic Data Generation Benchmark Procedure:

1. Implement an equivalent baseline phase-field fracture model in both frameworks using identical material properties, geometries, boundary conditions and meshing procedures.
2. Introduce the same modification to the constitutive equations in both frameworks. 
3. Measure development time, code complexity, and debugging effort require to implement and validate the modification.
4. Generate a synthetic dataset using the same design variables (as defined in step 1.) under both frameworks.
5. Compare computational performance, numerical consistency and throughput through distributed simulations with identical resource allocations on OSU's HPC.
6. Train the point cloud model on each dataset and evaluate performance against a common independent validation set. 

Acceptance Criteria:
- Both implementations produce physically consistent solutions and demonstrate numerical convergence.
- Constitutive-model modifications are successfully implemented and validated in both frameworks.
- Development time, code complexity, and maintainability are evaluated.
- Runtime, peak memory usage, and successful simulations per HPC node-hour are recorded.
- Both frameworks generate datasets using an identical output schema with documented failure rates.
- Results are reproducible by another team member.

## Constraints

N/A, both frameworks are open source and can be run 100% locally, therefore no data needs to be public. 

## Delegation and Validation

- **What you handed to AI tools** Documentation and Framework sources where provided to AI tools after initial, manual analysis to verify correctness.
- **What checks made that reasonable**: AI-generated comparisons were checked against official framework documentation, published benchmarks, and publicly available source code. No automated tests or CI checks were performed, as the analysis does not introduce changes to the codebase.
- **Where you stopped and validated the result yourself, and why those points.** I manually reviewed the performance benchmarks between the two frameworks under the provided benchmarks. Since adoption of a single framework is not easily reversed the decision remains proposed until the team completes the verification plan and reviews the results.

## Decision

Discussions on this decision are still ongoing. 