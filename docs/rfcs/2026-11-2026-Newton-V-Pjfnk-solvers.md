# RFC: PJFNK Solver vs. NEWTON Solver.

| Field | Value |
| --- | --- |
| Author | John Scumniotales |
| Date | 2026-11-22 |
| Status | Proposed |
| Decision Date | (fill when decided) |
| Deciders | Tiago, Dr. Chen |
| Superseded By | Not applicable |

## Problem and Context

Our capstone project attempts to predict how nuclear grade graphite deforms and cracks under external loads using ML models. To train these models we need both real-world and synthetic data. For synthetic data we are running finite element (FE) simulations using the phasefield fracture model. 
Finite element analysis assembles partial differential equations and uses algorithms like the newton method to solve them. There is a known issue with moose to calculate the jacobian of the system. The jacobian is required for derivative based solvers like the newton method, to solve this issue MOOSE defaults
to an iterative and less-efficient solver that can approximate the true jacobian. Using the jacobian directly however is much more computationally efficient.

## Options

### Option A: Patch Jacobian & Adopt NEWTON Solver

Description, benefits, costs, and when it would be the right choice.
Making these changes would allow us to increase the speed of FEA simulations by using partial derivatives directly to solve the PDEs. Only a handful of lines of code changes are required but more validation against
the original paper's results are required.

### Option B: Patch Jacobian & Keep Current Solvers

Patch Moose's Jacobian for the phase field material model but keep the PJFNK solver and LU preconditioner. This may increase the speed of the simulations because PJFNK will have to do less inner solves to approximate the jacobian.
Since the patch to the material model doesnt change the governing equations we could just compare residual errors instead of actual physical quantities. 

### Option C: Do Nothing

Use the current PJFNK solver used by Chakraborty et. al. This makes defending our choice much easier because of scientific precedent, however generating data will take longer. This would be the right choice
if patching MOOSE's jacobian calculations and switiching to the NEWTON solver cause significant discrepancies between our results and the original papers. 
## Analysis

## Recommendation and Verification Plan

Initial tests show small physical differences between the PJFNK solver and NEWTON solver but with orders of magnitude speedups. Higher resolution benchmarks are still required to determine if the NEWTON solver is a viable optioon.
We must analyze differences in peak stress, peak displacement, residual errors, and rejected linear solves. 

## Constraints

We are only working with the graphite data now so no IP restrictions apply. Future work with silicone-carbide will require more care and protections. 

## Delegation and Validation

For verifying that MOOSE's phasefield material model's jacobian calculation was wrong I asked codex why the simulations where rejecting linear solves. It gave me the required code changes and 
showed that the error for the Jacobian drops when the patch is applied. It also told me that this issue was publicly known by the MOOSE development team but hasnt been fixed yet. 

A staging enviroment was used to analyze the differences between the implementations, CI could also be used in the feature to reject implementations that produce results that differ too much from our original
chakrabory paper reproduction.

After running the batch scripts with the different solvers I manually inspected differences in peak stress and other parity metrics. 

## Decision

The decision still remains pending. Cheap benchmarks show small differences between the implementations but more in-depth and high-quality benchmarks are still underway. 

## References

1. https://github.com/idaholab/moose/discussions/18265
