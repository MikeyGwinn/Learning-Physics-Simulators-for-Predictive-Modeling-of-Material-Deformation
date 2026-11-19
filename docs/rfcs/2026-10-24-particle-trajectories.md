# RFC: Converting MOOSE Results into Consistent Particle Trajectories

| Field | Value |
| --- | --- |
| Author | Gunnar Larsen |
| Date | 2026-10-24 |
| Status | Proposed |
| Decision Date | Pending |
| Deciders | To be agreed with the simulation and modeling teams |
| Superseded By | Not applicable |

## Problem and Context

Our capstone project aims to train a neural physics model to predict material deformation. The simulation team has MOOSE simulator running on the HPC and has decided to use it for fracture simulations. We are starting with an existing particle dynamics model, which needs a history of point movement to predict what happens next. However, we have not agreed on or implemented how MOOSE results will become those trajectories. This connection is part of [issue #11](https://github.com/MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation/issues/11). MOOSE represents the material using a mesh, which divides the material into small elements connected at points called nodes. It calculates displacement, or movement from the original position, at those nodes. We could use the nodes themselves as our particles, or choose separate material points and calculate their movement from the simulation. In either case, each particle must represent the same material point across frames. For example, particle 12 must continue to describe the same point as the specimen deforms. Selecting new points independently at every frame could make a change in point selection look like material movement. 

Mesh refinement means dividing elements into smaller elements to resolve more detail, which can introduce new nodes. Refining the mesh before a simulation starts is compatible with choosing particles afterward. Changing the nodes during the simulation is harder because the same set of particles would no longer be available directly. A "fixed mesh" here means keeping the same nodes and connections during loading; the material can still deform. We also need to distinguish initial coordinates from displaced positions and preserve the order and spacing of simulation steps. This decision is still open, so we can address these issues before producing a large dataset. Changing the conversion method later could mean converting the data again and repeating work. We need one method that both teams can implement and check.

## Options

### Option A: Use Nodes from a Fixed Mesh

We could select mesh nodes once and follow their displacement throughout the simulation. MOOSE's `NodalValueSampler` exports node IDs, coordinates, and nodal variables [1]. This gives us a direct connection between simulation results and particles, with little additional processing. However, areas with a finer mesh would contribute more particles, and adding or removing nodes during loading would complicate tracking. This makes sense if we keep the same mesh nodes throughout each simulation.

### Option B: Sample Persistent Reference Points

We could choose points in the initial material independently of the mesh nodes. For each frame, we would estimate their displacement from the surrounding mesh values, a process called interpolation. MOOSE provides point sampling [2]. We would track the same original material locations even if the mesh changes. However, interpolation needs checks near pores and cracks, and we would need our own particle IDs because the sampler's reported IDs identify containing elements. This makes sense if the mesh needs to change during loading or we need more control over particle placement.

### Option C: Do Nothing Yet

We could keep the original simulation outputs and postpone conversion. This allows the simulation team to continue experimenting, but leaves the modeling team without agreed trajectories, making it hard for them to continue working. 

## Analysis

The main tradeoff is implementation simplicity versus independence from the mesh. Option A uses existing nodal values directly. Option B gives us more control over particle placement, but introduces another numerical step that needs verification. Neither approach has been tested yet. Coordinates also need careful handling. MOOSE supports output on either the original or displaced mesh. Adding displacement to coordinates that are already displaced would count movement twice. For an original coordinate `X` and total displacement `u(t)`, the particle position should be `X + u(t)`.

Timing matters because the existing neural model's loader calculates its "velocity" feature as a difference between consecutive positions without dividing by elapsed time [3]. Unequal intervals would therefore change the meaning of that feature. We need consistent spacing between saved frames for the initial model and must record whether those intervals represent physical time or progress through an applied loading schedule. Option A also produces more particles where the mesh is finer, so we should inspect their distribution before training.

## Recommendation and Verification Plan

I recommend Option A for the initial pipeline because it gives us the most direct starting point. We would finish any initial mesh refinement, select the material nodes to track, and keep that selection throughout the simulation. The exporter would record each selected node's ID and original coordinates, then collect its total displacement at each saved, converged step. It would match values by ID and calculate `position = original_position + displacement`. Sorting by ID would keep particle ordering consistent even if MOOSE writes rows in a different order.

Each exported sequence would contain three arrays:

| Array | What it stores |
| --- | --- |
| `positions` | The x and y position of every selected particle at every saved frame. |
| `particle_ids` | The node IDs in the same order as the particles in `positions`. Each ID keeps the same place throughout the sequence. |
| `step_values` | One value per saved frame showing when it occurs. The export must state whether these are times in seconds or values along a loading schedule. |

For example, two particles moving 10 micrometers to the right over 0.1 seconds would look like this:

```text
particle_ids = [12, 27]
step_values = [0.0, 0.1]  (seconds)
positions = [
    [[100, 50], [200, 50]],  (first frame)
    [[110, 50], [210, 50]]   (second frame)
]
```

The first x-y pair in each frame belongs to particle 12, and the second belongs to particle 27. Following one particle across frames gives its trajectory. Coordinates would use x right and y up. The exporter would convert declared source units explicitly; for example, 1 millimeter becomes 1,000 micrometers. 

The main verification will be an end-to-end test before generating a larger dataset. We will run a small synthetic case in MOOSE, export its results as particle trajectories, and pass those trajectories through the model's data loader. We will then run one training step and generate a prediction to check that the full pipeline connects correctly.

The test will pass if particle IDs stay consistent across frames, exported positions match the original coordinates plus MOOSE displacement after unit conversion, and the model produces a finite loss and predictions with the expected dimensions. Missing particles or incorrectly ordered steps should cause a clear error. We will save the simulation settings, export commands, and results so another teammate can repeat the test. This verifies conversion and integration; prediction accuracy will need separate evaluation. 

## Constraints

No licensing, data-use, IP, or confidentiality constraints specific to this conversion decision have been identified. The decision changes how simulation results are represented and does not require new external services, third-party datasets, or changes to data access.

## Delegation and Validation

I used Codex to help me understand and compare different methods for turning MOOSE results into particle trajectories. The decision is still open, and implementation and validation would follow team agreement.

## Decision

Proposed. The team still needs to review the conversion method.

## References

1. MOOSE. NodalValueSampler: https://mooseframework.inl.gov/source/vectorpostprocessors/NodalValueSampler.html.
2. MOOSE. PointValueSampler: https://mooseframework.inl.gov/source/vectorpostprocessors/PointValueSampler.html.
4. Kim, Sumukh, and Li. Particle dynamics model loader: https://github.com/chkim403/gs-particle-dynamics/tree/main.
