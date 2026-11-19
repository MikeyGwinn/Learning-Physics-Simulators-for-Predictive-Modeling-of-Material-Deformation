# RFC: Converting MOOSE Results into Consistent Particle Trajectories

| Field | Value |
| --- | --- |
| Author | Gunnar Larsen |
| Date | 2026-10-24 (draft), revised 2026-11-17 |
| Status | Accepted |
| Decision Date | 2026-11-17 |
| Deciders | Gunnar Larsen, Tiago Brockert, Ellie Ding |
| Superseded By | Not applicable |

## Problem and Context

Our capstone project, supervised by Dr. Tianyi Chen and Dr. Chanho Kim, is trying to predict how porous nuclear graphite deforms and cracks under load. We can simulate this in MOOSE on the OSU HPC, but each simulation takes an hour or more, which is too slow for quick predictions. Instead, we are training a neural network on these simulations, starting from Dr. Kim's particle dynamics model [3], and will later check it against Dr. Chen's SEM tests of ET-10 graphite. The work is for Dr. Chen's group, and the long-term goal is a digital twin that can predict damage in reactor parts.

The team is split into a simulation team that runs MOOSE and a modeling team that trains the model. This RFC is about the connection between them (issue #11 [6]), which is how a MOOSE run gets turned into the particle trajectories the model reads. MOOSE breaks the specimen into a mesh of small elements connected at points called nodes, and calculates how far each node moves (its displacement). The model needs points that are the same piece of material in every frame, for example particle 12 has to describe the same spot the whole time, otherwise changing which points we picked would look like the material moving.

This decision needs to be made now because we have started making the dataset. We generated 500 pore samples (issue #30, dataset-v0/batch.json) and started running them through MOOSE. One 8 μm mesh run took 56 minutes 42 seconds on 12 cores and created a 221 MB result file (HPC job 22112216, issue #29). What a run saves is set by its input file, and right now the graphite input only saves the damage field in its smaller output (validation/rve/rve_paper.i, lines 229–231), so displacements are only in the full result file.

The conversion code itself is easy to change, as long as we keep the raw results we can just convert them again. Two parts are hard to reverse though. The first is what each of the ~500 runs saves, since if something is missing we would have to rerun about 470 hours of jobs. The second is how particles and frame spacing are defined, since changing them later means our trained models and earlier results can't be compared. If we do nothing, the runs will keep saving outputs that weren't designed for the model, and the modeling team, who already has a working 2D baseline waiting for data (Ellie Ding, Discord, Nov 17 [9]), stays stuck.

## Options

### Option A: Use the Nodes of a Fixed, Uniform Mesh

We would pick a fixed set of mesh nodes and track them for the whole run, with each particle's position being its node's original coordinate plus its displacement. MOOSE's NodalValueSampler already outputs node IDs, coordinates, and nodal values [1], so the exporter only has to match values by ID.

The main cost is that the number of points depends on the mesh. The graphite input uses a uniform grid (rve_paper.i, lines 29–33), which gives 15,876 nodes at an 8 μm element size and 63,001 at 4 μm. For comparison, one scene from Dr. Kim's dataset has 1,811 particles (modeling/gs-particle-dynamics/data.txt). About 20% of the nodes are also inside pores (the target porosity is 0.2), so they need to be masked. If the mesh changed during loading the fixed set of nodes would break, and a mesh refined around the pores would make the points unevenly spaced. This option makes sense if the mesh is uniform and stays the same for the whole run, which is how our current MOOSE inputs already work.

### Option B: Sample Persistent Reference Points by Interpolation

We would choose points in the undeformed specimen separately from the nodes, for example a grid matching the points tracked in Dr. Chen's experiments. For each frame we would estimate their displacement from the nodes around them (interpolation), using MOOSE's PointValueSampler [2].

This adds another numerical step that needs its own testing, especially near cracks, where the damage band (length scale of 10 μm, rve_paper.i line 21) is only a few elements wide. We would also have to track particle IDs ourselves, since the sampler's IDs are for the element containing the point [2]. This option makes sense if the mesh changes during a run (like the adaptive refinement Chakraborty et al. [5] used around pores), if we need far fewer points than nodes, or if simulated points need to line up with tracked points in real data.

### Option C: Do Nothing Yet

We would keep the full result file from every run and decide on the conversion later. At 221 MB per 8 μm run, 500 runs would take about 110 GB, and several times more with a 4 μm mesh. Uploading large files to Box takes hours (Tiago Brockert, Discord, Nov 16 [9]), so this would all sit on shared HPC storage. The modeling team would still have no data, and anything the runs didn't save would have to be simulated again. This would make sense if the solver setup was still changing and no dataset runs had started. That was true in October, but it isn't anymore.

## Analysis

The first thing to look at is how each option fits the model. Dr. Kim said we can start with an evenly spaced point cloud, and pick the model's grid_size setting based on the spacing between points (Dr. Kim, Discord, Nov 16 [9]). The baseline uses a Point Transformer V3 backbone [4] with a grid_size setting in its config (Ellie Ding, Discord, Nov 17 [9]). Option A on a uniform mesh gives us exactly this, and Option B could too, but only with interpolation. A conformal mesh refined around the pores, the geometry approach in Tiago's RFC [7], would bunch nodes up at pore edges, which goes against what Dr. Kim suggested, so the recommendation requires a uniform mesh.

Units also matter. The graphite input works in micrometers (rve_paper.i, line 6), while the pore generator and SEM mesher use millimeters (AGENTS.md). Since grid_size is in coordinate units, the exporter converts everything to millimeters.

Frame spacing is probably the biggest issue. Dr. Kim's loader calculates velocity as means3D[t] − means3D[t−1] without dividing by a time step (modeling/gs-particle-dynamics/tools/dataset.py, line 272), and our baseline replaced velocity with displacement per frame and removed the time step completely, on Dr. Kim's advice [9]. So a frame's displacement only makes sense if every frame is the same amount of loading apart. We have already seen this problem in the real data, where SEM frames are usually 5 ms apart but some are 6 ms (Michael Gwinn, Discord, Nov 16 [9]). In the simulations, the current input uses adaptive time steps (rve_paper.i, line 254) that get smaller when the solver is struggling, which usually happens as a crack grows, and it saves every tenth step (line 271). So the saved frames are uneven right where the cracks form. The simulation time equals the top edge's displacement (line 13), so equal spacing means equal top displacement between frames. This also matches how the project measures predictions, in load steps (docs/requirements.md, lines 13 and 55). Our design doc asks whether frames should be based on force instead (docs/design.md, line 90), but force drops after the peak as cracks grow, while the top displacement keeps increasing every frame, so displacement is the only one that can space frames evenly. Force is still saved with each sample, since rve_paper.i already calculates it (lines 185–198), so the load-conditioning work can use it.

Positions also have to be calculated correctly. The result files store original coordinates and displacements separately, and our animation script builds the deformed shape as coordinate plus displacement (validation/exo_anim.py, lines 51 and 70). Adding displacement to already displaced coordinates would count the movement twice.

For cost, exported positions for 15,876 nodes as 32-bit floats take about 127 KB per frame, or 13 MB per sample at 100 frames and 6 GB for all 500, compared to 110 GB of full result files. Training cost is less clear, since 16,000–63,000 points is 9–35 times Dr. Kim's example scene. If that becomes a problem, Option A can thin the points evenly by keeping every other node.

For maintainability, Option A's exporter just matches IDs, while Option B needs interpolation code and tests. Neither option sends data anywhere new, so there isn't a security difference. Learning directly on mesh nodes also has published precedent in MeshGraphNets [8].

There are still a few things we don't know. We haven't tested whether the model trains well with 16,000+ points per frame, whether points inside pores should be dropped or masked, or how much error Option B's interpolation has near cracks. The storage numbers are also based on one measured run.

## Recommendation and Verification Plan

I recommend Option A, which the team has adopted, with a few conditions:

1. The mesh is uniform and stays the same for each run, with no adaptive remeshing (our current inputs already do this).
2. Frames are saved at equal increments of top displacement, for example every 0.04 μm up to 4 μm (100 frames). Tiago will choose the MOOSE output setting to do this.
3. All nodes are exported in ID order, and nodes inside pores are kept but masked, using the 0.01 stiffness cutoff exo_anim.py uses for pore elements (lines 57–61), until Dr. Kim tells us whether to drop them.
4. Coordinates are in millimeters, with x right, y up, and the origin at the bottom-left corner.
5. Positions are original coordinate plus displacement.

Each frame is saved as params_coarse.npz containing means3D with shape [1, N, 3] with z set to 0, the format the baseline reads (Ellie Ding, Discord, Nov 17 [9], and docs/design.md). Samples go into train/ or test/ based on the dataset's split.csv (issue #30). Each sample also gets a metadata file with the particle IDs, step values (top displacement in mm), force at each frame, pore mask, generator seed, and MOOSE input settings. For example, two particles that each move up 0.00002 mm between the first two frames:

```text
particle_ids = [12, 27]
step_values  = [0.0, 0.00004]   (top displacement, mm)
means3D      = [[[0.10000, 0.50000], [0.20000, 0.50000]],   (frame 0)
                [[0.10000, 0.50002], [0.20000, 0.50002]]]   (frame 1)
```

Since the decision has been made, these checks are how we will know it worked. They all have to pass before the 500 dataset runs start, and checks 1–3 also run in CI on a small saved result file. Results go on issue #11, and if anything fails we fix it before running the dataset.

1. Known answer test: Run a plate with no pores under the same loading as the graphite input. Without pores the plate stretches evenly, so we know exactly where every node should be. In every frame, the top row of nodes has to move up by that frame's step value, the bottom row can't move, and the middle row has to move half as far as the top. If movement is counted twice or the units are mixed up, this is off by a factor of 2 or 1,000.
2. Equal spacing: The gap between every pair of frames has to be the same, otherwise the exporter stops with an error.
3. Same particles: Every frame has the same number of particles in the same order, with no missing values.
4. Model test: One sample from the dataset loads into the baseline, one training step runs without errors, and the predictions have shape [N, 2].

Checks 1–3 also run on the notched-plate benchmark (validation/sent/sent_paper.i), so we know the exporter works on more than one simulation. These checks make sure the conversion is correct, not that the model is accurate, which is measured separately (docs/requirements.md, Success metric).

## Constraints

I checked for confidentiality limits first. We are only working with graphite data for now, both simulated and Dr. Chen's ET-10 data, which he approved for our public repo (AGENTS.md), so confidentiality doesn't limit this decision. The partner company's SiC/SiC data can't go in the public repo (AGENTS.md), so if we start using it, the exports from it would need private storage and this section would need to be updated.

Storage is also a constraint. We can't commit generated output to the repo (AGENTS.md), so exports live on the HPC share and are archived in Box. Since large uploads take hours, this favors a small export at fixed steps and rules out Option C's plan of keeping every full result file.

I also checked the licenses. Dr. Kim's code is MIT-licensed (modeling/gs-particle-dynamics/LICENSE), so we can match its format and adapt its loader as long as we keep the license notice. Its dataset terms (modeling/gs-particle-dynamics/docs/dataset_terms_of_use.md) only cover his video dataset, which we aren't using. MOOSE is LGPL-2.1 [7], which covers the MOOSE code, not our exported results. None of these rule out an option.

## Delegation and Validation

For the first draft, I used Codex to help explain MOOSE's output options and compare Options A and B. For this revision, I used Claude Code to check the draft against the rubric and reviews, find supporting lines in the repo, and calculate node counts and storage estimates. The team made the decision, and I didn't hand off any of the checks below.

This was reasonable because every claim names the file, line, issue, or message it came from, so each one can be checked directly. Verification checks 1–3 also catch the biggest mistakes no matter who writes the exporter, and they run in CI.

There were a few points where I checked the results myself:

- I read tools/dataset.py line 272 and Ellie's description of her change, since frame spacing decides what the model's main input means, and getting it wrong would mess up every training sample.
- I checked the output settings in rve_paper.i (lines 229–231, 254, and 271), since a missing variable or uneven frames would mean rerunning about 470 hours of jobs.
- I measured the run time and result file size on the HPC (job 22112216).
- I read Dr. Kim's messages in our Discord, since they are the partner's view.

## Decision

Accepted on November 18, 2026 by Gunnar Larsen, Tiago Brockert, and Ellie Ding. We are going with Option A, using the nodes of a fixed, uniform mesh, with equally spaced frames, coordinates in millimeters, and export to means3D. The implementation is tracked in issue #11. If we ever need a mesh that changes during loading, a new RFC will replace this one in favor of Option B.

## References

1. MOOSE Framework. "NodalValueSampler." https://mooseframework.inl.gov/source/vectorpostprocessors/NodalValueSampler.html. Accessed November 17, 2026.
2. MOOSE Framework. "PointValueSampler." https://mooseframework.inl.gov/source/vectorpostprocessors/PointValueSampler.html. Accessed November 17, 2026.
3. C. Kim. gs-particle-dynamics (code for "Learning a Particle Dynamics Model with Real-world Videos," CVPR 2026 Findings). https://github.com/chkim403/gs-particle-dynamics. Accessed November 17, 2026. Copy in this repo: modeling/gs-particle-dynamics/ (PR #25).
4. X. Wu et al. "Point Transformer V3: Simpler, Faster, Stronger." CVPR 2024.
5. P. Chakraborty, P. Sabharwall, and M. C. Carroll. "A phase-field approach to model multi-axial and microstructure dependent fracture in nuclear grade graphite." Journal of Nuclear Materials 475 (2016): 200–208. https://doi.org/10.1016/j.jnucmat.2016.04.006.
6. Project issue #11, "Define simulation-to-model data interface," opened October 19, 2026, in the team's GitHub repository (MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation). Accessed November 17, 2026.
7. T. Brockert. "RFC 1: How microstructure gets into MOOSE, and how trajectories get out." docs/rfcs/RFC1 Tiago Capstone(1).md, October 25, 2026.
8. T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. Battaglia. "Learning Mesh-Based Simulation with Graph Networks." ICLR 2021.
9. Team Discord channel. Messages from Dr. Chanho Kim, Michael Gwinn, and Tiago Brockert (November 16, 2026) and Ellie Ding (November 17, 2026).
