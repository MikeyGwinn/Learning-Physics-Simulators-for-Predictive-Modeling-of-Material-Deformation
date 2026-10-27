# Requirements

*Draft. Thresholds to confirm with professors on Fri.*

## Research question

Our question is whether a particle-based neural simulator trained on phase-field fracture simulations of porous graphite (later will be a SiC-SiC composite) can predict how a sample deforms over various load steps more accurately than simple extrapolation baselines.

Right now, we are predicting point displacements 10 load steps ahead given the current and previous couple of states and the loading conditions.

Next, we will see if training on simulations translates well to Dr. Chen's real SEM ET-10 graphite data. Then (probably in the winter), we will predict where damage will form in the sample, and thus where a crack will propagate.

## Data

For data, we have two sources. Both are converted to the format defined in docs/design.md ("Data format v0"). This is the format that will allow the model to learn from our simulation data.

For simulated data, we are using MOOSE phase field fracture runs following Chakraborty et al. 2016's method. We will create these using SEM images and John's mesher, and from generated pore layouts from Gunnar's generator.

We also have Dr. Chen's notched-beam bending tests of ET-10 graphite comprising our real data (SEM video, DIC, and force/displacement logs). Dr. Chen approved this graphite data for our public repo. The future SiC data has to stay private.

## Baselines

    Zero motion
    Constant increment (repeat the last step's motion)
    Load-scaled increment

## Validation

Before we can create any full simulation datasets, we must perform a validation that our data matches the physics. We must make sure MOOSE reproduces Chakraborty's notched specimen benchmark within 0-15% at least, we need good mesh convergence (peak load changes by less than 5% between the two finest meshes), and we need the holes and the soft region pores to agree within 0-10%.

## Acceptance criteria (this term)

By the end of the term, a good walking skeleton would be simulation -> export -> loader -> model -> rollout with end to end data and a CI test that asserts a known result. Hopefully, we would also have a full, or at least half of a dataset in our data format v0. We should also evaluate the trained model against the baselines on held-out samples.

## Success metric 

Mean position error 10 load steps ahead, on new data that the model has never seen.

## Later work

Later in the year, in other terms, we could also make modifications to model output, 3D simulation, and generalize the pipeline to work with our SiC composite.

## These AREN'T goals, we already have enough on our plate.

    Stress or force as a model output
    3D simulation
    Fatigue, temperature, or irradiation effects
    Building a new point tracker or physics model
    Running new experiments
    Real-time deployment / full digital twin

