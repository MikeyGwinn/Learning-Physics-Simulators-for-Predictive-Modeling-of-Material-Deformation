This is a replication of Chakraborty et al.'s (2016) phase-field fracture graphite model in MOOSE including scripts that turn the run into figures and crack animations to compare with the paper's results. This uses the same model as the paper, with pores represented as low stiffness regions applied through a stiffness map onto the mesh. I had Claude compile a readme with some differences between the two models + current results (DETAILS.md).

sent/ is the paper's SENT fracture benchmark (on a metallic material) + a plotting script.

rve/ is the graphite input which uses Gunnar's pore generator + a plotting script

exo_anim.py animates the runs from the Exodus outputs

paper_digitized/ has the info for the Fig. 1b + Fig 4 curves on the paper itself for us to use as validation targets.

So far, for validation, our SENT slope is off only by .1%, our RVE graphite pore test is 14% stronger than Chakraborty's sample, but the crack pattern qualitatively matches the ones described in the paper. We just need to find a way to knock down the peak strength a little bit.
