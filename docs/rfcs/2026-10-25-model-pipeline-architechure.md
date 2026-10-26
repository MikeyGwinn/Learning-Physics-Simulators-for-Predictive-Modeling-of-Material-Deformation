# RFC: Machine Learning Pipeline Architecture

<!-- Decision record header. Leave Status as "Proposed" until your team decides,
     before or after the reviews. Then fill in the outcome and the decision date.
     The document stays where it is; this header is what makes it a record. -->

| Field | Value |
| --- | --- |
| Author | Michael Gwinn |
| Date | 2026-10-25 |
| Status | Proposed |
| Decision Date | Undecided |
| Deciders | Modeling Team (Ellie Ding, William Tu) |
| Superseded By | Not Applicable |

## Problem and Context

Our capstone project aims to train a neural physics model to predict material deformation and identify where materials fail under stress. The undergraduate team is split into two groups: one focuses on simulation, and the other focuses on modeling. One of the modeling team's first tasks is to set up Chanho Kim's point-cloud model, described in his CVPR 2026 paper, as the starting point for a baseline that predicts material deformation. Producing this baseline requires a machine-learning pipeline for data ingestion, preprocessing and feature engineering, model training and validation, and inference.

The team needs to choose an architecture for the machine-learning pipeline. This decision should be made early because switching architectures later may require substantial changes. As a pipeline grows, some architectures can become harder to manage. For example, in a monolithic pipeline, adding different model backends may make dependencies between stages increasingly difficult to manage. Switching later to a more modular architecture could require restructuring how stages exchange data and how training, hyperparameter tuning, inference, and validation are configured. The team's use of Oregon State University's high-performance computing (HPC) cluster for training will also affect how training is configured.

If the team does not make an architecture decision, development may proceed as an unstructured monolith with tightly coupled code and little infrastructure for expansion. Choosing an architecture will guide how the team develops the pipeline.

## Options

### Option A: Component-based Pipeline

A component-based machine-learning pipeline is modular and loosely coupled. Each stage is an independent component with defined inputs and outputs, allowing stages to be replaced, rerun (for example, during training or hyperparameter tuning), or reused independently.

**Benefits:** Independent components would let the team test, rerun, and replace pipeline stages without rerunning or modifying the entire workflow. Clear interfaces could help the simulation and modeling groups coordinate, implement and compare model backends, and adapt training configurations for different models on the OSU HPC cluster.

**Costs:** The team would need to spend time defining and maintaining each component's inputs, outputs, and configuration, and testing that the components work together. These interfaces may add unnecessary overhead if the project uses only one or two models. Changing an interface later could also require updates across multiple components.

### Option B: Monolithic Pipeline

A monolithic machine-learning pipeline builds stages that depend on one another. The stages must run in sequence, and each may expect specific inputs and configurations from earlier stages.

**Benefits:** A monolithic pipeline could be faster to build for the initial baseline, with fewer interfaces and less up-front design. Keeping the workflow in one place may make execution straightforward while the pipeline has only a small number of stages and model configurations.

**Costs:** Because stages depend on one another, a change to a model configuration or data format could affect other parts of the pipeline. Testing or rerunning a single stage independently may be difficult. Adding model backends or workflows could also increase coupling, making maintenance and future restructuring more expensive as the project evolves.

### Option C: Do Nothing

The main costs of making no architecture decision are potentially greater restructuring effort later and more complicated integration between modeling and simulation. These costs may become apparent during the middle or latter part of the capstone, when development, testing, and experimentation with new models and approaches are underway.

## Analysis

The key tradeoff for this project is whether the time spent defining component interfaces is worth the ability to change and rerun parts of the modeling workflow independently. A monolithic pipeline is likely the simplest way to implement and run the first baseline, since the planned workflow initially centers on one point-cloud model (GeeksforGeeks, 2020). It also avoids defining interfaces before the team has settled on data formats and model requirements. However, the project expects to experiment with preprocessing, hyperparameter tuning, model backends, and integration with simulation outputs. In a monolith, changes in these areas may affect several stages at once, making it harder to isolate problems, rerun only the relevant work, or compare experiments consistently. This increases the risk that initial simplicity will lead to integration and restructuring work as the project develops (GeeksforGeeks, 2020).

A component-based design better supports those expected experiments. Data ingestion, preprocessing and feature engineering, training, validation, inference, and hyperparameter tuning can expose explicit inputs and outputs so stages can be tested, reused, or varied without rewriting the entire workflow (Emergent Mind, 2026). This design may also help the modeling and simulation groups coordinate around shared data contracts and make model-specific training configurations easier to manage on the OSU HPC cluster. The cost is the up-front work required to define and maintain contracts between pipeline stages. Overly fine-grained components could overwhelm a small team, especially if the project uses only one or two models.

Neither architecture inherently improves model accuracy or execution speed, and no project-specific benchmarks or prototypes are available yet to quantify runtime or maintenance costs. Given the planned experimentation, lightweight component boundaries appear to offer more long-term value, provided the team validates them with the initial baseline and avoids adding complexity the workflow does not need. If no architecture is chosen, development may still begin quickly, but the project risks accumulating dependencies that become harder to untangle during later experiments and integration with the simulator.

## Recommendation and Verification Plan

I propose a component-based machine-learning pipeline. Although the modeling group is small and establishing a baseline quickly could help guide the project, a modular architecture is better suited to its expected needs. The project will likely explore different types of training data and data-processing methods, as well as experiment with different models. A modular architecture would better support these changes and may reduce future maintenance, helping the project continue smoothly after the capstone.

The team will consider the architecture successful when it has an end-to-end machine-learning pipeline that the modeling group can collaborate on, with each component having clear inputs, outputs, documentation, and responsibilities. The team should also be able to add planned features without redesigning the architecture. Warning signs would include excessive dependencies between stages, needing to restructure the pipeline as features are added, or having to change multiple components for a change that should affect only one. The modeling group will review these criteria after establishing the project baseline.

## Constraints

The main constraint is data use. The team may receive access to additional protected SEM image data, so it must consider confidentiality when deciding what information to store in the project's open-source GitHub repository.

## Delegation and Validation

- **What you handed to AI tools:** I used AI to help draft some of the options' benefits and costs and part of the analysis, using the information already in this document as context. It also did a proof read for grammer at the very end.
- **What checks made that reasonable:** The main check is review of this document by its two required reviewers.
- **Where you stopped and validated the result yourself, and why those points:** I wrote most of the document and reviewed the AI-assisted content myself.

What is assessed is in the [AI policy](https://capstone.alexulbrich.com/assignments/introduction/#ai-policy-applies-to-every-assignment).

## Decision

Proposed: Adopt a component-based architecture.

## References

Emergent Mind. (2026, July 7). Component-based machine learning (CBML). Emergent Mind. https://www.emergentmind.com/topics/component-based-machine-learning-cbml

GeeksforGeeks. (2020, March 24). Monolithic vs. microservices architecture. GeeksforGeeks. https://www.geeksforgeeks.org/software-engineering/monolithic-vs-microservices-architecture/