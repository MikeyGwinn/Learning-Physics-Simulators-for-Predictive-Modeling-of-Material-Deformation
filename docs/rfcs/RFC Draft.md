# RFC: Representing Loading Conditions in a Material Deformation Model

## 1. Problem and context

Our team is adapting an existing machine learning model for effects of collision between objects to predict material deformation. We need to decide how to represent loading conditions in the model's inputs. In the current setup, loading is controlled through prescribed boundary displacements, so the immediate decision is how to encode those displacements and communicate their effects throughout the material representation.

We should make this decision before redesigning the model because it affects the input features, architecture, and data preprocessing.

## 2. Options

### Option A

Add displacement information to the features associated with each Gaussian, and use the existing PointConv layers to spread these features to the surrounding material.  Whether the current architecture communicates boundary information far enough into the material requires testing.

**Costs:** Input and preprocessing changes, implementation and testing time, and maintenance of the feature definitions and their mapping to Gaussians.

### Option B

Represent each load with a token and add an attention layer so that each token can use information from all other tokens.

**Costs:** Additional architectural changes, token and position encoding, and computational overhead due to additional layer.

### Option C: Retain the current input representation

Do not add explicit loading-condition features at this stage. Retain this configuration as a baseline while assessing the proposed changes.

**Costs:** Inflexibility in responding to different loading conditions. Can cause more work later if the current inputs aren't adequate.

## 3. Analysis

**Option A** appears to require the fewest architectural changes because it builds on the existing PointConv pipeline. It is therefore a reasonable starting point for the current boundary-displacement setup. Its main uncertainty is it's sufficient to capture the resulting deformation, particularly ones far from the boundaries. Simpler implementation does not eliminate maintenance or validation costs.

**Option B** offers a way to combine information across distant regions. It may be useful if local propagation proves insufficient or future data include more varied loading conditions. Our mentor has indicated that implementation may be manageable. The remaining uncertainties include integration effort, training behavior, and computational cost.

**Option C** avoids immediate changes to the input interface, making it the least complex option. Will likely result in worse performance, though its adequacy depends on how much loading information the current inputs already convey. 

## 4. Recommendation and verification plan

I recommend Option A as the initial implementation, due to the relatively simple loading conditions as of now. This approach fits the current boundary-displacement setup and reuses the existing architecture. We should consider Option B if testing shows that Option A cannot adequately capture deformation away from the boundaries or across more varied loading cases.

To assess whether this is the right choice, we will first assess input data to determine whether option A is the appropriate approach. We will then train options A and C and compare their performances. 

## 5. Constraints

The immediate constraint is the information available in the dataset. The proposed encoding requires identifiable boundaries and their displacements. The evaluation plan also requires suitable reference deformation data. Before using datasets, we must check confidentiality to confirm if we can use it.

## 6. Delegation and validation

AI tools were used to suggest ways to incorporate loading conditions and to help revise this RFC. At least one additional reviewer is expected to review and validate the document. In writing this draft, I only included explanations I understood. 
