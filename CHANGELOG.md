<a name="unreleased"></a>
## [Unreleased]


<a name="pre-release-archive"></a>
## [pre-release-archive] - 2026-10-02
### Code Refactoring
- new file structure for new types of VolCo settings.

### Features
- auxiliary supervision branch for object feature extraction.
- new volumeVAE configs.
- Avoid grid collapse when N=4.


<a name="v1.2.0"></a>
## [v1.2.0] - 2026-03-13
### Bug Fixes
- Correct the data augmenetation pipeline: 12 rotations -> 24 rotations.
- Removed unnecessary visualization that causes dataloader crash.

### Features
- Update visualization code.
- Ablation study code.


<a name="v1.1.1"></a>
## [v1.1.1] - 2026-02-28
### Bug Fixes
- Enable training on lower version cuda.
- Minor fixes on to make the diffusion training pipeline work with the new ae.
- (Partly) fixes the memory leakage problem.
- Fix the training latency in loading data and calculating unnecessary losses.
- Fix inconsistency in v2 VAE implementation.
- Update hand_cse implementation.
- Unified gridAE APIs
- Fix the abnormal time consumption of the training pass.
- Remove unnecessary reconstruction code.
- New training pipeline that fixes the NaN issue.

### Features
- Experiment code testing GT reconstruction capability.
- Explicitly add zero-contact to the training data.
- Evaluating with finer object models.
- New version of GridAE
- New pipeline for integrating multiple losses
- New optimization pipeline.
- New preprocessing & training pipeline for gridVAE
- Implement test script for hybrid method for dual diffusion.


<a name="v1.1.0"></a>
## [v1.1.0] - 2026-02-12
### Bug Fixes
- Update testing scripts to suit the new HandObject loading pipeline.
- Minor fixes regarding the weights.
- Update visualization code for graspDiffTrainer.
- Update stable loss to mask out small contacts.
- stable loss enabled.
- Update hand_translation calculation in HandObject.
- Implement a safer way to freeze VAE.
- Fix the dimension mismatch of objtrans.
- Fix the error in visualization.
- Update the checkpoint saving and loading scripts.

### Code Synchronization
- Update checkpoint details.
- Train using new schedule predictiong x0 again.

### Features
- Implement stable loss.
- Trial to integrate hand prior into training & inference.
- New inference-time latent projection.
- Implement Hand Imputation VAE.
- Add ASAM optimizer support & visualization wrapping.
- HAndVAE training using HOI4D dataset
- Add data augmentation
- Add consistency loss and precalculation of point adjacency.
- New reachability loss.
- Detecting the difference between GaussianDiffusion and previous DDPM


<a name="v1.0.1"></a>
## [v1.0.1] - 2026-01-31
### Bug Fixes
- Fix missing attributes regarding msdf_path.

### Code Refactoring
- Make the test dataset only load one object at a time and sample in batches.

### Code Synchronization
- Testing epsilon in GaussianDiffusion to study the difference.

### Features
- Add repulsive loss for grasp optimization.
- Enable mdm-style diffusion training & inference.
- Separate the UNet model with diffusion scheduler.
- Enable sampling test.
- rework the visualization and logging of the model.
- Updated data loading and processing scripts accordingly.
- Implement scripts to preprocess the grid data.


<a name="v1.0.0"></a>
## [v1.0.0] - 2026-01-15
### Bug Fixes
- Fix the problem of unaligned hand & object.
- Update data loading scripts to allow preprocessing of grid data.
- Fix the dimension mismatches for gridae test.
- Fix bugs; Added random seeds for reproducibility.

### Features
- Version 1 of the diffusion training.
- Update the gridae to VAE structure.
- Enable checkpointing according to validation performance.
- Add an extra linear layer in encoding & decoding.
- Enable baseline training&testing.


<a name="v0.2.0"></a>
## [v0.2.0] - 2026-01-04
### Code Refactoring
- Reorganized multiple file structures.

### Features
- New baseline training code.
- Enable end-to-end training with autoencoder initialization.
- Add evaluation code.
- Enable e2e training of the generator.
- Add support for generator training.
- Implement conditional vqvae.


<a name="v0.1.1"></a>
## [v0.1.1] - 2025-12-25
### Code Refactoring
- Move visualization code to vis.py.

### Features
- New structure, losses & visualization for unconditional LGVQVAE training & testing.


<a name="v0.1.0"></a>
## v0.1.0 - 2025-12-23
### Features
- Built the structure of end-to-end model.
- Enable dumping local grids first.


[Unreleased]: /compare/pre-release-archive...HEAD
[pre-release-archive]: /compare/v1.2.0...pre-release-archive
[v1.2.0]: /compare/v1.1.1...v1.2.0
[v1.1.1]: /compare/v1.1.0...v1.1.1
[v1.1.0]: /compare/v1.0.1...v1.1.0
[v1.0.1]: /compare/v1.0.0...v1.0.1
[v1.0.0]: /compare/v0.2.0...v1.0.0
[v0.2.0]: /compare/v0.1.1...v0.2.0
[v0.1.1]: /compare/v0.1.0...v0.1.1
