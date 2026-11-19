# Testing RGB Feature Transfer for Pose-Free Feed-Forward Gaussian Reconstruction from Events

## Abstract

An event camera records signed brightness changes. Recovering absolute appearance and consistent geometry from these changes requires information beyond a single local event count. We investigate whether training-time supervision from an RGB vision foundation model helps an event encoder infer a grayscale Gaussian scene. We develop a controlled two-view experiment in which the student receives only events and predicts the relative camera transform and Gaussian parameters in one forward pass. We evaluate interpolated views of held-out EV3DS objects and test whether replacing the events with packets from another scene changes reconstruction quality. The latter test addresses a confound in this benchmark: a shared camera path and a nearly uniform background can yield high image-level scores even when the predicted scene is insensitive to the input.

## Problem and method

Given event packets ending at two source times, the reconstruction network estimates one scene in the coordinate frame of the first camera. It uses the dataset's fixed intrinsics. RGB frames, source extrinsics and depth masks are available for supervision. Rendering a requested view requires a query transform with respect to the first camera. We evaluate reconstruction with a supplied query transform, so this experiment does not measure localization of an unknown target camera.

Each packet is partitioned into two temporal bins and two polarity channels. Log-transformed counts are encoded with shared convolutions. An ordered pair of feature grids predicts the relative camera translation and a six-dimensional rotation representation. Reversing the order of the feature grids gives the second source its own local appearance and geometry estimates. At each grid location the model predicts positive depth and Gaussian radius, opacity and grayscale intensity. We lift centers along calibrated rays, transform the second cloud into the first frame, project isotropic covariances through the perspective Jacobian and composite in depth order.

For RGB supervision, the left source image is passed through the frozen image encoder extracted from the published DVLT checkpoint. A learned projection maps the fused event grid into the teacher feature dimension. We minimize one minus the patchwise cosine similarity together with image and relative-pose losses. This pilot tests transfer of single-image features. The question of whether DVLT's multiview looping layers supply additional geometric supervision requires a separate teacher arm. [DVLT](https://github.com/nv-tlabs/dvlt)

## Experimental design

We take the first 12 training and six validation scenes in the EV3DS scene lists. Eight triplets per scene share the same temporal offsets. Events from the two source intervals end two frames before and after the target. Each arm receives 400 AdamW updates on batches of eight. We pair initializations and sampled batches between the zero-weight and 0.1-weight distillation arms. The student predicts 96 Gaussians and renders a 32 by 24 image. All windows are offline interpolations.

We compare full-frame training with a second objective that adds mean absolute error weighted by the fractional foreground coverage of each target pixel. Coverage is obtained by area pooling positive reference depths. Evaluation reports framewise PSNR on the full image and foreground-weighted PSNR. For the latter, squared errors are weighted by fractional coverage before computing the logarithm. We also cyclically permute scenes at fixed time or zero both event packets and render against the original target. Positive matched-minus-permuted PSNR indicates a benefit from matching event content to the scene.

## Pilot results

Mean validation PSNR (dB) over three paired training seeds (10, 11, 12). Each seed uses the same six held-out scenes.

| Training objective | Teacher weight | Full image | Foreground | Foreground, permuted | Foreground, zero |
|---|---:|---:|---:|---:|---:|
| Full frame | 0.0 | 27.573 | 16.339 | 16.343 | 16.290 |
| Full frame | 0.1 | 27.583 | 16.324 | 16.315 | 16.291 |
| Full frame + foreground | 0.0 | 25.930 | 15.946 | 15.053 | 16.322 |
| Full frame + foreground | 0.1 | 26.401 | 16.456 | 15.305 | 16.322 |

With full-frame supervision, adding the teacher changes mean foreground PSNR by −0.015 dB. With foreground-weighted supervision, the corresponding change is +0.510 dB. For the latter distilled model, permuting scenes reduces foreground PSNR by 1.151 dB, but removing all events reduces it by only 0.134 dB. This asymmetry matters: an incompatible event packet can actively degrade the prediction. A large permutation penalty alone therefore overstates the benefit of receiving the correct packet. The small gain over the zero-input control motivates testing whether a multiview teacher can transfer more scene-specific geometry.

### Camera-spacing stress test

We froze the foreground-weighted, distilled seed-10 checkpoint and changed the distance between the two source indices. We used the same six validation objects and eight starting frames. Reversing the packet order changes the desired reference frame while keeping the source evidence and target timestamp fixed. The four-frame forward input reproduces the training protocol. Mean relative rotation error rose from 0.168° there to 14.482° when source order was reversed. At an eight-frame forward spacing, error was 7.128°, compared with 7.007° for all-zero packets. At a sixteen-frame forward spacing the corresponding values were 21.545° and 21.411°. The small rotation error at the training spacing is consistent with learning the common camera increment. Novel-view PSNR was insensitive to this failure: reversing the four-frame pair increased full-frame PSNR from 26.647 to 26.902 dB. These measurements motivate separating camera generalization from image similarity in the evaluation.

## Scope of inference

The scene is the evaluation unit. Adjacent windows share content, and different seeds evaluate the same scenes. The official test split is reserved. A common trajectory and fixed intrinsics permit camera prediction from a learned prior. Relative pose accuracy on these clips alone cannot establish event-based pose estimation. The coarse spatial grid also limits measurements of edge fidelity. Real-event evaluation, variable baselines, separate geometry metrics, published reconstruction baselines and comparisons against a multiview RGB teacher are needed to support a submission-level claim.

The proposed follow-up replaces binned counts with a predictive event representation and compares ordinary patch matching with supervision that depends on event support. Both changes should be ablated independently under identical pose and rendering losses. [Fast Feature Field](https://github.com/grasp-lyrl/fast-feature-fields) · [REALM](https://github.com/utiasSTARS/REALM)
