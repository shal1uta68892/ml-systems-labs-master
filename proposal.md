# Individual Project Proposal (Draft)

**Status:** Draft only; replace the student fields and obtain instructor approval before submission.

**Student name:** `[Enter student name]`  
**Student ID:** `68892`

## Problem

Fashion-MNIST classification is a small but practical image-classification task for
testing whether a neural network can meet resource limits on a constrained
inference device. The Lab 3 CNN provides a reproducible baseline, with 20,490
parameters and a reported test accuracy of 88.93%. This project will optimize
that classifier for low-latency, low-footprint inference without materially
reducing classification quality.

## Dataset

Use the official Fashion-MNIST training and test sets. Reserve 5,000 training
images for validation and use the remaining 55,000 images for training, with
seed 42 and the same normalization as Lab 3. Report the official test-set
accuracy only after model selection is complete.

## Success metric and deployment constraints

The target is at least 88% test accuracy, batch-1 inference latency no greater
than 50 ms, and a serialized model no larger than 5 MB. Record the device,
runtime, measurement method, and latency distribution so the constraints can be
checked on the intended deployment target.

## Planned optimization

Apply post-training quantization to the trained CNN, calibrating with a
representative subset of the training data. Compare the quantized model against
the FP32 baseline for test accuracy, batch-1 latency, and serialized size.
Retain the quantized model only if it meets all three thresholds; otherwise,
report the trade-off and test structured pruning as a fallback.

## Feasibility and risks

The dataset and baseline are small, and the existing pipeline already provides
reproducible data splits and model training. Quantization support and latency
depend on the deployment backend, so the target runtime must be selected and
measured before claiming deployment feasibility.

**Instructor approval:** `[Pending]`
