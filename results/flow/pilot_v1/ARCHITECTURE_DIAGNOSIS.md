# Post-hoc architecture diagnosis

The approved width-512 MLP has a fixed linear velocity output. Its 784-dimensional center output has rank 512; at least 272 center directions lie outside that output span.

Write the center velocity as `v_c(x,t) = W_c h(x,t) + b_c`. For any orthonormal basis Q of the null space of `W_c^T`, integration gives `Q^T x_c(1) = Q^T x_c(0) + Q^T b_c`. Therefore those Gaussian initial components survive with unit variance, up to a fixed shift, even with more training. Midpoint integration preserves this identity up to floating-point error. The time embedding does not remove the fixed output-span restriction.

Using the saved 2,000-update checkpoint and its already-generated 64 fixed validation images, the projected initial standard deviation was 1.00304; generated, bias-corrected standard deviation was 1.00304. RMS conservation error was 2.81e-07.

This is a post-hoc mathematical/CPU diagnostic, not an additional transfer run. It explains a structural obstacle in this pilot architecture. The final 10,000-update checkpoint additionally produced extreme samples; that instability is separate from this exact rank restriction.

A follow-up should first add a full-rank input-to-output path or use a spatial architecture, then establish good generation. The evaluator also needs independently validated rejection of non-digit images. These are revised protocol choices and were not launched under the original pilot approval.
