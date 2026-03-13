# VQFM4STYLE

<p align="center">
  <img src="media/anim.gif" alt="VQFM4STYLE animation demo" width="100%" />
</p>

`VQFM4STYLE` is a research codebase for style-aware human motion generation and real-time character control.

It follows a two-stage pipeline:
- train a `VQ-VAE` to learn a compact discrete representation of 375-d motion features;
- train a style-conditioned `Flow Matching` controller on top of the frozen latent space to generate the next motion step.

In short, this project is best understood as an experimental platform for motion style control, rollout, and visualization.
