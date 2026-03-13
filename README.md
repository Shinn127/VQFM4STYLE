# VQFM4STYLE

<p align="center">
  <video src="media/anim.mp4" poster="media/Figure.png" controls muted playsinline preload="metadata" width="100%"></video>
</p>

<p align="center">
  <a href="media/anim.mp4">Open demo video</a>
</p>

`VQFM4STYLE` is a research codebase for style-aware human motion generation and real-time character control.

It follows a two-stage pipeline:
- train a `VQ-VAE` to learn a compact discrete representation of 375-d motion features;
- train a style-conditioned `Flow Matching` controller on top of the frozen latent space to generate the next motion step.

In short, this project is best understood as an experimental platform for motion style control, rollout, and visualization.
