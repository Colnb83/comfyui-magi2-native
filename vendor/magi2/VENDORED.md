# Vendored MAGI-2 inference code

`inference/` and `configs/` are copied from a fork of SandAI's MAGI-2-preview inference
repository (Apache-2.0, see LICENSE), branch `integration`, commit `b7ae46c`, plus the training code of branch `feat/train`, commit `72ea703`
(`inference/train/`, and two opt-in hooks: `MAGI2_MOE_AUTOGRAD` / `MAGI2_LAYER_CHECKPOINT` in
`model/magi2_preview.py`, `MAGI2_ADAPTER` in `infra/checkpoint/load_checkpoint.py`).

Changes against upstream, all opt-in through environment variables: NVFP4 expert storage,
single-process operation (no torch.distributed), running without MagiCompiler / flash_attn /
diffusers / unfoldNd, sequential guidance, a memory-bounded refiner forward. The node pack
sets the switches it needs; see `fork_env.py`.

To use a different copy of the fork, set `MAGI2_FORK_DIR` or place it at `magi2_fork/`.
