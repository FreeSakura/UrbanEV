# Executed model-source snapshots

These directories preserve source files executed by the reported UrbanEV and Paris development-only pipelines. They are research snapshots, not a unified production API.

- `caper/`: local expert implementation and dependency closure. The upstream CAPER paper-to-code identity remains unknown and is not claimed.
- `timexer/`: executed TimeXer snapshot plus embedding, attention, and masking dependencies; retained under the TSLib MIT license.
- `router/`: fixed-fusion and learned-router evaluation code.
- `distillation/`: formal residual-distillation cell runner and aggregate analysis.
- `paris/`: Paris development-only CAPER/TimeXer adapter, common helpers, model config, and aggregate analysis.

The file-level manifest records hashes and licensing. Checkpoints and Chronos weights are excluded.
