# Public/private boundary

The repository and Release packages are produced by positive allowlists. They are not exclusion copies of the research workspace.

Never publish:

- Paris formal/protected targets, reference files, predictions, partial runs, or private Git objects;
- Paris development raw state shards or any complete raw dataset;
- target-bearing arrays, oracle arrays, checkpoints lacking redistribution permission, or Chronos weights;
- private vault paths, access-control details, local absolute paths, tokens, caches, attachments, or automation traces.

Public prediction packages may include model predictions, target indices/times, entity identifiers, masks, target hashes, configuration hashes, and schema metadata. The packages must not include target values or oracle outputs. A hash identifies expected bytes; it does not grant access to them.

Every package family has a closed schema. Unknown keys, object arrays, unregistered string metadata, unsafe ZIP members, manifest/ZIP membership differences, physical paths, and secret patterns fail the public gate. Current-tree JSON/YAML/TOML values, PDF text, ZIP text members, and NPZ string arrays are scanned after decoding.

The superseded immutable `v0.9.0-preprint` record has one documented non-secret path-metadata exception. It is not a target or credential disclosure. The current tree and `v0.9.1-preprint` remove that metadata; the old tag is not moved or rewritten.

`PROTECTED_BOUNDARY_PUBLIC.json` is deliberately non-reconstructive: it records role dates, row counts, hashes, and `analytical_access_count: 0` only.
