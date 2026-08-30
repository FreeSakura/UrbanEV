# Public/private boundary

The repository is produced by a positive allowlist. It is not an exclusion copy of the research workspace.

Never publish:

- Paris formal/protected targets, reference files, predictions, partial runs, or private Git objects;
- Paris development raw state shards or any complete raw dataset;
- target-bearing arrays, oracle arrays, checkpoints lacking redistribution permission, or Chronos weights;
- private vault paths, access-control details, local absolute paths, tokens, caches, attachments, or automation traces.

Public prediction packages may include model predictions, target indices/times, entity identifiers, masks, target hashes, configuration hashes, and schema metadata. The packages must not include target values or oracle outputs. A hash identifies expected bytes; it does not grant access to them.

`PROTECTED_BOUNDARY_PUBLIC.json` is deliberately non-reconstructive: it records role dates, row counts, hashes, and `analytical_access_count: 0` only.
