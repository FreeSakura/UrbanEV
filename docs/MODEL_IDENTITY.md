# Model identity policy

`models/MODEL_SOURCE_MANIFEST.csv` is authoritative for the source files actually executed in the audited study. Each row records a repository-relative path, SHA-256 digest, origin category, license, and local-modification status.

CAPER is treated as a locally executed expert implementation whose upstream paper-to-code identity remains unknown; this repository does not claim official upstream authorship. TimeXer records the executed local snapshot and its required embedding, attention, and masking dependencies. Router, distillation, and Paris files are project implementations or adapters. Chronos weights are not redistributed; the lock records the official revision and expected weight hash.
