# Contributing

Reports that improve artifact integrity, reproduction instructions, schema validation, or claim-to-artifact alignment are welcome.

Before opening a pull request, run:

```bash
python -m pip install -e .[test]
pytest
python scripts/privacy_audit.py --root . --git-history
python scripts/build_manifest.py
```

Do not attach raw datasets, target-bearing arrays, model weights without redistribution permission, local paths, credentials, or Paris formal/protected material. Use the issue templates for a claim mismatch or public-artifact bug. Scientific result changes require a new versioned protocol and are outside ordinary maintenance pull requests.
