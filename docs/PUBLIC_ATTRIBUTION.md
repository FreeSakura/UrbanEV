# Public attribution and privacy scope

The current source tree uses **FreeSakura** as the contributor attribution. Personal contact information, institutional affiliation and persistent personal researcher identifiers have been removed from the project's author records, package metadata and current manuscript author blocks. The affected manuscript PDFs and historical-wrapper PDF metadata have been rebuilt. Third-party authorship, citations and license notices remain attributed to their owners.

This is pseudonymous public attribution, not an anonymity guarantee. The GitHub account remains public. Earlier commits, historical tags and automatically generated source archives can still contain prior author information. This update does not rewrite Git history or replace immutable release records. Copies, forks, caches and previously downloaded documents are outside the current-tree cleanup.

New commits for this update use the contributor's GitHub no-reply identity. A regression gate checks the project's structured author fields, author macro and PDF author metadata without storing removed personal values in the test suite. The general privacy scanner additionally checks PDF text, document metadata and annotation payloads for private paths and secrets. Neither automated gate is a proof that every possible personal detail has been detected.

The research update publishes aggregate statistics and derivations only. Raw observations, missingness masks, per-station prediction arrays and individual verification candidates are not part of this publication.
