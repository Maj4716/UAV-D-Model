# R06 paired diagnostic submission set

This directory contains two paired single-factor diagnostics (four solver jobs).
Submit the two Wave 1 controls first. Submit and interpret Wave 2 only after the
controls pass target-solver parsing, normal termination, and output-readability
checks. Each job directory is self-contained and has exactly one head deck.

The historical `p0_r06_real_output_contract.json` is intentionally copied as a
frozen pre-generation snapshot; its `not_generated` wording is historical and is
not edited retroactively. Current lifecycle state is recorded by the batch/job
manifests, static validation, release-readiness report, and bundle manifests.

These diagnostic candidates are not production response points. Do not use them
for P0 fitting, P1 submission, local-engine calibration, or reference labels until
the returned evidence passes the separately frozen run and response gates.
