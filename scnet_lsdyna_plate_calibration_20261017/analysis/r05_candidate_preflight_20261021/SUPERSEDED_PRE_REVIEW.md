# Superseded pre-review R05 candidate

The first locally generated R05 candidate and bundle were withdrawn before any
solver submission. They are retained in the two `superseded_pre_review_*`
directories only as audit evidence.

Reason: independent review found that the initial runtime ledger could accept
non-finite values, normalize QoI by an incomplete observed element population,
understate crossing-plus-deletion ambiguity, and that the initial bundle builder
did not live-revalidate a stored PASS report. The pre-review package must not be
submitted or used for calibration.

The authoritative candidate is regenerated at the project-level
`manual_upload_20261021_r05_candidate` path only after these issues and their
regression tests pass.
