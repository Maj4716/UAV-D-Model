# P0 R05 candidate worklog

Date: 2026-10-21 (Asia/Shanghai)

1. Re-read the R04 Stage 3–5 assessment and treated its failures as design evidence, not as a basis for post-hoc acceptance.
2. Audited the historical generator and validator. Confirmed their fixed paths would overwrite normalized R04 artifacts and that they did not bind role/speed/mesh semantics or a robust primary QoI.
3. Checked the LS-DYNA R15/R16 D3PART contract. Froze `PSETID` in Card 1 field 5 and a `*SET_PART_LIST_TITLE` containing fragment PIDs 2001–2007 only.
4. Froze `design/p0_cases_r05.json` before any R05 solver outcome:
   - 0.75 mm speed scan: 600/800/1000/1200/2000 m/s;
   - 2000 m/s mesh scan: 1.00/0.75/0.50 mm;
   - termination 0.04 ms;
   - full-model D3PLOT 0.0005 ms and fragment-only D3PART 0.00005 ms;
   - capture plane z=-12 mm;
   - explicit availability, stability, classification, bracket, and mesh gates.
5. Added a dedicated R05 generator. It copies all bound inputs into a new self-contained directory, stages generation, refuses an existing target, and never writes the R04 package.
6. Added an R05 validator. It checks the complete inherited R15 keyword contract, D3PART card and part set, exact case semantics, capture-plane sampling resolution, frozen thresholds, mesh references, part/material/EOS references, velocities, geometry, mass integrals, P1 denial state, and hashes.
7. Generated `manual_upload_20261021_r05_candidate/`; static validation returned PASS.
8. Added the pure capture-ledger algorithm and synthetic regression tests. It tracks initial reference mass through first crossing, live upstream, deletion before/after crossing, and temporal ambiguity; unavailable values remain null rather than becoming zero.
9. Added deterministic generation, overwrite refusal, tamper rejection, and R04-preservation tests.
10. Ran the then-existing 20 pre-review project-local unit tests: PASS, including PSETID mismatch, plate-PID inclusion, and duplicate-fragment-PID rejection.
11. Ran keyword-format self-test: PASS.
12. The first legacy postprocessor self-test attempted to use the sandboxed Windows system TEMP and failed with `PermissionError`. Set `TEMP/TMP` to `D:\Desktop\UAV新\.tmp\r05_selftests` and reran unchanged: PASS.
13. Built an explicit-whitelist ZIP with 8 candidate files plus one embedded bundle manifest. ZIP CRC and every embedded SHA-256 verified: PASS.
14. No solver job was submitted. R04 evidence was read only for the documented diagnostic/design repair, not for post-hoc R05 acceptance or engine/dataset parameter selection. No R04 result, material value, P1 row, engine code, dataset, training run, or sealed Test was modified; no training data or sealed Test was read for model selection.

15. Independent pre-submit review found that the first candidate could accept non-finite QoI, self-normalize a truncated D3PART element set, misclassify crossing-plus-deletion in one output interval, and bundle a stale PASS report. The candidate was therefore withdrawn before submission; no solver evidence was affected.
16. Tightened the pre-registered contract before any R05 outcome: outward-positive qP, centroid-translational qE, identity-bound expected reference mass, worst-case nonperforation mass bound, adjacent-pair qV jump rule, explicit run-integrity definitions, and fail-closed missing/non-finite handling.
17. The first generated candidate and bundle will be retained under a superseded-pre-review evidence directory. Final candidate identities and test counts are recorded only after regeneration and full regression.

18. Moved, rather than deleted, the withdrawn package and bundle to `superseded_pre_review_candidate/` and `superseded_pre_review_bundle/`; added `SUPERSEDED_PRE_REVIEW.md` so they cannot be mistaken for the upload target.
19. Hardened the frozen measurement contract: element position is the eight-node arithmetic centroid; `qP` is outward-positive; `qE` is element-centroid translational energy; `qM/qP/qE` use the identity-bound expected reference mass and associated `v0` scales; `qV_over_v0=qP/qM` and `qV_mps=qV_over_v0*impact_speed_mps`.
20. Hardened fail-closed handling: missing reference mass contributes to the worst-case crossing bound; alive-to-deleted motion that can cross within one output interval is temporal ambiguity; missing/non-finite inputs and derived values are rejected; confirmed nonperforation requires the complete worst-case crossing mass bound to be at most 0.5%.
21. Hardened the static validator to freeze the complete measurement and gate documents and to lock material, EOS, contact, section, timestep, energy, boundary, and termination cards. Self-consistent material/card tuning with refreshed hashes is now rejected against the R04 baseline.
22. Hardened the response assessor to use strict JSON, freeze raw and canonical design identities plus deck/case-map/package-manifest identities, recompute availability/classification/`qV`, compare only adjacent confirmed-perforation speeds, and reject missing or unavailable mesh QoIs.
23. Added `scripts/extract_r05_d3part.py`. The adapter checks D3PART identity, required arrays and header flags, shapes, finite values, PID/part index, initial alive state, element counts, first time, state spacing, and termination coverage; it computes reference mass by eight-node volume integration and material density before building each PID ledger.
24. The D3PART adapter remains `unverified_until_real_r05_d3part`. Its assessor bridge requires separately bound PASS run-integrity and real-integration evidence; synthetic interoperability cannot promote this status.
25. Hardened the bundle builder to rerun live static validation, require exact equality with the stored PASS, reject output paths inside the package or identical ZIP/manifest destinations, and collect only the explicit eight-file whitelist.
26. Regenerated the authoritative candidate and bundle. Final identities after the wording clarification in step 31: deck `6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c`; raw design `b1d2027d57fc6d9cd1ee3f932ea044fd16a112344d8f2b6f508071f45191a06f`; canonical design `85b35f1a23ac0b822d0ce65fd9232327c9e566e5cfdddb080814e01414ebdf97`; case map `b05b59249e49c3e9d15904f475c2f9478bdff1638fb797d1b552298490aa5295`; package manifest `e178b6c28832519aaa2befdc969dcd93d2141ea4073ad740b52b17968d92ba44`; ZIP `4d52697f1ea2c3f56cc7aea1c6753b49a95764e72c16a3fe141c5f5ecfb86372` (3,160,411 bytes).
27. Ran the first complete hardened `test_r05*.py` suite: 50/50 PASS in 51.077 s. R05 scripts/tests `py_compile`, keyword-format self-test, workspace-TEMP postprocessor self-test, static validator, ZIP CRC/whitelist/embedded hashes/external bundle identity, and R04 preservation all PASS.
28. During final evidence collection, one manual validator invocation used `..\manual_upload_20261021_r05_candidate` from the project directory and therefore raised `FileNotFoundError`; rerunning with the actual absolute package path returned PASS. This was a command-path error, not a candidate failure.
29. The first ad-hoc ZIP whitelist check expected the embedded name `bundle_manifest.json`; the actual frozen name is `candidate_bundle_manifest.json`. The corrected exact-set check passed together with CRC and every hash. This was a diagnostic-name error, not a ZIP defect.
30. The platform software identifier `ls-dyna_R15_mmp_d` in `UPLOAD_README.txt` was cross-checked against the returned R04 Slurm metadata (`appVersion=ls-dyna_R15_mmp_d`); the `mmp` spelling is the SCNet catalog identifier, while local result folders use `R15_mpp_d`.
31. Final terminology audit clarified that `qM/qP/qE` use the expected mass and associated `v0` scales while `qV_over_v0=qP/qM`, and replaced ambiguous experimental-V50 wording with deterministic numerical transition terminology. The preceding bundle was moved intact to `superseded_before_qv_wording_clarification_bundle/`, then the authoritative bundle was rebuilt; deck, design, case map, material, P1 file, package manifest, and static report remained unchanged.
32. Re-ran all `test_r05*.py` tests after that candidate-document change and bundle rebuild: 50/50 PASS in 46.927 s. Rechecked the new ZIP: CRC returned no bad member, all 9 names were unique and exactly matched the eight-file whitelist plus `candidate_bundle_manifest.json`, every embedded hash matched, and external size/hash binding passed.

Final boundary: R05 is prepared and statically validated only. LS-DYNA parse/runtime acceptance, D3PART readability, capture-ledger availability, mesh consistency, physical validity, P0 response admission, and P1 admission all remain pending new returned results.
