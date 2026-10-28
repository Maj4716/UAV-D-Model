# Job 72197064 R05 failure worklog

Date: 2026-10-23 (Asia/Shanghai)

1. Treated `D:\Downloads\uav_hf_p0_plate_r05.zip` as immutable source evidence. Recorded size 3,164,360 bytes and SHA-256 `d543cf5f084b458b93c8428b33f103ed7e5b2372417eeb3f4a162ad526327706`.
2. Audited the ZIP before extraction: 14 unique flat entries, no traversal path, and `ZipFile.testzip()` returned no bad member.
3. Extracted an analysis copy to `analysis/72197064_r05_failure_20261023/raw_return/` and used it read-only; all derived evidence remains outside the source archive.
4. Compared the eight returned candidate files against `manual_upload_20261021_r05_candidate/`. Every size and SHA-256 matched byte for byte, including deck SHA-256 `6fbd753a113d0214e7ba4ff5136dc526d151e6f2c52c94e70399b29c4b9d3d0c`.
5. Re-ran `validate_p0_r05.py` against the returned input set with workspace-owned `TEMP/TMP`. Static validation returned PASS and was saved as `static_revalidation.json`.
6. Read the actual Slurm script, job log, stdout, stderr, host file, and `fort.59`. The earliest causal failure was the inability to connect to ANSYS license server `27890@login01`, FlexNet `-15,570`, before keyword parsing.
7. Confirmed LS-DYNA Error 70022 and application exit code 12. `std.err` and `fort.59` were empty; no `d3hsp`, `mes*`, solver `.out`, `binout`, `d3plot`, or `d3part` existed.
8. Compared with successful R04 Job 72095164. The same R15.0.2 build and ANSYS-only license mode successfully checked out `dyna` and 31 `FEAT_DYSMP` licenses on 2026-10-21, so expiry, deck identity, and intended runtime configuration are not supported as causes.
9. Found one documentation-only path spelling difference: candidate instructions say `LSDYNA`, while the portal consistently used `LS_DYNA` and found the deck. This did not cause Job 72197064 to fail; the retry checklist uses the observed platform path.
10. Decision: do not create R06 and do not change any physics or resource setting. Retry the exact frozen R05 candidate once after license recovery, in a new empty work directory. If `-15,570` repeats, stop and escalate to SCNet with both job IDs.
11. No P1 job, engine fitting, dataset generation, training, validation promotion, or sealed-Test access was performed.
