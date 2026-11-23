#!/usr/bin/env bash
# Run the proxy-convergence session's oracle scripts without clobbering its evidence.
# Each script writes into proxy_convergence_20261123/results/; originals are backed
# up first, every new output is copied here, and the originals are restored at the end.
set -u
ROOT="D:/Desktop/UAV新"
STUDY="$ROOT/artifacts/validation/proxy_convergence_20261123"
HERE="$ROOT/artifacts/validation/data_generation_fixes_20261123/peer_checks"
PY313="python"
PY312="/d/ProgramData/Anaconda/envs/pytorch312-lmdt/python.exe"
FILES="teacher_equivalence_check.json final_module_consistency.json reproduction_check.json"

for f in $FILES; do cp "$STUDY/results/$f" "$HERE/peer_original_$f"; done
cd "$STUDY/scripts" || exit 1

run() {  # tag interpreter script args... ; copies the script's output file
  local tag="$1" interp="$2" out="$3"; shift 3
  echo "== $tag ($($interp -c 'import sys,numpy;print(sys.version.split()[0],numpy.__version__)'))"
  "$interp" "$@" > "$HERE/$tag.log" 2>&1
  echo "exit=$? -> $tag"
  cp "$STUDY/results/$out" "$HERE/$tag.json"
}

run reproduction_check.py313 "$PY313" reproduction_check.json check_reproduction.py
run final_module_consistency_60.py313 "$PY313" final_module_consistency.json check_final_module_consistency.py 60
run teacher_equivalence_4.py312 "$PY312" teacher_equivalence_check.json check_teacher_equivalence.py 4
run teacher_equivalence_4.py313 "$PY313" teacher_equivalence_check.json check_teacher_equivalence.py 4

for f in $FILES; do cp "$HERE/peer_original_$f" "$STUDY/results/$f"; done
echo "originals restored"
echo ALL_DONE
