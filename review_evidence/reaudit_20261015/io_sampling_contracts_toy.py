"""Artificial identity/mask/set contract checks; no domain data or simulation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT))
from damage_engine.sampling_audit import audit_sampling
from damage_engine.errors import InputValidationError


def rows():
    probability=[.5,.25,1.,.5]
    return [{"candidate_id":f"toy-{i}","lineage_id":f"lineage-{i}","epistemic_group_id":f"group-{i}",
             "role":"toy-role","stream":"toy-stream","stratum":"toy-stratum","split":"train",
             "selected":i!=1,"weight_mask":i!=3,"inclusion_probability":p,"design_weight_raw":1/p}
            for i,p in enumerate(probability)]


def main():
    gates={"positivity_min":.1,"min_ess_fraction":.5,"max_weight_cv":10.}
    latent=np.asarray([[.2],[.1],[.8],[.9]])
    result={"scope":"artificial arrays and candidate identities only","guards":{}}
    defects={
        "duplicate_candidate_id":lambda x:x[1].update(candidate_id=x[0]["candidate_id"]),
        "lineage_crosses_split":lambda x:x[1].update(lineage_id=x[0]["lineage_id"],split="test"),
        "epistemic_group_crosses_split":lambda x:x[1].update(epistemic_group_id=x[0]["epistemic_group_id"],split="test"),
        "probability_above_one":lambda x:x[1].update(inclusion_probability=1.1),
        "negative_weight":lambda x:x[1].update(design_weight_raw=-1.),
        "nonboolean_selected":lambda x:x[1].update(selected="false"),
    }
    for name,mutate in defects.items():
        current=deepcopy(rows())
        mutate(current)
        try:
            audit_sampling(current,latent,quality_gates=gates)
            result["guards"][name]=False
        except InputValidationError:
            result["guards"][name]=True
    for name,value in (("nan_latent",np.nan),("out_of_bounds_latent",1.01)):
        current=latent.copy()
        current[0,0]=value
        try:
            audit_sampling(rows(),current,quality_gates=gates)
            result["guards"][name]=False
        except InputValidationError:
            result["guards"][name]=True
    report=audit_sampling(rows(),latent,quality_gates=gates,
                          executed_candidate_ids=["toy-0","toy-2","toy-3"],
                          successful_candidate_ids=["toy-0","toy-2"],
                          admitted_candidate_ids=["toy-0"],refused_candidate_ids=["toy-3"])
    result["weighted_ks_matches_hand_calculation"]=bool(np.isclose(report["selected_weighted"]["ks_by_axis"][0],2/3-.2))
    result["ess_matches_hand_calculation"]=bool(np.isclose(report["effective_sample_size"],9/5))
    result["set_counts"]={name:report["sets"][name]["count"] for name in ("candidate_frame","selected","executed","successful","admitted","refused")}
    result["coverage_guaranteed"]=report["coverage_guaranteed"]
    result["strict_json_serialization_passed"]=bool(json.dumps(report,allow_nan=False))
    try:
        audit_sampling(rows(),latent,quality_gates=gates,executed_candidate_ids=["toy-1"])
        result["guards"]["executed_must_be_selected"]=False
    except InputValidationError:
        result["guards"]["executed_must_be_selected"]=True
    assert all(result["guards"].values())
    assert result["weighted_ks_matches_hand_calculation"] and result["ess_matches_hand_calculation"]
    Path(__file__).with_suffix(".json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"guards_passed":len(result["guards"]),"hand_calculations_passed":2,"real_domain_records_read":0}))


if __name__=="__main__":
    main()
