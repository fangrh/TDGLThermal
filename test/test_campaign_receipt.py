"""A saved PASS log must not bypass campaign protocol/provenance gates."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("campaign_receipt", Path(__file__).parents[1] / "bench" / "summarize_campaign.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


def make_campaign(root):
    jobs = []
    for current in (.06,.18,.27):
        for seed in range(2026110401,2026110409):
            stage=root/f"Je{current:g}"/f"seed{seed}"/"segment1"
            stage.mkdir(parents=True)
            paths=[]
            for backend,role in (("reference","corrected_reference"),("cpu_compatible","compatible_candidate")):
                data=dict.fromkeys(receipt.REQUIRED_MANIFEST,"fixture")
                data.update(backend=backend,role=role,status="Success",complete=True,
                    horizon=5000.,final_time=5000.,noise=.001,Nx=60,Ny=30,hx=.5,hy=.5,
                    tolerance=1e-4,skip_ratio=.9,snapshot_interval=1.,seed=seed,Je=current,
                    solver="Tsit5",precision="ComplexF64/Float64",save_fields=True,wall_seconds=1.,
                    rhs_calls=42,accepted_steps=6,rejected_steps=0)
                path=stage/f"{backend}.json";path.write_text(json.dumps(data));paths.append(str(path))
            (stage/"compare.stdout.log").write_text(json.dumps(dict(passed=True,errors=[],field_errors=dict.fromkeys(receipt.FIELDS,0.))))
            segment=dict(segment=1,returncode=0,timed_out=False,compare_returncode=0,compare_timed_out=False,
                compare_command=["compare","--reference",paths[0],"--candidate",paths[1]])
            jobs.append(dict(current=current,seed=seed,status="complete",segments=[segment]))
    (root/"campaign_summary.json").write_text(json.dumps(dict(jobs=jobs,counts=dict(complete=24,failed=0,incomplete=0,deferred=0))))


def test_receipt_rejects_empty_campaign(tmp_path):
    (tmp_path/"campaign_summary.json").write_text(json.dumps(dict(jobs=[],counts=dict(complete=0))))
    result=receipt.summarize(tmp_path)
    assert not result["valid"] and result["counts"]["matched_pairs"]==0


def test_receipt_rejects_wrong_horizon_despite_passed_comparison(tmp_path):
    make_campaign(tmp_path)
    assert receipt.summarize(tmp_path)["valid"]
    path=tmp_path/"Je0.06"/"seed2026110401"/"segment1"/"reference.json"
    data=json.loads(path.read_text());data.update(horizon=100.,final_time=100.)
    path.write_text(json.dumps(data))
    result=receipt.summarize(tmp_path)
    assert not result["valid"]
    assert any("protocol mismatch horizon" in x for x in result["failures"])


def test_receipt_rejects_changed_initial_state(tmp_path):
    make_campaign(tmp_path)
    path=tmp_path/"Je0.06"/"seed2026110401"/"segment1"/"cpu_compatible.json"
    data=json.loads(path.read_text());data["initial_hash"]="different"
    path.write_text(json.dumps(data))
    result=receipt.summarize(tmp_path)
    assert not result["valid"]
    assert any("paired provenance mismatch initial_hash" in x for x in result["failures"])
