from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
from analyze_ndr_interval import adjacent_gate, _stationarity


def points(values):
    return [{"values": list(v)} for v in values]


def test_positive_curve_fails():
    r = adjacent_gate(points([[0.0] * 8, [0.01] * 8, [0.02] * 8, [0.03] * 8]))
    assert r["resolved"] is False


def test_smooth_eight_seed_decrease_resolves():
    r = adjacent_gate(points([[0.03] * 8, [0.02] * 8, [0.01] * 8, [0.0] * 8]))
    assert r["resolved"] is True


def test_single_large_jump_does_not_resolve():
    r = adjacent_gate(points([[0.03] * 8, [0.03] * 8, [-0.02] * 8, [-0.02] * 8]))
    assert r["resolved"] is False


def test_stationarity_failure_blocks_gate():
    r = adjacent_gate(points([[0.03] * 8, [0.02] * 8, [0.01] * 8, [0.0] * 8]), stationarity_ok=False)
    assert r["resolved"] is False


def test_zero_stationarity_passes():
    assert _stationarity([0.0] * 8)["pass"] is True


def test_missing_seed_cannot_resolve_interval():
    result = adjacent_gate(points([[0.03] * 7, [0.02] * 7, [0.01] * 7, [0.0] * 7]))
    assert result["resolved"] is False


def test_total_gate_uses_joint_edge_upper_bounds():
    result = adjacent_gate(points([[0.03] * 8, [0.02] * 8, [0.01] * 8, [0.0] * 8]))
    for candidate in result["two_edge_candidates"]:
        assert candidate["simultaneous_total_upper"] == sum(result["edges"][i]["upper95"] for i in candidate["edges"])


def test_uncertain_adjacent_drops_do_not_inherit_exact_endpoint_resolution():
    alternating=[-0.005,0.005]*4
    result=adjacent_gate(points([[0.0]*8,[-0.006+x for x in alternating],[-0.012]*8,[-0.018+x for x in alternating]]))
    assert all(edge["upper95"] < 0 for edge in result["edges"])
    assert not result["resolved"]
