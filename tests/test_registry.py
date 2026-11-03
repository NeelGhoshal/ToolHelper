import pytest

from toolhelper.efficiency.registry import ORDER, Inputs, build_strategies, parse_spec


def test_specs_parse_and_strategies_apply_in_working_order(dataset_index):
    assert parse_spec("e4:mode=repair,top_k=5") == ("E4", {"mode": "repair", "top_k": 5})

    class Run:
        run_id, stop_reason, tool_calls = "r1", "max_turns", []

    built = build_strategies(["E9", "E6:mode=top_n", "E4:mode=repair", "E3:modes=hints+rewrite", "E7:ttl_omop=60"],
                             Inputs(traces=[Run()], index=dataset_index))
    assert [s.name.split("_")[0] for s in built] == ["E3", "E4", "E7", "E6", "E9"]
    assert built[0].modes == {"hints", "rewrite"} and built[1].mode == "repair" and built[2].ttl["omop"] == 60
    assert sorted(s.name.split("_")[0] for s in built) == sorted(set(ORDER) & {"E3", "E4", "E6", "E7", "E9"})


def test_bad_specs():
    with pytest.raises(ValueError, match="unknown strategy"):
        parse_spec("E10")
    with pytest.raises(ValueError, match="each strategy once"):
        build_strategies(["E2", "E2:level=aggressive"], Inputs())
    with pytest.raises(ValueError, match="dataset index"):
        build_strategies(["E3"], Inputs())
