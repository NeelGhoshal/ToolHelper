from collections import Counter

from toolhelper.config import TASKS_FILE
from toolhelper.context.local_store import LocalStore
from toolhelper.experiments.experiment import folds, kind, load_experiment_config, write_experiment
from toolhelper.experiments.runner import load_tasks


def test_folds_are_disjoint_complete_and_balanced():
    tasks = load_tasks(TASKS_FILE)
    f = folds(tasks)
    assert set(f["A"]).isdisjoint(f["B"]) and sorted(f["A"] + f["B"]) == sorted(t.id for t in tasks)
    assert abs(len(f["A"]) - len(f["B"])) <= 1
    by_id = {t.id: t for t in tasks}
    for k in ("medical", "biomcp", "omop", "cross"):
        a, b = (sum(kind(by_id[i]) == k for i in f[x]) for x in "AB")
        assert abs(a - b) <= 1, k
    traps = Counter(x for x in "AB" for i in f[x] if by_id[i].traps)
    assert abs(traps["A"] - traps["B"]) <= 2
    assert folds(list(reversed(tasks))) == f  # deterministic


def test_experiment_config_and_thing():
    cfg = load_experiment_config()
    assert cfg.baseline_repeats == 3 and set(cfg.arms) == {f"E{i}" for i in range(1, 10)}
    store = LocalStore()
    thing = write_experiment(store, cfg, {"catalog_hash": "x"}, {"A": ["a"], "B": ["b"]})
    assert thing.name == "experiments/p10" and thing.data["status"] == "proposed"
