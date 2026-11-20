from evals.build_dataset import detokenize, usable
from evals.scorers import (
    ai_recall,
    calibration,
    concise,
    human_specificity,
    verdict_correct,
)

AI = {"label": "ai"}
HUMAN = {"label": "human"}


def out(verdict="ai", p=90, rendered="short reply"):
    return {"verdict": verdict, "ai_probability": p, "reasons": [], "rendered": rendered}


def scores(result):
    return {s.name: s.score for s in result}


def test_verdict_correct_and_committed():
    assert scores(verdict_correct({}, out("ai"), AI)) == {"verdict_correct": 1, "committed": 1}
    assert scores(verdict_correct({}, out("ai"), HUMAN)) == {"verdict_correct": 0, "committed": 1}
    assert scores(verdict_correct({}, out("unsure"), AI)) == {"verdict_correct": 0, "committed": 0}


def test_class_specific_scores_skip_other_class():
    assert ai_recall({}, out("ai"), AI).score == 1
    assert ai_recall({}, out("ai"), HUMAN) is None
    assert human_specificity({}, out("ai"), HUMAN).score == 0
    assert human_specificity({}, out("human"), AI) is None


def test_calibration_oracle_vs_coin_flip():
    assert calibration({}, out(p=100), AI).score == 1
    assert calibration({}, out(p=0), HUMAN).score == 1
    assert calibration({}, out(p=50), AI).score == 0.75
    assert calibration({}, out(p=0), AI).score == 0
    assert calibration({}, out(p=500), AI).score == 1  # clamped


def test_constant_ai_baseline_scores_half_on_balanced_data():
    labels = [AI, HUMAN] * 10
    accuracy = sum(verdict_correct({}, out("ai"), y)[0].score for y in labels) / len(labels)
    assert accuracy == 0.5


def test_concise_thresholds():
    assert concise({}, out(rendered="w " * 60)).score == 1
    assert concise({}, out(rendered="w " * 105)).score == 0.5
    assert concise({}, out(rendered="w " * 200)).score == 0


def test_hc3_detokenizer():
    raw = "You ca n't just do that ! Who 's next ( and why ) ? He said \" hello \" ."
    assert detokenize(raw) == "You can't just do that! Who's next (and why)? He said \"hello\"."


def test_usable_rejects_leftover_artifacts_and_bad_lengths():
    good = " ".join(["word"] * 50) + "."
    assert usable(good)
    assert not usable("too short.")
    assert not usable(good + " see URL_0 for more")
    assert not usable(good + " and this , that")
