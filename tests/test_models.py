from evals.models import CANDIDATES, MIXED_GENERATORS, resolve_models


def test_resolve_models():
    assert resolve_models(None, "claude-opus-5-5") == ["claude-opus-5-5"]
    assert resolve_models("  ", "x") == ["x"]
    assert resolve_models("candidates", "x") == list(CANDIDATES)
    assert resolve_models("mixed", "x") == MIXED_GENERATORS
    assert resolve_models("a, b ,a,,c", "x") == ["a", "b", "c"]


def test_presets_are_diverse():
    labs = {m.split("-")[0] for m in CANDIDATES}
    assert {"claude", "gpt", "gemini", "kimi"} <= labs
    assert len({m.split("-")[0] for m in MIXED_GENERATORS}) == len(MIXED_GENERATORS)
