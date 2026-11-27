import numpy as np

from axis_ai.features import compute_features
from axis_ai.geometry import transform_person
from axis_ai.guide import GuideDecision, decide
from axis_ai.scorers.rank import RankScorer, make_pairs
from conftest import make_person


def _persons(cx, n, seed):
    rng = np.random.default_rng(seed)
    return [make_person(cx=cx + rng.normal(0, 0.02), top=0.2 + rng.normal(0, 0.02),
                        w=0.3 + rng.normal(0, 0.01), h=0.6 + rng.normal(0, 0.02)) for _ in range(n)]


def test_pairs_exclude_offscreen_and_have_expected_count():
    own = [make_person(cx=0.5) for _ in range(5)]
    others = [make_person(cx=0.8)]
    pairs = make_pairs(own, others, n_aug=20, seed=0)
    assert 5 < len(pairs) <= 5 * 20 + 5
    assert all(a.shape == (6,) and b.shape == (6,) and np.all(np.isfinite(b)) for a, b in pairs)
    # 화면 가장자리 인물은 변형이 많이 버려진다
    edge = [make_person(cx=0.95, w=0.1)]
    assert len(make_pairs(edge, [], n_aug=20, seed=0)) < 20
    # 다른 스타일 쌍은 원본 수만큼 더해진다
    assert len(make_pairs(own, others, n_aug=3, seed=0)) == len(make_pairs(own, [], n_aug=3, seed=0)) + 5


def test_pairs_deterministic():
    own = _persons(0.33, 4, 1)
    a = make_pairs(own, [], seed=3)
    b = make_pairs(own, [], seed=3)
    assert len(a) == len(b)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_rank_scorer_prefers_originals_on_heldout():
    own = _persons(0.33, 40, 1)
    others = _persons(0.66, 40, 2)
    pairs = make_pairs(own[:30], others[:30], n_aug=20, seed=0)
    s = RankScorer.fit("thirds", pairs, "v1", epochs=200, seed=0)
    rng = np.random.default_rng(5)
    wins = total = 0
    for p in own[30:]:
        base = s.score(compute_features(p))
        assert 0.0 <= base <= 1.0
        for _ in range(10):
            v = transform_person(p, rng.choice([-1, 1]) * rng.uniform(0.05, 0.25),
                                 rng.choice([-1, 1]) * rng.uniform(0.05, 0.25), rng.uniform(0.7, 1.4))
            if v is None:
                continue
            total += 1
            wins += base > s.score(compute_features(v))
    assert total > 0 and wins / total >= 0.8


def test_rank_scorer_save_load_roundtrip(tmp_path):
    own = _persons(0.33, 10, 1)
    s = RankScorer.fit("thirds", make_pairs(own, _persons(0.66, 10, 2), n_aug=5), "v7", epochs=20)
    path = tmp_path / "sub" / "thirds.pt"
    s.save(path)
    t = RankScorer.load(path)
    assert (t.style_id, t.version, t.kind) == ("thirds", "v7", "rank")
    for p in own[:3]:
        f = compute_features(p)
        assert abs(s.score(f) - t.score(f)) < 1e-6


def test_guide_works_with_rank_scorer():
    own = _persons(0.33, 10, 1)
    s = RankScorer.fit("thirds", make_pairs(own, [], n_aug=5), "v1", epochs=20)
    assert isinstance(decide(make_person(cx=0.6), s), GuideDecision)


def test_rank_scores_are_calibrated_so_guidance_reaches_ok():
    from axis_ai.eval.synthetic_guide import follow_guidance, perturb

    own = _persons(0.33, 40, 1)
    others = _persons(0.66, 40, 2)
    sc = RankScorer.fit("a", make_pairs(own[:30], others[:30], n_aug=20, seed=0), "t", epochs=150)
    held = [sc.score(compute_features(p)) for p in own[30:]]
    assert float(np.median(held)) >= 0.8  # 스타일에 맞는 원본은 높은 점수
    variants = [perturb(p, "RIGHT", 0.2) for p in own[30:]]
    assert float(np.median([sc.score(compute_features(v)) for v in variants])) < 0.6
    reached = [follow_guidance(v, sc) is not None for v in variants]
    assert sum(reached) >= 7  # 10장 중 7장 이상 OK에 닿는다


def test_rank_load_without_calibration_keeps_old_behaviour(tmp_path):
    import torch

    sc = RankScorer.fit("a", make_pairs(_persons(0.33, 10, 1), [], n_aug=5, seed=0), "t", epochs=20)
    path = tmp_path / "old.pt"
    sc.save(path)
    d = torch.load(path, weights_only=True)
    d.pop("calib")
    torch.save(d, path)
    old = RankScorer.load(path)
    f = compute_features(make_person(cx=0.33))
    x = (torch.tensor(f.vector(), dtype=torch.float32) - old.mean) / old.std
    with torch.no_grad():
        expected = float(torch.sigmoid(old.net(x.unsqueeze(0)))[0, 0])
    assert old.score(f) == expected
