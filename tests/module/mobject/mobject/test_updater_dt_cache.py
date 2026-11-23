"""Updaters keep being called with or without dt after caching the check."""

from functools import partial

from manim import Mobject


def test_time_based_and_plain_updaters_are_dispatched_correctly():
    mob = Mobject()
    calls = []

    def with_dt(m, dt):
        calls.append(("dt", dt))

    def plain(m):
        calls.append(("plain",))

    mob.add_updater(with_dt)
    mob.add_updater(plain)
    mob.add_updater(partial(lambda m, dt, tag: calls.append((tag, dt)), tag="partial"))
    mob.add_updater(lambda m: calls.append(("lambda",)))
    for dt in (0.1, 0.2):
        mob.update(dt)
    assert calls == [
        ("dt", 0.1), ("plain",), ("partial", 0.1), ("lambda",),
        ("dt", 0.2), ("plain",), ("partial", 0.2), ("lambda",),
    ]
    assert mob.has_time_based_updater()
