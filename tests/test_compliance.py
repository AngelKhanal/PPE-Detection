from src.compliance.association import associate
from src.compliance.rules import ComplianceRules
from src.detection.detector import Detection
from src.tracking.tracker import Track

PERSON = Track(1, (100, 100, 200, 400))  # height 300


def det(label, box):
    return Detection(label, 0.9, box)


def test_helmet_on_head_assigned():
    found = associate([PERSON], [det("helmet", (130, 100, 170, 140))])
    assert found == {1: {"helmet"}}


def test_vest_on_torso_assigned():
    found = associate([PERSON], [det("vest", (110, 180, 190, 300))])
    assert found == {1: {"vest"}}


def test_helmet_at_feet_not_assigned():
    found = associate([PERSON], [det("helmet", (130, 350, 170, 390))])
    assert found == {1: set()}


def test_unknown_class_ignored():
    found = associate([PERSON], [det("unknown", (110, 150, 190, 300))])
    assert found == {1: set()}


def test_smallest_containing_person_wins():
    big = Track(1, (0, 0, 400, 400))
    small = Track(2, (100, 100, 200, 300))
    found = associate([big, small], [det("helmet", (130, 100, 170, 140))])
    assert found[2] == {"helmet"}
    assert found[1] == set()


def test_violation_after_limit():
    r = ComplianceRules(["helmet"], 3)
    assert r.update({1: set()}) == {1: set()}
    assert r.update({1: set()}) == {1: set()}
    assert r.update({1: set()}) == {1: {"helmet"}}


def test_item_resets_counter():
    r = ComplianceRules(["helmet"], 3)
    r.update({1: set()})
    r.update({1: set()})
    r.update({1: {"helmet"}})
    assert r.update({1: set()}) == {1: set()}