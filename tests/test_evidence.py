import numpy as np

from src.compliance.evidence import EvidenceLogger

FRAME = np.zeros((50, 50, 3), dtype=np.uint8)


def rows(logger):
    return logger.csv_path.read_text().strip().splitlines()[1:]


def test_logs_once_per_violation(tmp_path):
    log = EvidenceLogger(tmp_path)
    log.update(FRAME, {1: {"helmet"}})
    log.update(FRAME, {1: {"helmet"}})
    assert len(rows(log)) == 1
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_logs_again_after_recovery(tmp_path):
    log = EvidenceLogger(tmp_path)
    log.update(FRAME, {1: {"helmet"}})
    log.update(FRAME, {1: set()})
    log.update(FRAME, {1: {"helmet"}})
    assert len(rows(log)) == 2


def test_new_missing_item_logs_again(tmp_path):
    log = EvidenceLogger(tmp_path)
    log.update(FRAME, {1: {"helmet"}})
    log.update(FRAME, {1: {"helmet", "vest"}})
    assert len(rows(log)) == 2