class ComplianceRules:
    def __init__(self, required, violation_frames):
        self.required = set(required)
        self.limit = violation_frames
        self.missing = {}  # (track_id, item) -> consecutive frames missing

    def update(self, found):
        """Return {track_id: set of items in violation}."""
        status = {}
        for tid, labels in found.items():
            violations = set()
            for item in self.required:
                key = (tid, item)
                self.missing[key] = 0 if item in labels else self.missing.get(key, 0) + 1
                if self.missing[key] >= self.limit:
                    violations.add(item)
            status[tid] = violations
        return status