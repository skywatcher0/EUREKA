"""Select a source-validated augmentation of the incumbent expression."""


def choose_source_program(base, candidates):
    eligible = [(r["source_score"], p.count_unique_nodes(), name, p, r)
                for name,p,r in candidates if r["accepted"]]
    if not eligible:
        return base, {"origin":"base","accepted":False}
    _, _, name, program, report = min(eligible, key=lambda row: row[:3])
    return program, {**report,"origin":name}
