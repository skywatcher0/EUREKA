"""Fit a single source-validated EUREKA expression."""
from eureka._method.candidate_selection import fit_eureka as fit_base
from eureka._method.families.eureka_interpolation import assess_interpolation
from eureka._method.families.eureka_radial import assess_radial
from eureka._method.source_selection import choose_source_program


def choose_consensus(base, candidates):
    supported = [(name, model, report) for name, model, report in candidates
                 if report["accepted"] and min(report["by_environment"]) > 0.]
    return choose_source_program(base, supported)


def fit_eureka(X_train, y_train, env_train, X_val, y_val, env_val, seed=0):
    args = X_train, y_train, env_train, X_val, y_val, env_val
    incumbent, prior = fit_base(*args, seed=seed)
    candidates = [(name, *fitter(incumbent, *args, seed=seed)) for name, fitter in
                  (("interpolation", assess_interpolation), ("projection", assess_radial))]
    chosen, decision = choose_consensus(incumbent, candidates)
    return chosen, {**prior, **decision, "incumbent_origin": prior["origin"],
                    "chosen_loeo": decision.get("source_score", prior["chosen_loeo"])}
