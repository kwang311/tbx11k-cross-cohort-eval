# -*- coding: utf-8 -*-
"""Shared numeric layer for the Paper 2 statistics scripts (v2.0.0).

Why this file exists
--------------------
`stats_paper2.py` and `stats_cross_cells.py` had grown the same numeric blocks
(prediction loading with cross-seed label asserts, percentile bootstrap, AUC
variants, mean/SD over seeds, artefact writing). They were duplicated, so a
caliber change had to be made twice and could silently drift between the two
artefacts. Everything numeric now lives here once; the two entry scripts keep
only their parameter block and the calls that assemble their own payload.

Caliber (unchanged by this refactor, source of truth = the pre-refactor scripts)
-------------------------------------------------------------------------------
* bootstrap: 2,000 resamples, percentile method (2.5 / 97.5), ``RandomState(0)``,
  resampling unit = individual image (patients are not resampled).
* seed dispersion: mean +/- SD with ``ddof=1`` over the training seeds.
* restricted active-vs-latent score (PRIMARY): ``p_latent / (p_latent + p_active)``
  on the active/latent subset only; ``p_latent`` and ``p_latent - p_active`` are
  kept as reported alternatives.
* decision rule: ``argmax`` over the class posteriors of the seed ensemble, no
  tuned threshold.

Numeric-equivalence rules honoured here
---------------------------------------
* The RNG is consumed once per iteration for *every* iteration, including
  resamples that are later discarded, so skipping degenerate resamples never
  shifts the random stream.
* Probabilities are kept in the dtype they were stored in. The four-class
  ensemble mean is taken on the stored (float32) stack and cast to float64
  *after* averaging, exactly as before; per-seed metrics cast *before* the
  metric. Both orders are reproduced, so no leaf value moves.
* Each statistic that needs a CI opens its own ``RandomState(seed)``; callers
  therefore get the same intervals as the two-call pattern they replace.

Entry points: ``code/stats_paper2.py``, ``code/stats_cross_cells.py``.
"""
import glob
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# ---------------------------------------------------------------- constants
CLASS_NAMES = ["healthy", "sick_but_non-tb", "active_tb", "latent_tb"]
NUM_CLASSES = len(CLASS_NAMES)
BOOT_ITERS = 2000          # resamples per interval
BOOT_SEED = 0              # RandomState(0)
CI_PCT = (2.5, 97.5)       # percentile method, 95% CI
RESTRICTED_MODES = ("renorm", "latent_only", "difference")

# ------------------------------------------------------------------- loading


def load_run_files(pattern, expect=None):
    """Load per-seed prediction archives and their shared label vector.

    Input
        pattern : glob pattern of ``.npz`` files holding ``probs`` and ``labels``
                  (e.g. ``predictions/four_class/baseline_s*.npz``).
        expect  : optional expected file count; asserted when given.

    Output
        (runs, labels) where ``runs`` is the list of ``np.load`` handles in
        sorted filename order and ``labels`` is the label vector of the first
        run. Every run is asserted to carry an identical label vector.

    Caliber
        Same discovery, ordering and assertions as ``stats_paper2.py``
        (files4/filesa block) and ``stats_cross_cells.py`` (per-cell block).
        Arrays are returned untouched: the stored dtype (float32) is preserved
        so that later averaging happens in the original precision.
    """
    files = sorted(glob.glob(pattern))
    if expect is not None:
        assert len(files) == expect, f"expect {expect} npz, found {len(files)}"
    assert files, f"no predictions matched {pattern}"
    runs = [np.load(f) for f in files]
    labels = runs[0]["labels"]
    for r in runs[1:]:
        assert np.array_equal(r["labels"], labels), "val labels differ between seeds"
    return runs, labels


def load_binary_seed_scores(files, score_col=1):
    """Stack per-seed positive-class scores for one evaluation set.

    Input
        files     : sorted list of ``.npz`` paths (one per training seed), each
                    holding a ``(n, 2)`` ``probs`` matrix and ``labels``.
        score_col : column taken as the score (1 = positive class).
    Output
        (P_seeds, labels) with ``P_seeds`` of shape ``(n_seeds, n)`` float64 and
        ``labels`` the shared label vector.

    Caliber
        Cast to float64 *before* stacking, as in ``stats_cross_cells.py``; every
        seed is asserted to carry an identical label vector.
    """
    rows, labels = [], None
    for f in files:
        d = np.load(f)
        p, y = d["probs"].astype(np.float64)[:, score_col], d["labels"]
        if labels is None:
            labels = y
        else:
            assert np.array_equal(y, labels), f"label mismatch across seeds: {f}"
        rows.append(p)
    return np.vstack(rows), labels


def ensemble_probabilities(runs, key="probs"):
    """Seed-ensemble probability matrix.

    Input   : list of ``np.load`` handles (one per seed), each with ``key`` of
              shape ``(n, n_classes)``.
    Output  : ``(n, n_classes)`` float64 array, mean over seeds.
    Caliber : mean over the *stored* dtype, cast to float64 afterwards
              (``stats_paper2.py``: ``np.mean(np.stack(...), axis=0).astype``).
    """
    return np.mean(np.stack([r[key] for r in runs]), axis=0).astype(np.float64)


# ------------------------------------------------------------------- metrics


def macro_ovr_auc(probs, labels, n_classes=NUM_CLASSES):
    """Macro one-vs-rest AUC over a fixed class list (unweighted class mean)."""
    return float(roc_auc_score(labels, probs, multi_class="ovr", average="macro",
                               labels=list(range(n_classes))))


def weighted_ovr_auc(probs, labels, n_classes=NUM_CLASSES):
    """Frequency-weighted one-vs-rest AUC (same definition as the draft)."""
    pc, counts = [], []
    for c in range(n_classes):
        y = (labels == c).astype(int)
        pc.append(roc_auc_score(y, probs[:, c]))
        counts.append(int((labels == c).sum()))
    pc, counts = np.array(pc), np.array(counts, dtype=float)
    return float((pc * counts / counts.sum()).sum())


def per_class_ovr_auc(probs, labels, n_classes=NUM_CLASSES):
    """One-vs-rest AUC per class, keyed by ``CLASS_NAMES``."""
    out = {}
    for c in range(n_classes):
        y = (labels == c).astype(int)
        out[CLASS_NAMES[c]] = float(roc_auc_score(y, probs[:, c]))
    return out


def binary_auc(y, score):
    """Plain binary AUC."""
    return float(roc_auc_score(y, score))


def restricted_pair_auc(probs, labels, mode="renorm"):
    """Active-vs-latent AUC on the active/latent subset only.

    Input
        probs  : ``(n, 4)`` class posteriors of the seed ensemble.
        labels : ``(n,)`` class indices (2 = active, 3 = latent).
        mode   : ``"renorm"`` (PRIMARY, ``p_lat / (p_lat + p_act)``),
                 ``"latent_only"`` (``p_lat``) or ``"difference"``
                 (``p_lat - p_act``).

    Output
        (auc, y, score) with ``y`` = 1 for latent images and ``score`` the
        restricted score on the subset (so the caller can bootstrap both).

    Caliber
        Positive class = latent; a 1e-12 guard on the denominator keeps the
        primary score finite; ``labels`` outside {2, 3} are dropped.
    """
    mask = np.isin(labels, [2, 3])
    y = (labels[mask] == 3).astype(int)
    p_act, p_lat = probs[mask][:, 2], probs[mask][:, 3]
    if mode == "renorm":
        s = p_lat / (p_lat + p_act + 1e-12)
    elif mode == "latent_only":
        s = p_lat
    elif mode == "difference":
        s = p_lat - p_act
    else:
        raise ValueError(mode)
    return float(roc_auc_score(y, s)), y, s


def confusion_argmax(labels, probs, n_classes=NUM_CLASSES):
    """Argmax decision rule: confusion counts and per-class recall.

    Output
        (cm, recall, pred) with ``cm`` as a nested list (rows = true,
        columns = predicted, class order = ``CLASS_NAMES``) and ``recall``
        keyed by class name = cm[c][c] / row sum.
    Caliber
        No tuned threshold: predicted class = argmax over the posteriors.
    """
    pred = probs.argmax(1)
    cm = confusion_matrix(labels, pred, labels=list(range(n_classes))).tolist()
    recall = {CLASS_NAMES[c]: int(cm[c][c]) / int(sum(cm[c])) for c in range(n_classes)}
    return cm, recall, pred


def accuracy(labels, pred):
    """Plain accuracy."""
    return float(accuracy_score(labels, pred))


def mean_sd(values):
    """Mean and sample SD (ddof=1) over seeds, as reported in the draft."""
    return float(np.mean(values)), float(np.std(values, ddof=1))


def min_max(values):
    """Minimum and maximum over seeds."""
    return float(np.min(values)), float(np.max(values))


# ----------------------------------------------------------------- bootstrap


def boot_percentile_ci(fn, n, iters=BOOT_ITERS, seed=BOOT_SEED, pct=CI_PCT,
                       guard=None, on_error="skip"):
    """Percentile bootstrap CI of an image-level statistic.

    Input
        fn      : ``fn(idx) -> scalar``; ``idx`` indexes the evaluation set.
        n       : number of images resampled per iteration.
        iters   : resample count (2,000 for every published interval).
        seed    : ``RandomState(seed)``; a fresh stream per call.
        pct     : percentile pair (2.5 / 97.5).
        guard   : optional ``guard(idx) -> bool``; a True result discards the
                  resample *before* ``fn`` is called (label-degeneracy check).
        on_error: ``"skip"`` drops resamples where ``fn`` raises ``ValueError``
                  (the pre-refactor behaviour of ``stats_paper2.py``);
                  ``"raise"`` propagates instead (``stats_cross_cells.py``).

    Output
        (lo, hi, kept) — interval endpoints and the number of resamples that
        contributed.

    Caliber
        ``rng.choice(n, n, replace=True)`` is drawn for every iteration, so
        discarding a resample never shifts the random stream. Discarded
        resamples are counted in ``kept`` so coverage stays auditable.
    """
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(iters):
        idx = rng.choice(n, n, replace=True)
        if guard is not None and guard(idx):
            continue
        try:
            vals.append(fn(idx))
        except ValueError:
            if on_error == "raise":
                raise
    lo, hi = np.percentile(vals, [pct[0], pct[1]])
    return float(lo), float(hi), len(vals)


def boot_auc_ci(y, score, iters=BOOT_ITERS, seed=BOOT_SEED):
    """95% bootstrap CI of a binary AUC (fresh ``RandomState(seed)``)."""
    return boot_percentile_ci(lambda i: float(roc_auc_score(y[i], score[i])),
                              len(y), iters, seed)


def boot_auc_minus_chance_ci(y, score, iters=BOOT_ITERS, seed=BOOT_SEED):
    """95% bootstrap CI of AUC - 0.5 (same estimator, shifted by chance level)."""
    return boot_percentile_ci(lambda i: float(roc_auc_score(y[i], score[i])) - 0.5,
                              len(y), iters, seed)


def boot_ci_seedmean(P_seeds, labels, iters=BOOT_ITERS, seed=BOOT_SEED):
    """CI of the *mean-over-seeds* AUC (the estimator reported in Table 3).

    Input
        P_seeds : ``(n_seeds, n)`` per-seed probabilities for one matrix cell.
        labels  : ``(n,)`` binary labels of the target evaluation set.

    Output
        (lo, hi, kept).

    Caliber
        Each resample recomputes the AUC of *every* seed on that resample and
        averages over seeds, so the interval belongs to the same estimator as
        the point estimate (fix of 2026-10-07). Resamples with a single class
        present are discarded via ``guard`` (not via exception handling), and
        the RNG is consumed identically either way.
    """
    n = labels.shape[0]
    guard = lambda idx: labels[idx].min() == labels[idx].max()  # noqa: E731
    fn = lambda idx: float(np.mean([roc_auc_score(labels[idx], P_seeds[s][idx])  # noqa: E731
                                    for s in range(P_seeds.shape[0])]))
    return boot_percentile_ci(fn, n, iters, seed, guard=guard, on_error="raise")


def boot_ci_ensemble(probs, labels, iters=BOOT_ITERS, seed=BOOT_SEED):
    """CI of the AUC of the seed-averaged probability vector (reference only).

    Not used as the CI of the reported point estimate; kept so the historical
    ``ci95_ensemble_reference`` field keeps its meaning.
    """
    n = len(labels)
    guard = lambda idx: labels[idx].min() == labels[idx].max()  # noqa: E731
    fn = lambda idx: roc_auc_score(labels[idx], probs[idx])  # noqa: E731
    return boot_percentile_ci(fn, n, iters, seed, guard=guard, on_error="raise")


# -------------------------------------------------------------------- output


def write_json(path, payload):
    """Write an artefact as UTF-8 JSON, indent 2, non-ASCII kept verbatim."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_lines(path, lines):
    """Write a text artefact: one element per line, trailing newline."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
