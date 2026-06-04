# us (c2c-direct) vs the baseline (nvidia-fastconformer), side by side.
# default tests = v1 + mixed100. you can pass others, see --tests.
#
# reports all four metrics with the span-aware scorer:
#   recall    = how many expected verses we found (in order)
#   precision = how many of our emitted verses were right
#   accuracy  = samples where the set matches exactly
#   seqacc    = samples where the ordered sequence matches exactly
#
# each side runs in its own subprocess, their `shared` packages clash otherwise.
# live progress bars, summary per test.
#
#   ./.venv/bin/python compare/run_comparison.py --tests v1 mixed100
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Set by main() in --json mode: emit(**obj) writes one JSON event line to the
# real stdout while all human prints/tqdm are redirected to stderr. In normal
# mode emit() is a no-op and prints go to stdout as usual.
EMIT = lambda **_: None

REPO = Path("/tmp/opencode/quran-research")
RECITERS_DIR = REPO / "data" / "reciters"
V1_CORPUS = REPO / "yazin" / "benchmark" / "test_corpus"
WORKER = Path("/home/cyber/Documents/repos/offline-tarteel/scripts/alamenium_worker.py")
# Use OUR venv (has onnxruntime + c2c-direct deps). The alamenium venv doesn't
# have onnxruntime and would crash on our c2c-direct-onnx impl.
PYTHON = Path("/home/cyber/Documents/repos/offline-tarteel/.venv/bin/python")
# Force CPU: cuDNN version mismatch on this box causes NeMo CUDA to fail with
# "GET was unable to find an engine". CPU is fast enough for these 54 samples.
PYTHON_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}

# Our experiments, mapped to alamenium's compare framework
OFFLINE_TARTEEL = Path("/home/cyber/Documents/repos/offline-tarteel")
IMPLS = {
    "OURS  c2c-direct-onnx (int8)": {
        "project_root": str(OFFLINE_TARTEEL),
        "run_py": str(OFFLINE_TARTEEL / "experiments" / "c2c-direct-onnx" / "run.py"),
    },
    "OURS  c2c-direct (NeMo .nemo)": {
        "project_root": str(OFFLINE_TARTEEL),
        "run_py": str(OFFLINE_TARTEEL / "experiments" / "c2c-direct" / "run.py"),
    },
    "OURS  c2c-direct-mixed (int4+int8)": {
        "project_root": str(OFFLINE_TARTEEL),
        "run_py": str(OFFLINE_TARTEEL / "experiments" / "c2c-direct-mixed" / "run.py"),
    },
    "YAZIN nvidia-fastconformer": {
        "project_root": str(OFFLINE_TARTEEL),
        "run_py": str(OFFLINE_TARTEEL / "experiments" / "nvidia-fastconformer" / "run.py"),
    },
}

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False


class C:
    R = "\033[0m"; B = "\033[1m"; DIM = "\033[2m"
    GRN = "\033[32m"; RED = "\033[31m"; YEL = "\033[33m"; CYN = "\033[36m"; MAG = "\033[35m"


_STDERR_TAIL: list[str] = []


# --------------------------------------------------------------------------
# Scoring (span-aware; mirrors ours/benchmark/scoring.py)
# --------------------------------------------------------------------------
def predict_to_emissions(pred: dict) -> list[tuple]:
    if not pred or pred.get("surah", 0) == 0:
        return []
    s = pred["surah"]; a = pred["ayah"]
    end = pred.get("ayah_end") or a
    return [(s, ay) for ay in range(a, end + 1)]


def score_one(expected: list[tuple], emitted: list[tuple]) -> dict:
    # the four metrics for one sample
    if not expected:
        return {"recall": 1.0, "precision": 1.0, "acc": 1.0, "seq": 1.0}
    if not emitted:
        return {"recall": 0.0, "precision": 0.0, "acc": 0.0, "seq": 0.0}
    matched = 0; pi = 0; mp = set()
    for e in expected:
        for j in range(pi, len(emitted)):
            if emitted[j] == e:
                matched += 1; mp.add(j); pi = j + 1; break
    recall = matched / len(expected)
    precision = len(mp) / len(emitted)
    acc = 1.0 if set(emitted) == set(expected) else 0.0
    seq = 1.0 if emitted == expected else 0.0
    return {"recall": recall, "precision": precision, "acc": acc, "seq": seq}


# --------------------------------------------------------------------------
# Sample sources
# --------------------------------------------------------------------------
def samples_from_v1():
    # the upstream v1 set. has expected_verses + also_accept + multi-verse spans
    man = json.load(open(V1_CORPUS / "manifest.json"))["samples"]
    out = []
    for s in man:
        expected = [(e["surah"], e["ayah"])
                    for e in s.get("expected_verses",
                                   [{"surah": s["surah"], "ayah": s["ayah"]}])]
        also = [[(e["surah"], e["ayah"]) for e in alt] for alt in s.get("also_accept", [])]
        out.append({
            "id": s["id"],
            "path": str(V1_CORPUS / s["file"]),
            "expected": expected,
            "also_accept": also,
            "source": s.get("source", "?"),
            "category": s.get("category", "?"),
        })
    return out, "v1 corpus (53, mixed personal+professional, multi-verse)"


def samples_from_corpus(cdir, build_hint):
    # load any cached corpus dir (personal/pro/mixed)
    man = cdir / "manifest.json"
    if not man.exists():
        sys.exit(f"corpus not built: {cdir}\n  run: {build_hint}")
    out = []
    for s in json.load(open(man))["samples"]:
        out.append({
            "id": s["id"],
            "path": str(cdir / s["file"]),
            "expected": [(e["surah"], e["ayah"])
                         for e in s.get("expected_verses",
                                        [{"surah": s["surah"], "ayah": s["ayah"]}])],
            "also_accept": [],
            "source": s.get("source", "?"),
            "category": s.get("category", "single"),
        })
    n_pers = sum(1 for s in out if s["source"] == "personal")
    n_pro = sum(1 for s in out if s["source"] == "professional")
    if n_pers and n_pro:
        desc = f"{len(out)} mixed ({n_pers} personal + {n_pro} professional)"
    elif n_pers:
        desc = f"{len(out)} personal (RetaSy crowdsourced)"
    elif n_pro:
        desc = f"{len(out)} professional"
    else:
        desc = f"{len(out)} samples"
    return out, desc


def samples_from_reciters(n, seed):
    # n random single verses spread across the 20 pro reciters
    reciters = sorted(d for d in RECITERS_DIR.iterdir() if d.is_dir())
    rng = random.Random(seed)
    pool = []
    for d in reciters:
        for mp3 in d.glob("*.mp3"):
            stem = mp3.stem
            if len(stem) == 6 and stem.isdigit():
                pool.append((d.name, int(stem[:3]), int(stem[3:]), str(mp3)))
    rng.shuffle(pool)
    out = []
    for rec, s, a, p in pool[:n]:
        out.append({
            "id": f"{rec}/{s:03d}{a:03d}",
            "path": p,
            "expected": [(s, a)],
            "also_accept": [],
            "source": rec,
            "category": "single",
        })
    return out, f"{len(out)} random verses across {len(reciters)} professional reciters"


# --------------------------------------------------------------------------
# Run one implementation in a subprocess with a live progress bar
# --------------------------------------------------------------------------
def run_impl(label, cfg, samples, progress_cb=None):
    worker_samples = [{"id": s["id"], "path": s["path"]} for s in samples]
    payload = json.dumps({"project_root": cfg["project_root"],
                          "run_py": cfg["run_py"], "samples": worker_samples})
    proc = subprocess.Popen(
        [str(PYTHON), str(WORKER)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, env=PYTHON_ENV,
    )
    proc.stdin.write(payload); proc.stdin.close()

    total = len(samples)
    bar = tqdm(total=total, desc=f"{C.CYN}{label}{C.R}", unit="clip",
               ncols=90, colour="cyan") if HAVE_TQDM else None
    if not HAVE_TQDM:
        print(f"  {label}: running {total} clips...", flush=True)

    preds = {}
    import threading
    _STDERR_TAIL.clear()

    def pump_progress():
        last = 0
        for line in proc.stderr:
            raw = line.rstrip("\n"); line = raw.strip()
            if line.startswith("PROG"):
                _, done, tot, _el = line.split(); done = int(done)
                if bar:
                    bar.update(done - last)
                elif done % max(1, total // 20) == 0:
                    print(f"    {label}: {done}/{tot}", flush=True)
                last = done
                if progress_cb:
                    progress_cb(done, total)
            elif line.startswith("LOAD"):
                if bar:
                    bar.set_postfix_str("loading model...")
            elif line and not any(k in line for k in
                                  ("NeMo", "rank=0", "OneLogger", "exporter",
                                   "Redirects", "megatron", "Setting error",
                                   "PySoundFile", "audioread", "Deprecated",
                                   "librosa", "FutureWarning", "UserWarning",
                                   "warnings.warn", "will be removed")):
                _STDERR_TAIL.append(raw + "\n")

    th = threading.Thread(target=pump_progress, daemon=True)
    th.start()

    for line in proc.stdout:
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        preds[r["id"]] = r
    rc = proc.wait()
    th.join(timeout=2)
    if bar:
        bar.n = total; bar.refresh(); bar.close()
    if rc != 0 and len(preds) < total:
        print(f"{C.RED}  {label}: worker exited rc={rc} with {len(preds)}/{total} "
              f"predictions. Last stderr:{C.R}\n{''.join(_STDERR_TAIL[-12:])}")
    return preds


def evaluate(samples, preds):
    # add up the four metrics over all samples, handling also_accept per sample
    agg = defaultdict(float)
    by_source = defaultdict(lambda: defaultdict(float))
    by_source_n = defaultdict(int)
    detail = {}
    for s in samples:
        p = preds.get(s["id"], {})
        emitted = predict_to_emissions(p)
        sc = score_one(s["expected"], emitted)
        for alt in s["also_accept"]:
            asc = score_one(alt, emitted)
            if asc["seq"] > sc["seq"]:
                sc = asc
        for k in ("recall", "precision", "acc", "seq"):
            agg[k] += sc[k]
            by_source[s["source"]][k] += sc[k]
        by_source_n[s["source"]] += 1
        detail[s["id"]] = (sc, emitted)
    n = len(samples)
    return {
        "n": n,
        "recall": agg["recall"] / n, "precision": agg["precision"] / n,
        "acc": agg["acc"] / n, "seq": agg["seq"] / n,
        "by_source": {src: {k: by_source[src][k] / by_source_n[src]
                            for k in ("recall", "precision", "acc", "seq")}
                      for src in by_source},
        "by_source_n": dict(by_source_n),
        "detail": detail,
    }


def run_test(name, samples, desc, seed):
    print(f"\n{C.B}{'='*82}{C.R}")
    print(f"{C.B}  TEST: {name}   ({desc}){C.R}")
    print(f"{C.B}{'='*82}{C.R}\n")
    EMIT(event="test_start", test=name, desc=desc, n=len(samples),
         impls=list(IMPLS))

    results = {}; timings = {}
    for label, cfg in IMPLS.items():
        EMIT(event="impl_start", test=name, impl=label)
        t0 = time.time()
        preds = run_impl(label, cfg, samples,
                         progress_cb=lambda d, t, _l=label: EMIT(
                             event="progress", test=name, impl=_l, done=d, total=t))
        timings[label] = time.time() - t0
        results[label] = evaluate(samples, preds)
        EMIT(event="impl_done", test=name, impl=label, time_s=timings[label])

    print(f"\n  {C.B}{'Implementation':<22}{'Recall':>9}{'Prec':>9}{'Acc':>9}"
          f"{'SeqAcc':>9}{'Time':>9}{'ms/clip':>9}{C.R}")
    print(f"  {'-'*76}")
    for label in IMPLS:
        r = results[label]; wt = timings[label]
        col = C.GRN if r["recall"] >= 0.95 else (C.YEL if r["recall"] >= 0.85 else C.RED)
        print(f"  {label:<22}{col}{r['recall']:>8.1%}{C.R}{r['precision']:>9.1%}"
              f"{r['acc']:>9.1%}{r['seq']:>9.1%}{wt:>8.1f}s{wt/max(1,r['n'])*1000:>8.0f}")

    # Head-to-head: pairwise for any N impls (was hardcoded to 2 in alamenium's
    # original). We compute it for the first two impls only to preserve the
    # alamenium-style output, and add a "all" tally for the full matrix.
    impls_list = list(IMPLS)
    if len(impls_list) >= 2:
        la, lb = impls_list[0], impls_list[1]
        da, db = results[la]["detail"], results[lb]["detail"]
        a_ok = lambda i, d: d[i][0]["seq"] == 1.0
        both = sum(1 for i in da if a_ok(i, da) and a_ok(i, db))
        neither = sum(1 for i in da if not a_ok(i, da) and not a_ok(i, db))
        a_only = sum(1 for i in da if a_ok(i, da) and not a_ok(i, db))
        b_only = sum(1 for i in da if not a_ok(i, da) and a_ok(i, db))
        print(f"\n  {C.B}Head-to-head (exact sequence, {la} vs {lb}){C.R}")
        print(f"    both correct : {C.GRN}{both}{C.R}    both wrong : {C.RED}{neither}{C.R}"
              f"    {la} only : {C.CYN}{a_only}{C.R}    {lb} only : {C.MAG}{b_only}{C.R}")

    # Per-source breakdown (only interesting on mixed v1).
    first = impls_list[0]
    if len(results[first]["by_source"]) > 1:
        print(f"\n  {C.DIM}per-source recall (n) -- {first}:{C.R}")
        for src in sorted(results[first]["by_source"]):
            print(f"    {src:<14} n={results[first]['by_source_n'][src]:<4} ", end="")
            for label in impls_list:
                r = results[label]["by_source"][src]["recall"]
                print(f"  {label}={r:.0%}", end="")
            print()

    out = REPO / "compare" / f"results_{name}_seed{seed}.json"
    summary = {
        "test": name, "desc": desc, "seed": seed, "n": len(samples),
        "results": {label: {k: results[label][k]
                            for k in ("n", "recall", "precision", "acc", "seq",
                                      "by_source", "by_source_n")}
                    | {"wall_time_s": timings[label]} for label in IMPLS},
        "head_to_head": {"both_correct": both, "both_wrong": neither,
                         f"{la}_only": a_only, f"{lb}_only": b_only},
    }
    json.dump(summary, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"\n  {C.DIM}saved -> {out}{C.R}")
    EMIT(event="test_done", **summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", nargs="+", default=["v1", "mixed100"],
                    help="any of: v1 | mixed<N> | personal<N> | pro<N> "
                         "(cached corpora) | <int> (random professional-only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true",
                    help="stream machine-readable JSON events on stdout (for the "
                         "local web UI); human output + progress bars go to stderr")
    a = ap.parse_args()

    if a.json:
        # Hand the real stdout to EMIT only; send every human print()/tqdm bar to
        # stderr so the JSON event stream stays clean and parseable line-by-line.
        global EMIT
        _fh = os.fdopen(os.dup(1), "w")
        sys.stdout = sys.stderr

        def EMIT(**obj):  # noqa: F811
            _fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            _fh.flush()

    if not HAVE_TQDM:
        print("note: tqdm not found, progress will just be periodic prints")

    # cached-corpus test types -> (dir name, build hint)
    CACHED = {
        "mixed": lambda n: (f"corpus_mixed_{n}",
                            f"./.venv/bin/python compare/build_mixed_corpus.py --n {n}"),
        "personal": lambda n: (f"corpus_personal_{n}",
                               f"./.venv/bin/python compare/build_mixed_corpus.py --personal {n}"),
        "pro": lambda n: (f"corpus_pro_{n}",
                          f"./.venv/bin/python compare/build_mixed_corpus.py --professional {n}"),
    }

    for t in a.tests:
        if t == "v1":
            samples, desc = samples_from_v1()
            run_test("v1", samples, desc, a.seed)
            continue
        matched = False
        for prefix, resolve in CACHED.items():
            if t.startswith(prefix) and t[len(prefix):].isdigit():
                n = int(t[len(prefix):])
                dirname, hint = resolve(n)
                samples, desc = samples_from_corpus(REPO / "compare" / dirname, hint)
                run_test(t, samples, desc, a.seed)
                matched = True
                break
        if matched:
            continue
        # bare integer -> random professional-only
        n = int(t)
        samples, desc = samples_from_reciters(n, a.seed)
        run_test(str(n), samples, desc, a.seed)

    EMIT(event="all_done")


if __name__ == "__main__":
    main()
