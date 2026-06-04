# runs ONE implementation in its own subprocess.
# why a subprocess: ours/ and yazin/ both have a top-level `shared` package, you
# cant import both in one process. so each worker puts one project root on the
# path, loads its run.py once (the nemo load is the slow bit), then streams
# predictions out as json lines.
#
# the annoying part: run.py and nemo print to stdout, which would garbage up our
# json. so we dup the real stdout to a private handle and point fd 1 at stderr.
# json results go out the private handle, all the library noise goes to stderr.
#
# stdin: json {project_root, run_py, samples:[{id,path,surah,ayah},...]}
# stdout: one json line per sample. stderr: "PROG done total elapsed" + noise
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")


def main():
    # 1. Read the request BEFORE touching the fds.
    req = json.load(sys.stdin)

    # 2. Carve out a private results channel = a dup of the real stdout (fd 1),
    #    then redirect fd 1 -> fd 2 so prints/NeMo logs land on stderr, not here.
    results = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    # Re-point Python's sys.stdout at the (now stderr-backed) fd 1 so any
    # module-level print() is captured as noise rather than crashing the parser.
    sys.stdout = os.fdopen(1, "w", closefd=False)

    project_root = req["project_root"]
    run_py = req["run_py"]
    samples = req["samples"]

    sys.path.insert(0, project_root)

    total = len(samples)
    print(f"LOAD start total={total}", file=sys.stderr, flush=True)

    spec = importlib.util.spec_from_file_location("impl_run", run_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        try:
            r = mod.predict(s["path"])
            out = {
                "id": s["id"],
                "surah": int(r.get("surah", 0)),
                "ayah": int(r.get("ayah", 0)),
                "ayah_end": r.get("ayah_end"),
            }
        except Exception as e:  # noqa: BLE001
            out = {"id": s["id"], "surah": 0, "ayah": 0, "ayah_end": None,
                   "error": str(e)[:200]}
        results.write(json.dumps(out) + "\n")
        results.flush()
        print(f"PROG {i} {total} {time.time()-t0:.1f}", file=sys.stderr, flush=True)

    print(f"DONE total={total} elapsed={time.time()-t0:.1f}", file=sys.stderr, flush=True)
    results.close()


if __name__ == "__main__":
    main()
