"""
STEP 5 (chart) - Render the sweep from the saved CSV.

THE WINDOWS DLL TRAP - worth reading before adding any plotting to this project.

  Running this project's interpreter by its full path - which is what an IDE, a
  scheduled task, or `"C:\\...\\envs\\hospital\\python.exe" script.py` all do -
  does NOT activate the conda environment. So `<env>/Library/bin` is missing from
  PATH, and that is where conda keeps matplotlib's native dependencies (freetype,
  libpng, zlib).

  Without it, matplotlib dies on a WINDOWS FATAL EXCEPTION (0xc06d007f, a C++
  exception from the DLL loader). It is NOT a Python exception: the process
  terminates instantly, try/except cannot catch it, and unflushed stdout is lost -
  so the symptom is a script that prints nothing and exits 127.

  How this was pinned down, because the first two diagnoses were both wrong:
    - looked like `ax.plot()` was fatal        -> wrong, it was whatever ran next
    - looked like `set_yticks()` was fatal     -> wrong, same reason
    - looked nondeterministic                  -> wrong
    - the real tell: `05_process_map.py` renders fine while a three-line
      matplotlib script fails 3 times out of 3. The difference is that 05 calls
      `ensure_graphviz()`, which prepends `<env>/Library/bin` to PATH for
      Graphviz - and incidentally fixes matplotlib too.

  So the fix below is not a workaround for a matplotlib bug; it is supplying the
  library search path that activating the environment would have supplied.

  FOR STEP 7: the Streamlit dashboard will hit exactly this if launched without
  activating the env. Either activate it, or call `ensure_dlls()` at startup.
"""

import os
import sys
from pathlib import Path


def ensure_dlls() -> None:
    """
    Put the conda environment's native library directory on the DLL search path.

    MUST run before matplotlib is imported, which is why the import sits below
    this function rather than at the top of the file.
    """
    lib = Path(sys.executable).parent / "Library" / "bin"
    if lib.exists():
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))


ensure_dlls()

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")                                 # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import pandas as pd                                   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "outputs" / "11_cashless_vs_selfpay.csv"
OUT_PNG = ROOT / "outputs" / "11_cashless_vs_selfpay.png"

DEADLINE_H = 3.0          # IRDAI/HLT/CIR/PRO/84/5/2024 clause 16(a)


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(
            f"Not found: {IN_CSV} - run: python src/11_cashless_vs_selfpay.py")

    df = pd.read_csv(IN_CSV)
    med = df.pivot(index="q", columns="payment_mode", values="median_h")
    breach = df.pivot(index="q", columns="payment_mode", values="breach_3h_pct")
    cash = df[df["payment_mode"] == "cashless"].set_index("q")
    xs = [float(q) for q in med.index]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6))

    # --- panel 1: median discharge cycle time -------------------------------
    for mode, colour in [("cashless", "#c0392b"), ("self_pay", "#2980b9")]:
        ax1.plot(xs, [float(v) for v in med[mode]], marker="o", color=colour,
                 linewidth=2.2, label=mode)
    ax1.plot([min(xs), max(xs)], [DEADLINE_H, DEADLINE_H], color="black",
             linestyle="--", linewidth=1.5,
             label=f"IRDAI deadline ({DEADLINE_H:.0f} h)")
    ax1.set_xlabel("document query probability  q   (UNSOURCED - swept)")
    ax1.set_ylabel("median discharge cycle time (hours)")
    ax1.set_title("Medically ready to actually leaving", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- panel 2: breach rate ------------------------------------------------
    for mode, colour in [("cashless", "#c0392b"), ("self_pay", "#2980b9")]:
        ax2.plot(xs, [float(v) for v in breach[mode]], marker="s", color=colour,
                 linewidth=2.2, label=mode)
    ax2.set_xlabel("document query probability  q   (UNSOURCED - swept)")
    ax2.set_ylabel("% breaching the 3-hour rule")
    ax2.set_title("Share breaching IRDAI clause 16(a)", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # --- panel 3: the decomposition the real log cannot do -------------------
    labels = [f"q={q:g}" for q in reversed(list(med.index))]
    work = [float(cash.loc[q, "mean_work_h"]) for q in reversed(list(med.index))]
    queue = [float(cash.loc[q, "mean_queue_h"]) for q in reversed(list(med.index))]
    ax3.barh(labels, work, color="#8e44ad", label="processing (too many STEPS)")
    ax3.barh(labels, queue, left=work, color="#f39c12",
             label="queueing (too little CAPACITY)")
    ax3.set_xlabel("mean hours per cashless patient")
    ax3.set_title("Cashless delay: almost entirely PROCESSING",
                  fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Discharge delay swept across the UNSOURCED query-loop probability q\n"
        "No single q is a finding: cashless is slower at EVERY value, and the "
        "delay is processing, not queueing",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
