"""Reproduce every result and figure.  Usage:  cd src && python run_all.py   (prints the time taken by each part)."""
import subprocess, sys, time
for f in ("part3_theory.py", "part2_montecarlo.py", "part5_numerical_hjb.py", "part1_data_backtest.py", "part4_multiasset.py"):
    t = time.time(); print(f"== {f}", flush=True)
    subprocess.run([sys.executable, f], check=True, stdout=subprocess.DEVNULL)
    print(f"   ok ({time.time()-t:.0f}s)", flush=True)
