import subprocess, sys, time
t0 = time.perf_counter(); prev = t0
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding='utf-8', errors='replace', bufsize=1)
for line in p.stdout:
    now = time.perf_counter()
    line = line.rstrip()
    if line.strip():
        print(f'[+{now-t0:7.1f}s Δ{now-prev:6.1f}s] {line}', flush=True)
        prev = now
p.wait()
print(f'[总耗时 {time.perf_counter()-t0:.1f}s] rc={p.returncode}')
