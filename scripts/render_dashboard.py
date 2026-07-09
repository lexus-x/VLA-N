"""Fill n-vla-dashboard.template.html's <!--LIVE_*--> placeholders from the
snapshot files gen_dashboard.sh drops in a work dir, and write the result
atomically so a browser mid-auto-reload never sees a half-written file.
"""
import html
import json
import os
import sys


def read(path, default=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().replace("\r", "")
    except FileNotFoundError:
        return default


def main():
    template_path, out_path, work_dir, interval = sys.argv[1:5]

    ts = read(os.path.join(work_dir, "ts.txt")).strip()
    gpu_csv = read(os.path.join(work_dir, "gpu.csv")).strip()
    proc_txt = read(os.path.join(work_dir, "proc.txt")).strip()
    newest_name = read(os.path.join(work_dir, "newest_name.txt")).strip() or "none"
    results_raw = read(os.path.join(work_dir, "results.json"), "[]").strip()

    gpu = {"0": {"mem": "?", "util": "?"}, "1": {"mem": "?", "util": "?"}}
    for line in gpu_csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, used, total, util = parts
        if idx in gpu:
            gpu[idx] = {"mem": f"{used}/{total} MiB", "util": util}

    proc_count = sum(1 for l in proc_txt.splitlines() if "python" in l.lower())
    job_status = "job running" if proc_count > 0 else "idle"

    try:
        cells = json.loads(results_raw) if results_raw else []
    except json.JSONDecodeError:
        cells = []

    rows = []
    for c in cells:
        sr = c.get("success_rate")
        sr_pct = f"{sr * 100:.1f}%" if isinstance(sr, (int, float)) else "?"
        badge_class = "badge-good" if isinstance(sr, (int, float)) and sr > 0 else "badge-neutral"
        jerk = c.get("mean_jerk")
        jerk_s = f"{jerk:.3f}" if isinstance(jerk, (int, float)) else "?"
        hz = c.get("achieved_hz")
        hz_s = f"{hz:.2f}" if isinstance(hz, (int, float)) else "?"
        cell_id = html.escape(str(c.get("cell_id", "?")))
        rows.append(
            f'        <tr><td>{cell_id}</td>'
            f'<td><span class="badge {badge_class}">{sr_pct}</span></td>'
            f'<td>{jerk_s}</td><td>{hz_s}</td></tr>'
        )
    rows_html = "\n".join(rows) if rows else '        <tr><td colspan="4" class="muted">no results file found yet</td></tr>'

    tpl = read(template_path)
    subs = {
        "<!--LIVE_TS-->": html.escape(ts or "?"),
        "<!--LIVE_INTERVAL-->": str(interval),
        "<!--LIVE_GPU0_MEM-->": html.escape(gpu["0"]["mem"]),
        "<!--LIVE_GPU0_UTIL-->": html.escape(gpu["0"]["util"]),
        "<!--LIVE_GPU1_MEM-->": html.escape(gpu["1"]["mem"]),
        "<!--LIVE_GPU1_UTIL-->": html.escape(gpu["1"]["util"]),
        "<!--LIVE_PROC_COUNT-->": str(proc_count),
        "<!--LIVE_JOB_STATUS-->": job_status,
        "<!--LIVE_RESULTS_FILE-->": html.escape(newest_name),
        "<!--LIVE_RESULTS_ROWS-->": rows_html,
    }
    for k, v in subs.items():
        tpl = tpl.replace(k, v)

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(tpl)
    os.replace(tmp_path, out_path)


if __name__ == "__main__":
    main()
