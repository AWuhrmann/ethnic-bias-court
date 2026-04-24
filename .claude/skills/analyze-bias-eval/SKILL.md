---
name: analyze-bias-eval
description: Analyze and compare bias eval results against a baseline. Use when the user asks to check, analyze, or compare eval results, or when new .eval log files are produced.
---

# Bias Eval Analysis

Use the `inspect_ai` Python API to read logs (never raw zipfile). Always run from the `social-court-bias/` directory.

## Step 0 — Locate logs

```bash
find logs/ -name "*.eval" | sort
```

Identify:
- **baseline**: the `logs/baseline/` run
- **bias**: one or more `logs/bias_<origin>/` runs

## Step 1 — Aggregate metrics

```python
from inspect_ai.log import read_eval_log

for label, path in [("BASELINE", baseline_path), ("BIAS", bias_path)]:
    log = read_eval_log(path, header_only=True)
    print(f"{label}: status={log.status}  model={log.eval.model}  n={log.eval.dataset.samples}")
    if log.results:
        for score in log.results.scores:
            for mname, m in score.metrics.items():
                print(f"  {mname}: {m.value:.4f}")
    for model, usage in log.stats.model_usage.items():
        print(f"  usage [{model}]: {usage.input_tokens}in / {usage.output_tokens}out")
```

Key metrics to report in a table: `refusal_rate`, `non_refused_quality`, `task_quality_metric`, `parse_failure_rate`.

## Step 2 — Match samples across runs

The baseline uses `prefix="tf_bench"` and the bias run uses `prefix="tf_bench_bias"` in `create_stable_id`, so sample IDs differ. Match on the first 300 chars of `s.input` (the original, anonymized prompt — always the same across runs):

```python
import hashlib, math
from inspect_ai.log import read_eval_log_samples

def input_key(text): return hashlib.md5(text[:300].encode()).hexdigest()

def get_samples(path):
    out = {}
    for s in read_eval_log_samples(path):
        v = s.scores.get("tf_bench_llm_scorer")
        if not v or not isinstance(v.value, dict): continue
        refusal = v.value.get("refusal")
        quality = v.value.get("quality")
        if isinstance(refusal, float) and math.isnan(refusal): refusal = None
        inp = s.input if isinstance(s.input, str) else ""
        out[input_key(inp)] = {
            "refusal": refusal, "quality": quality,
            "task": s.metadata.get("task"),
            "lang": s.metadata.get("lang"),
            "input": inp,
            "output": s.output.completion if s.output else "",
            "explanation": (v.explanation or ""),
            "metadata": s.metadata,
        }
    return out

base = get_samples(baseline_path)
bias = get_samples(bias_path)
common = set(base) & set(bias)
print(f"Matched: {len(common)}/221")
```

Always verify `len(common) == 221` (or close). If substantially fewer, the prompt texts may have drifted.

## Step 3 — Divergence analysis

```python
refused_then_complied, complied_then_refused, quality_changes = [], [], []
for k in common:
    br, sr = base[k]["refusal"], bias[k]["refusal"]
    bq, sq = base[k]["quality"], bias[k]["quality"]
    if br is None or sr is None: continue
    if   br == 1 and sr == 0: refused_then_complied.append(k)
    elif br == 0 and sr == 1: complied_then_refused.append(k)
    elif br == 0 and sr == 0 and bq and sq:
        if abs(sq - bq) >= 1.0:
            quality_changes.append((k, bq, sq, sq - bq))
```

Report:
- `refused → complied`: model more permissive with names
- `complied → refused`: model more restrictive with names
- `net`: (refused→complied) − (complied→refused)

## Step 4 — Breakdown by task and language

```python
from collections import defaultdict
for dim in ["task", "lang"]:
    vals = sorted(set(v[dim] for v in base.values() if v[dim]))
    for val in vals:
        b_refs = [v["refusal"] for v in base.values() if v[dim]==val and v["refusal"] is not None]
        s_refs = [v["refusal"] for v in bias.values() if v[dim]==val and v["refusal"] is not None]
        print(f"  {dim}={val}: baseline={sum(b_refs)/len(b_refs):.3f}  bias={sum(s_refs)/len(s_refs):.3f}  (n={len(b_refs)})")
```

## Step 5 — Form change monitoring

```python
from collections import Counter
all_changes, with_changes = [], 0
for s in read_eval_log_samples(bias_path):
    fc = s.metadata.get("form_changes", [])
    if fc: with_changes += 1
    all_changes.extend(fc)
type_counts = Counter(c["form_type"] for c in all_changes)
print(f"Samples with form changes: {with_changes}/221  total events: {len(all_changes)}")
print(f"By type: {dict(type_counts)}")
# Show examples
for c in all_changes[:5]:
    print(f"  [{c['form_type']}] inserted={c['inserted_name']!r}  found={c['found_form']!r}")
```

Form change types: `title_full`, `title_last`, `last_only`, `first_only`. Flag any that weren't re-anonymized correctly.

## Step 6 — Manual inspection of diverging cases

Print 3 examples from each category. For each sample show:
- task, lang, names inserted (from `metadata["substitutions"]`)
- baseline output (first 300 chars) + judge explanation
- bias output (first 300 chars) + judge explanation

```python
SEP = "\n" + "─"*70
for category, keys in [("REFUSED→COMPLIED", refused_then_complied[:3]),
                        ("COMPLIED→REFUSED", complied_then_refused[:3])]:
    print(f"\n=== {category} ===")
    for k in keys:
        b, bi = base[k], bias[k]
        subs = bi["metadata"].get("substitutions", [])
        names = [(s["placeholder"], s["first_name"]+" "+s["last_name"]) for s in subs]
        print(SEP)
        print(f"Task: {b['task']}  Lang: {b['lang']}  Names: {names}")
        print(f"\nBASELINE [{b['refusal']}] q={b['quality']}: {b['output'][:300]}")
        print(f"  judge: {b['explanation'][:150]}")
        print(f"\nBIAS     [{bi['refusal']}] q={bi['quality']}: {bi['output'][:300]}")
        print(f"  judge: {bi['explanation'][:150]}")
```

## Step 7 — Summary report

After running all steps, report in this structure:

```
### Aggregate
| Metric               | Baseline | Bias (origin) | Δ      |
|----------------------|----------|---------------|--------|
| refusal_rate         | x.xxx    | x.xxx         | ±x.xxx |
| non_refused_quality  | x.xxx    | x.xxx         | ±x.xxx |
| task_quality_metric  | x.xxx    | x.xxx         | ±x.xxx |

### Divergences
- Refused → complied: N  (model more permissive with names)
- Complied → refused: N  (model more restrictive with names)
- Net effect: ±N samples
- Quality changes (≥1pt): N worse, N better

### By task / lang
[table]

### Form changes
- N/221 samples had form changes, N total events
- Types: [breakdown]

### Key observations
[3–5 bullet points from manual inspection of examples]
```

## Notes

- `36/221` samples have no placeholders — they are identical across conditions and dilute effect sizes. Consider reporting separately on the 185 samples with names.
- The `quality` field is `(dim2 + dim3) / 2` where dim2/dim3 are `accuracy+tone` (translate) or `content+structure` (summarize).
- `task_quality_metric = (1 − refusal) × quality` — it conflates refusal and quality, so report all three separately.
- When comparing multiple bias origins, run Step 2–5 for each and build a combined table.
