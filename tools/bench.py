"""Run the public set and report metrics by scenario.

LANE C OWNS THIS FILE, and owns it first -- the other two lanes measure with
it, so it needs to land on main ahead of everything else.

    python3 -m tools.bench                     # metrics + first-hit histogram
    python3 -m tools.bench --failures 5        # dump failed session transcripts
    python3 -m tools.bench --transcript 3      # dump any 3 sessions, hit or miss
    python3 -m tools.bench --depth             # where the target really ranks
    python3 -m tools.bench --compare a.json b.json
    python3 -m tools.bench --sweep TJ_OPEN_QUESTION_BASELINE=3,4,5 TJ_OPEN_QUESTION_DECAY=0.5,0.65,0.8

The transcript/depth modes re-run the public set through a local copy of the
evaluator's session loop (`replay`) so we can watch the dialogue. That loop is
a line-for-line mirror of `evaluator.evaluate`; `--verify` asserts the two
still agree, so a future evaluator change cannot silently rot the diagnostics.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent

SCENARIOS = ("buying", "browsing", "intent_override", "boundary")


# ---------------------------------------------------------------- scoring ---

def technical_score(hit_rate: float, mrr: float, mttc: float) -> tuple[float, float]:
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency, efficiency


# ---------------------------------------------------------------- latency ---

class TimedAgent:
    """Wall-clock proxy around the real Agent.

    The submission rules require a latency disclosure, and the evaluator is
    off-limits -- so time the calls from outside instead of instrumenting the
    scored loop. Every attribute that is not `reset`/`respond` falls through,
    so the evaluator cannot tell the difference.
    """

    def __init__(self, agent) -> None:
        self._agent = agent
        self.turn_ms: list[float] = []
        self.reset_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        started = time.perf_counter()
        self._agent.reset(session_id, user_profile)
        self.reset_ms.append((time.perf_counter() - started) * 1000.0)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        response = self._agent.respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - started) * 1000.0)
        return response

    def __getattr__(self, name):
        return getattr(self._agent, name)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. No numpy dependency in the scored path."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * len(ordered) + 0.5)) - 1))
    return ordered[index]


def latency_summary(turn_ms: list[float], load_seconds: float) -> dict:
    if not turn_ms:
        return {}
    # The first turn of the whole run pays a one-off index warm-up that no
    # later turn repeats. Report it separately rather than letting one outlier
    # distort the steady-state percentiles.
    cold = turn_ms[0]
    warm = turn_ms[1:] or turn_ms
    return {
        "turns_measured": len(turn_ms),
        "cold_start_load_seconds": round(load_seconds, 3),
        "first_turn_ms": round(cold, 2),
        "mean_ms": round(statistics.fmean(warm), 2),
        "p50_ms": round(_percentile(warm, 0.50), 2),
        "p95_ms": round(_percentile(warm, 0.95), 2),
        "p99_ms": round(_percentile(warm, 0.99), 2),
        "max_ms": round(max(warm), 2),
        "total_seconds": round(sum(turn_ms) / 1000.0, 2),
    }


# ------------------------------------------------------------------- runs ---

def run(catalog: str, dataset: str, output: Optional[str] = None) -> dict:
    """The scored path: the real evaluator, untouched."""
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    started = time.perf_counter()
    agent = TimedAgent(Agent(catalog))
    load_seconds = time.perf_counter() - started
    result = evaluate(agent, samples, catalog_ids, categories, products)
    latency = latency_summary(agent.turn_ms, load_seconds)
    if latency:
        result["latency_ms"] = latency
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def replay(
    catalog: str,
    dataset: str,
    depth: bool = False,
    depth_limit: int = 300,
    progress: bool = False,
) -> list[dict]:
    """Mirror of `evaluator.evaluate` that keeps the transcript.

    Every branch here matches the evaluator. If you change one, change it
    because the evaluator changed, and re-run --verify.
    """
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    agent = Agent(catalog)
    records: list[dict] = []

    for index, sample in enumerate(samples):
        if progress and index % 25 == 0:
            print(f"    replay {index}/{len(samples)}", file=sys.stderr)
        session_id = f"bench_{sample['sample_id']}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective, coarse_category(categories.get(target, [])), disclosed
        )

        turns: list[dict] = []
        hit_turn: Optional[int] = None
        best_rank: Optional[int] = None

        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

            record = {
                "turn": turn,
                "customer": user_message,
                "agent": response.get("message", ""),
                "ask_attribute": response.get("ask_attribute"),
                "target_in_top10": target in ranked,
                "counted": override_applied,
            }
            if depth:
                record["target_depth"] = _target_depth(agent, session_id, target, depth_limit)
            turns.append(record)

            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )

        state = agent._states.get(session_id)
        records.append({
            "sample_id": sample["sample_id"],
            "scenario_type": sample["scenario_type"],
            "hit": hit_turn is not None,
            "first_hit_turn": hit_turn,
            "best_rank": best_rank,
            "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            "target": target,
            "target_title": str(products.get(target, {}).get("title", ""))[:110],
            "intent_card": card,
            "turns": turns,
            "final_slots": _slot_snapshot(state),
            "unanswerable": sorted(state.unanswerable) if state else [],
            "asked": list(state.asked) if state else [],
        })
    return records


def _slot_snapshot(state) -> dict:
    if state is None:
        return {}
    snapshot: dict[str, dict] = {}
    for slot, values in state.slots.items():
        live = [v.value for v in values if v.polarity]
        dead = [v.value for v in values if not v.polarity]
        if live or dead:
            snapshot[slot] = {"active": live, "retracted": dead}
    return snapshot


def _target_depth(agent: Agent, session_id: str, target: str, limit: int) -> Optional[int]:
    """1-based position of the target in the FULL ranking, not just the top 10.

    This is the question 'would a deeper list have caught it?'. None means the
    retriever never surfaced it at all inside `limit`.
    """
    state = agent._states.get(session_id)
    if state is None:
        return None
    try:
        cands = agent.retriever.search(state, top_n=limit)
        cands = agent.retriever.rerank(cands, state)
    except TypeError:
        cands = agent.retriever.rerank(agent.retriever.search(state), state)
    except Exception:
        return None
    for position, candidate in enumerate(cands, start=1):
        if candidate.parent_asin == target:
            return position
    return None


# --------------------------------------------------------------- reporting ---

def _bar(count: int, width: int = 40, scale: int = 1) -> str:
    return "#" * min(width, max(0, count // max(1, scale)))


def report(result: dict, sessions: Optional[list[dict]] = None) -> None:
    sessions = sessions if sessions is not None else result.get("sessions", [])
    print(f"\n  TechnicalScore  {result['recommended_technical_score']:.4f}")
    print(f"  HitRate@10      {result['hit_rate_at_10']:.4f}")
    print(f"  MRR             {result['mrr']:.4f}")
    print(f"  MTTC            {result['mttc']:.3f}   (efficiency {result['efficiency']:.4f})")

    print(f"\n  {'scenario':16} {'n':>4} {'hit':>8} {'mrr':>8} {'mttc':>7} {'turns@hit':>10}")
    for name in SCENARIOS:
        m = result["scenario_metrics"].get(name)
        if not m:
            continue
        hits = [s for s in sessions if s["scenario_type"] == name and s["hit"]]
        mean_hit_turn = sum(s["first_hit_turn"] for s in hits) / len(hits) if hits else float("nan")
        print(f"  {name:16} {m['sample_count']:4} {m['hit_rate_at_10']:8.3f} "
              f"{m['mrr']:8.3f} {m['mttc']:7.2f} {mean_hit_turn:10.2f}")

    if sessions:
        _histogram(sessions)
        _rank_profile(sessions)
    _latency_report(result.get("latency_ms"))
    usage = result.get("reported_token_usage", {})
    print(f"\n  tokens          {usage.get('total_tokens', 0)}")


def _histogram(sessions: list[dict]) -> None:
    counts = collections.Counter(s["first_hit_turn"] for s in sessions)
    hits = [s for s in sessions if s["hit"]]
    print(f"\n  first-hit turn      (mean over hits only: "
          f"{sum(s['first_hit_turn'] for s in hits) / len(hits):.2f})" if hits
          else "\n  first-hit turn")
    cumulative = 0
    for turn in range(1, MAX_TURNS + 1):
        cumulative += counts.get(turn, 0)
        if counts.get(turn, 0) or cumulative < len(hits):
            print(f"    turn {turn:2}  {counts.get(turn, 0):4}  "
                  f"(cum {cumulative / len(sessions):5.1%})  {_bar(counts.get(turn, 0))}")
    misses = counts.get(None, 0)
    print(f"    miss     {misses:4}  ({misses / len(sessions):5.1%})       {_bar(misses)}")
    dead = sum(counts.get(t, 0) for t in range(5, MAX_TURNS + 1))
    if not dead and misses:
        print(f"    -> turns 5-{MAX_TURNS} produced ZERO hits: {misses} misses are "
              f"already permanent by turn 4.")


def _latency_report(latency: Optional[dict]) -> None:
    """The feasibility disclosure the submission rules ask for."""
    if not latency:
        return
    print(f"\n  latency per turn    ({latency['turns_measured']} turns measured, "
          f"cold start {latency['cold_start_load_seconds']:.1f}s)")
    print(f"    p50 {latency['p50_ms']:8.1f} ms      mean {latency['mean_ms']:8.1f} ms")
    print(f"    p95 {latency['p95_ms']:8.1f} ms      p99  {latency['p99_ms']:8.1f} ms")
    print(f"    max {latency['max_ms']:8.1f} ms      first turn (index warm-up) "
          f"{latency['first_turn_ms']:.1f} ms")


def _rank_profile(sessions: list[dict]) -> None:
    ranks = collections.Counter(s["best_rank"] for s in sessions if s["hit"])
    if not ranks:
        return
    total = sum(ranks.values())
    top1 = ranks.get(1, 0)
    print(f"\n  rank at first hit   (rank 1: {top1}/{total} = {top1 / total:.1%})")
    line = "    " + " ".join(f"{r}:{ranks.get(r, 0)}" for r in range(1, TOP_K + 1))
    print(line)


def depth_report(sessions: list[dict], limit: int) -> None:
    """How much hit rate a deeper recommendation list would buy."""
    misses = [s for s in sessions if not s["hit"]]
    if not misses:
        print("\n  no misses to analyse")
        return
    print(f"\n  DEPTH ANALYSIS -- {len(misses)} missed sessions, "
          f"target position in the final full ranking (limit {limit})")
    buckets = collections.Counter()
    for session in misses:
        depths = [t.get("target_depth") for t in session["turns"] if t.get("target_depth")]
        best = min(depths) if depths else None
        if best is None:
            buckets["never retrieved"] += 1
        elif best <= 10:
            buckets["1-10"] += 1
        elif best <= 20:
            buckets["11-20"] += 1
        elif best <= 50:
            buckets["21-50"] += 1
        elif best <= 100:
            buckets["51-100"] += 1
        else:
            buckets[f"101-{limit}"] += 1
    order = ["1-10", "11-20", "21-50", "51-100", f"101-{limit}", "never retrieved"]
    for key in order:
        if buckets.get(key):
            print(f"    {key:>16}  {buckets[key]:3}  {_bar(buckets[key])}")
    reachable = sum(buckets.get(k, 0) for k in ("11-20", "21-50", "51-100", f"101-{limit}"))
    print(f"\n    {reachable} of {len(misses)} misses ARE retrieved, just below rank 10.")
    print(f"    Paging the dead turns (5-{MAX_TURNS}) over them is worth up to "
          f"{reachable / max(1, len(sessions)):.1%} hit rate.")


def dump_transcripts(sessions: list[dict], count: int, only_failures: bool) -> None:
    pool = [s for s in sessions if not s["hit"]] if only_failures else list(sessions)
    label = "FAILED" if only_failures else "SESSION"
    print(f"\n  {'=' * 70}\n  {label} TRANSCRIPTS ({min(count, len(pool))} of {len(pool)})\n  {'=' * 70}")
    for session in pool[:count]:
        outcome = (f"HIT turn {session['first_hit_turn']} rank {session['best_rank']}"
                   if session["hit"] else "MISS")
        print(f"\n  [{session['sample_id']}] {session['scenario_type']}  --  {outcome}")
        print(f"  target  {session['target']}  {session['target_title']}")
        card = session.get("intent_card", {})
        print(f"  hard    {card.get('hard_constraints')}")
        print(f"  soft    {card.get('soft_preferences')}")
        for turn in session["turns"]:
            flag = ""
            if turn["target_in_top10"] and not turn["counted"]:
                flag = "   <- in top10 but override not yet applied, not counted"
            elif turn["target_in_top10"]:
                flag = "   <- HIT"
            depth = turn.get("target_depth")
            if depth and not turn["target_in_top10"]:
                flag = f"   (target at full-rank {depth})"
            print(f"    {turn['turn']:2} C: {turn['customer'][:150]}")
            print(f"       A: {turn['agent'][:150]}")
            print(f"       ask={turn['ask_attribute']}{flag}")
        print(f"  slots   {session['final_slots']}")
        print(f"  asked   {session['asked']}   unanswerable={session['unanswerable']}")


def metrics_from_sessions(sessions: list[dict]) -> dict:
    core = [
        {k: s[k] for k in ("sample_id", "scenario_type", "hit", "first_hit_turn",
                           "best_rank", "reciprocal_rank")}
        for s in sessions
    ]
    overall = metric_summary(core)
    score, efficiency = technical_score(overall["hit_rate_at_10"], overall["mrr"], overall["mttc"])
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for session in core:
        grouped[session["scenario_type"]].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
        "reported_token_usage": {"total_tokens": 0},
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": core,
    }


# ----------------------------------------------------------------- compare ---

def compare(left: str, right: str) -> None:
    a = json.loads(Path(left).read_text())
    b = json.loads(Path(right).read_text())
    name_a, name_b = Path(left).stem[:12], Path(right).stem[:12]
    print(f"\n  {'metric':24} {name_a:>12} {name_b:>12} {'delta':>10}")
    print(f"  {'-' * 60}")
    for key, label in (
        ("recommended_technical_score", "TechnicalScore"),
        ("hit_rate_at_10", "HitRate@10"),
        ("mrr", "MRR"),
        ("mttc", "MTTC (lower better)"),
    ):
        delta = b[key] - a[key]
        print(f"  {label:24} {a[key]:12.4f} {b[key]:12.4f} {delta:+10.4f}")

    print(f"\n  {'scenario':16} {'n':>4} {name_a:>10} {name_b:>10} {'delta':>10}")
    for name in SCENARIOS:
        ma, mb = a["scenario_metrics"].get(name), b["scenario_metrics"].get(name)
        if not ma or not mb:
            continue
        sa, _ = technical_score(ma["hit_rate_at_10"], ma["mrr"], ma["mttc"])
        sb, _ = technical_score(mb["hit_rate_at_10"], mb["mrr"], mb["mttc"])
        print(f"  {name:16} {ma['sample_count']:4} {sa:10.4f} {sb:10.4f} {sb - sa:+10.4f}")

    moved = _session_moves(a, b)
    if moved:
        print(f"\n  {len(moved)} sessions changed outcome")
        for sample_id, before, after in moved[:20]:
            print(f"    {sample_id:14} {before:>14}  ->  {after}")


def _session_moves(a: dict, b: dict) -> list[tuple[str, str, str]]:
    def described(result: dict) -> dict[str, str]:
        return {
            s["sample_id"]: (f"turn {s['first_hit_turn']} rank {s['best_rank']}"
                             if s["hit"] else "miss")
            for s in result.get("sessions", [])
        }

    before, after = described(a), described(b)
    return [(k, before[k], after[k]) for k in before if k in after and before[k] != after[k]]


# ------------------------------------------------------------------- sweep ---

def sweep(specs: list[str], catalog: str, dataset: str) -> None:
    """Grid over environment knobs, one subprocess per cell.

    A subprocess per cell costs an index rebuild but keeps the cells honest --
    module-level constants read their environment variable exactly once, at
    import, so an in-process sweep would silently measure the first cell over
    and over.
    """
    axes: list[tuple[str, list[str]]] = []
    for spec in specs:
        name, _, values = spec.partition("=")
        axes.append((name, [v for v in values.split(",") if v != ""]))
    names = [name for name, _ in axes]
    print(f"\n  SWEEP over {', '.join(names)}   "
          f"({len(list(itertools.product(*[v for _, v in axes])))} cells)")
    print(f"\n  {' '.join(f'{n:>22}' for n in names)} {'score':>9} {'hit':>7} {'mrr':>7} {'mttc':>7}")

    rows: list[tuple[tuple[str, ...], dict]] = []
    for combination in itertools.product(*[values for _, values in axes]):
        env = {**os.environ, **dict(zip(names, combination))}
        proc = subprocess.run(
            [sys.executable, "-m", "tools.bench", "--catalog", catalog,
             "--dataset", dataset, "--json", "--output", ""],
            env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  {' '.join(f'{c:>22}' for c in combination)}   FAILED")
            print(proc.stderr[-500:], file=sys.stderr)
            continue
        result = json.loads(proc.stdout)
        rows.append((combination, result))
        print(f"  {' '.join(f'{c:>22}' for c in combination)} "
              f"{result['recommended_technical_score']:9.4f} {result['hit_rate_at_10']:7.3f} "
              f"{result['mrr']:7.3f} {result['mttc']:7.2f}")

    if rows:
        best = max(rows, key=lambda row: row[1]["recommended_technical_score"])
        print(f"\n  best: {dict(zip(names, best[0]))} -> "
              f"{best[1]['recommended_technical_score']:.4f}")


# -------------------------------------------------------------------- main ---

def verify(catalog: str, dataset: str) -> None:
    """The replay loop must agree with the real evaluator, or it is lying."""
    print("  running real evaluator ...", file=sys.stderr)
    official = run(catalog, dataset)
    print("  running replay ...", file=sys.stderr)
    mirrored = metrics_from_sessions(replay(catalog, dataset))
    ok = True
    for key in ("hit_rate_at_10", "mrr", "mttc", "recommended_technical_score"):
        a, b = official[key], mirrored[key]
        flag = "OK " if abs(a - b) < 1e-9 else "DIFF"
        ok &= flag == "OK "
        print(f"  {flag} {key:30} evaluator={a:.6f}  replay={b:.6f}")
    print("\n  replay is faithful" if ok else "\n  REPLAY HAS DRIFTED FROM THE EVALUATOR")
    sys.exit(0 if ok else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Public-set benchmark")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json",
                        help="where to write results; empty string writes nothing")
    parser.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    parser.add_argument("--sweep", nargs="+", metavar="VAR=v1,v2",
                        help="grid over environment knobs, e.g. TJ_OPEN_QUESTION_BASELINE=3,4,5")
    parser.add_argument("--failures", type=int, default=0, metavar="N",
                        help="dump N failed session transcripts")
    parser.add_argument("--transcript", type=int, default=0, metavar="N",
                        help="dump N session transcripts, hit or miss")
    parser.add_argument("--only", metavar="SAMPLE_ID", help="dump one session by id")
    parser.add_argument("--scenario", choices=SCENARIOS, help="restrict transcripts to one scenario")
    parser.add_argument("--depth", action="store_true",
                        help="report where the target sits in the full ranking")
    parser.add_argument("--depth-limit", type=int, default=300)
    parser.add_argument("--verify", action="store_true",
                        help="check the replay loop still matches the evaluator")
    parser.add_argument("--json", action="store_true", help="print raw metrics json (for --sweep)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.sweep:
        sweep(args.sweep, args.catalog, args.dataset)
        return
    if args.verify:
        verify(args.catalog, args.dataset)
        return

    wants_replay = bool(args.failures or args.transcript or args.depth or args.only)

    if wants_replay:
        sessions = replay(args.catalog, args.dataset, depth=args.depth,
                          depth_limit=args.depth_limit, progress=True)
        result = metrics_from_sessions(sessions)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k != "sessions"}))
            return
        report(result, sessions)
        if args.depth:
            depth_report(sessions, args.depth_limit)
        pool = sessions
        if args.scenario:
            pool = [s for s in pool if s["scenario_type"] == args.scenario]
        if args.only:
            pool = [s for s in pool if s["sample_id"] == args.only]
            dump_transcripts(pool, len(pool), only_failures=False)
        elif args.failures:
            dump_transcripts(pool, args.failures, only_failures=True)
        elif args.transcript:
            dump_transcripts(pool, args.transcript, only_failures=False)
        return

    result = run(args.catalog, args.dataset, args.output or None)
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "sessions"}))
        return
    report(result)


if __name__ == "__main__":
    main()
