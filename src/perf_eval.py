"""
Script Usage:
    python batch_crossover.py
    python batch_crossover.py --batches 1 8 32 128 512 2048 --breakdown
    python batch_crossover.py --devices xpu --out xpu.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

SCRATCH = Path(__file__).resolve().parent
REPO    = Path('Set your system path') # Set your system path
for p in (str(REPO), str(SCRATCH)):
    if p not in sys.path:
        sys.path.insert(0, p)
        

from perf_eval_functions import (build_user_embedding_batch, load_serving,
                              resolve_device, sample_pool, sync)

EXPERIMENT_BATCHES = [1, 8, 32, 64, 128, 256, 512, 1024]

def measure(device_name, batches, movies, top_k, seed, min_iters, target_samples):
    # Measure per-batch latency (milliseconds)
    device = resolve_device(device_name)
    fs, model, all_ids, all_embs, ts = load_serving(device)
    _, pool = sample_pool(fs)
    rng = random.Random(seed)

    def one(reqs, dis):
        with torch.no_grad():
            U = build_user_embedding_batch(model, fs, reqs, dis, ts, device)
            '''
            all_embs @ U.T -> (9375,128) @ (128,B) → (9375, B). 단일 GEMM. 열 하나가 쿼리 하나의 전 영화 점수
            .topk(top_k, dim=0) -> 열마다 상위 10개[결과 (10, B)]
            .indices.T -> 행 하나가 영화 추천 결과를 나타냄            
            '''
            return (all_embs @ U.T).topk(top_k, dim=0).indices.T.cpu()

    out = {}

    for B in batches:
        reqs = [[(t, 5.0) for t in rng.sample(pool, movies)] for _ in range(B)]
        dis  = [[] for _ in range(B)]
        for _ in range(5):
            one(reqs, dis) # For warm-up
        sync(device) # Wait for all kernels in all streams on a device to complete.
        n = max(min_iters, int(target_samples / B))
        t0 = time.perf_counter()
        # n회를 돌려 총 시간을 n으로 나눕니다 (1회 measure만 반영할 경우, OS scheduling, cache 결과 등에 따라 튀는 값이 나올 수 있으므로)
        for _ in range(n):
            one(reqs, dis)
        sync(device)
        out[B] = (time.perf_counter() - t0) / n * 1000
    return out

def measure_breakdown(device_name, batches, movies, top_k, seed, min_iters, target_samples):
    """prepare / H2D / 연산 / D2H, 반환: {B: (prepare, h2d, cmp, d2h) ms}"""
    from perf_eval_functions import _genre_context

    device = resolve_device(device_name)
    fs, model, all_ids, all_embs, ts = load_serving(device)
    _, pool = sample_pool(fs)
    rng = random.Random(seed)

    n_genres = len(fs['genres_ordered'])
    t2m, m2i = fs['title_to_movieId'], fs['item_emb_movieId_to_i']
    pad = model.pad_idx

    def assemble_cpu(batch_liked):
        """build_user_embedding_batch 의 조립부와 동일 — CPU 텐서까지만."""
        ctxs, hists, rats, likeds = [], [], [], []
        for liked_w in batch_liked:
            ctxs.append(_genre_context(fs, liked_w, [], n_genres))
            lh = [(m2i[t2m[t]], w) for t, w in liked_w if t in t2m and t2m[t] in m2i]
            hists.append([h[0] for h in lh] or [pad])
            rats.append([h[1] for h in lh] or [0.0])
            likeds.append([h[0] for h in lh] or [pad])

        def pad_ids(seqs):
            L = max(len(s) for s in seqs)
            o = torch.full((len(seqs), L), pad, dtype=torch.long)
            for i, s in enumerate(seqs):
                o[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
            return o

        def pad_w(seqs):
            L = max(len(s) for s in seqs)
            o = torch.zeros(len(seqs), L)
            for i, s in enumerate(seqs):
                o[i, L - len(s):] = torch.tensor(s, dtype=torch.float)
            return o

        d = torch.full((len(batch_liked), 1), pad, dtype=torch.long)
        return (torch.tensor(ctxs, dtype=torch.float32), pad_ids(hists),
                pad_ids(likeds), d, pad_w(rats))

    out = {}
    for B in batches:
        reqs   = [[(t, 5.0) for t in rng.sample(pool, movies)] for _ in range(B)]
        ts_b   = ts.expand(B) if ts.numel() == 1 else ts
        for _ in range(5):
            dt = [x.to(device) for x in assemble_cpu(reqs)]
            with torch.no_grad(): # 본 로직은 추론만 수행하므로 모델의 기울기(gradient) 계산을 비활성화함.
                U = model.user_embedding(dt[0], dt[1], dt[2], dt[3], dt[4], ts_b)
                _ = (all_embs @ U.T).topk(top_k, dim=0).indices.T.cpu()
        sync(device)

        n = max(min_iters, int(target_samples / B))
        acc = [0.0, 0.0, 0.0, 0.0]
        for _ in range(n):
            a0 = time.perf_counter()
            cpu_t = assemble_cpu(reqs)                       # ① 파이썬 조립
            a1 = time.perf_counter()
            dev_t = [x.to(device) for x in cpu_t]            # ② H2D
            sync(device)
            a2 = time.perf_counter()
            with torch.no_grad():                            # ③ 연산
                U   = model.user_embedding(dev_t[0], dev_t[1], dev_t[2], dev_t[3], dev_t[4], ts_b)
                idx = (all_embs @ U.T).topk(top_k, dim=0).indices.T
            sync(device)
            a3 = time.perf_counter()
            _ = idx.cpu()                                    # ④ D2H
            sync(device)
            a4 = time.perf_counter()
            for i, d in enumerate((a1-a0, a2-a1, a3-a2, a4-a3)):
                acc[i] += d
        out[B] = tuple(x / n * 1000 for x in acc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', nargs='+', default=['cpu', 'xpu'], choices=['cpu', 'xpu'])
    ap.add_argument('--batches', nargs='+', type=int, default=EXPERIMENT_BATCHES)
    ap.add_argument('--movies', type=int, default=3, help='쿼리당 무작위 영화 수')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--min-iters', type=int, default=20, help='배치 크기별 최소 반복')
    ap.add_argument('--target-samples', type=int, default=3000,
                    help='배치 크기별 목표 총 쿼리 수 (반복 횟수 = 이 값 / 배치)')
    ap.add_argument('--torch-threads', type=int, default=None,
                    help='CPU intra-op 스레드 수 (미지정 시 torch 기본값)')
    ap.add_argument('--breakdown', action='store_true',
                    help='조립/H2D/연산/D2H 4단계로 분해 출력')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    batches = sorted(set(args.batches))
    print(f"배치 크기: {batches}")
    print(f"쿼리당 영화 {args.movies}편, top-{args.top_k}, "
          f"torch intra-op 스레드 {torch.get_num_threads()}\n")

    results, breakdowns = {}, {}
    for dev in args.devices:
        print(f"측정 중: {dev} ...", flush=True)
        if args.breakdown:
            bd = measure_breakdown(dev, batches, args.movies, args.top_k,
                                   args.seed, args.min_iters, args.target_samples)
            breakdowns[dev] = bd
            results[dev] = {B: sum(v) for B, v in bd.items()}
        else:
            results[dev] = measure(dev, batches, args.movies, args.top_k,
                                   args.seed, args.min_iters, args.target_samples)

    # ── 디바이스별 상세 ───────────────────────────────────────────────────────
    for dev in args.devices:
        print(f"\n── {dev} ────────────────────────────────────────────────")
        if args.breakdown:
            print(f"{'배치':>6} {'조립':>9} {'H2D':>8} {'연산':>9} {'D2H':>8} "
                  f"{'합계':>9} {'QPS':>11} {'us/query':>10}")
            for B in batches:
                a, h, c, d = breakdowns[dev][B]
                tot = a + h + c + d
                print(f"{B:>6} {a:>8.3f}m {h:>7.3f}m {c:>8.3f}m {d:>7.3f}m "
                      f"{tot:>8.3f}m {B/tot*1000:>11,.0f} {tot*1000/B:>10.1f}")
        else:
            print(f"{'배치':>6} {'ms/batch':>10} {'QPS':>11} {'us/query':>10}")
            for B in batches:
                ms = results[dev][B]
                print(f"{B:>6} {ms:>10.3f} {B/ms*1000:>11,.0f} {ms*1000/B:>10.1f}")

    # ── 교차점 ────────────────────────────────────────────────────────────────
    if len(args.devices) == 2 and set(args.devices) == {'cpu', 'xpu'}:
        print(f"\n── 교차점 ──────────────────────────────────────────────")
        print(f"{'배치':>6} {'CPU QPS':>11} {'XPU QPS':>11} {'XPU/CPU':>9}  판정")
        crossover = None
        for B in batches:
            c_qps = B / results['cpu'][B] * 1000
            x_qps = B / results['xpu'][B] * 1000
            ratio = x_qps / c_qps
            if ratio >= 1.0 and crossover is None:
                crossover = B
            mark = 'XPU 우세' if ratio >= 1.0 else 'CPU 우세'
            star = '  ← 교차' if B == crossover else ''
            print(f"{B:>6} {c_qps:>11,.0f} {x_qps:>11,.0f} {ratio:>8.2f}x  {mark}{star}")

        print()
        cb = max(batches, key=lambda B: B / results['cpu'][B])
        xb = max(batches, key=lambda B: B / results['xpu'][B])
        print(f"  CPU 최고: {cb/results['cpu'][cb]*1000:>8,.0f} QPS (배치 {cb})")
        print(f"  XPU 최고: {xb/results['xpu'][xb]*1000:>8,.0f} QPS (배치 {xb})")
        if crossover:
            print(f"  교차점  : 배치 {crossover} 부터 XPU 우세")
        else:
            print(f"  교차점  : 측정 범위 내에 없음 (--batches 로 더 큰 배치를 시도하세요)")

    if args.out:
        Path(args.out).write_text(json.dumps({
            'batches': batches, 'movies': args.movies, 'top_k': args.top_k,
            'torch_threads': torch.get_num_threads(),
            'ms_per_batch': {d: {str(B): v for B, v in results[d].items()} for d in results},
            'qps': {d: {str(B): B / results[d][B] * 1000 for B in batches} for d in results},
            'breakdown_ms': ({d: {str(B): list(v) for B, v in breakdowns[d].items()}
                              for d in breakdowns} if breakdowns else None),
        }, indent=2))
        print(f"\n→ {args.out} 저장")


if __name__ == '__main__':
    main()