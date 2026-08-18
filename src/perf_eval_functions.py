import argparse
import json
import queue
import random
import statistics
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path('/home/akssus12/fork/intelb70_recommendation_system')
sys.path.insert(0, str(REPO))

from src.inference import build_user_embedding
from src.llm_frontend import build_serving_model

SERVING    = REPO / 'serving'
MOVIES_CSV = REPO / 'data' / 'ml-32m' / 'movies.csv'

DISLIKED_MOVIE_VALUE = -2.0   # follow original repo's parameter

def resolve_device(name: str) -> torch.device:
    if name == 'cpu':
        return torch.device('cpu')
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        raise SystemExit(
            "XPU를 쓸 수 없습니다. 확인:\n"
            "  - torch가 XPU 빌드인가       : '+xpu' 포함 여부\n"
            "  - Level Zero 런타임이 있는가 : clinfo -l\n"
            "  - render 그룹에 속해 있는가  : id | grep render")
    return torch.device('xpu')

def sync(device: torch.device) -> None:
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

def load_serving(device: torch.device):
    fs  = torch.load(SERVING / 'feature_store.pt', weights_only=False)
    me  = torch.load(SERVING / 'movie_embeddings.pt', weights_only=False)
    cfg = fs['model_config']

    model = build_serving_model(fs, cfg)
    sd = torch.load(SERVING / 'model.pth', map_location='cpu', weights_only=True)
    for buf in ('genome_context_buffer', 'content_context_buffer', 'llm_feature_buffer'):
        sd.pop(buf, None)
    model.load_state_dict(sd)
    model.eval()

    all_ids  = list(me.keys())
    all_embs = torch.cat([me[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)
    ts = torch.bucketize(torch.tensor([float(fs['timestamp_bins'][-1].item())]),
                         fs['timestamp_bins'], right=False)

    return fs, model.to(device), all_ids, all_embs.to(device), ts.to(device)

def sample_pool(fs):    
    if not MOVIES_CSV.exists():
        raise SystemExit(f"{MOVIES_CSV} 없음.")
    df = pd.read_csv(MOVIES_CSV, usecols=['movieId'])
    corpus = set(fs['item_emb_movieId_to_i'].keys())
    titles = [fs['movieId_to_title'][int(m)] for m in df['movieId']
              if int(m) in corpus and int(m) in fs['movieId_to_title']]
    return len(df), titles

def _genre_context(fs, liked_with_w, disliked, n_genres):
    ctx = [0.0] * (2 * n_genres)
    title_to_movieId  = fs['title_to_movieId']
    movieId_to_genres = fs['movieId_to_genres']
    avg_rating_to_i   = fs['user_context_genre_avg_rating_to_i']
    watch_count_to_i  = fs['user_context_genre_watch_count_to_i']

    rating_sum, movie_count = {}, {}
    for t, w in list(liked_with_w) + [(t, DISLIKED_MOVIE_VALUE) for t in disliked]:
        mid = title_to_movieId.get(t)
        if mid is None:
            continue
        for g in movieId_to_genres.get(mid, []):
            rating_sum[g]  = rating_sum.get(g, 0.0) + w
            movie_count[g] = movie_count.get(g, 0) + 1

    total_assign = sum(movie_count.values())
    for g, rsum in rating_sum.items():
        if g in avg_rating_to_i:
            ctx[avg_rating_to_i[g]] = rsum / movie_count[g]
        if g in watch_count_to_i:
            ctx[watch_count_to_i[g]] = movie_count[g] / max(total_assign, 1)
    return ctx

def build_user_embedding_batch(model, fs, batch_liked, batch_disliked, ts, device):
    """batch_liked: list[list[(title, weight)]], batch_disliked: list[list[title]] → (B, 128)."""
    n_genres = len(fs['genres_ordered'])
    t2m      = fs['title_to_movieId']
    m2i      = fs['item_emb_movieId_to_i']
    pad      = model.pad_idx
    B        = len(batch_liked)

    ctxs, hists, rats, likeds, disikeds = [], [], [], [], []
    for liked_w, dis in zip(batch_liked, batch_disliked):
        ctxs.append(_genre_context(fs, liked_w, dis, n_genres))

        liked_hist = [(m2i[t2m[t]], w) for t, w in liked_w if t in t2m and t2m[t] in m2i]
        dis_hist   = [(m2i[t2m[t]], DISLIKED_MOVIE_VALUE) for t in dis if t in t2m and t2m[t] in m2i]
        history    = liked_hist + dis_hist

        hists.append([h[0] for h in history] or [pad])
        rats.append([h[1] for h in history] or [0.0])
        likeds.append([h[0] for h in liked_hist] or [pad])
        disikeds.append([h[0] for h in dis_hist] or [pad])

    def pad_ids(seqs):
        """오른쪽 정렬(왼쪽 패딩)로 배치 최대 길이에 맞춘다."""
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        return out

    def pad_w(seqs):
        """가중치는 0.0 으로 패딩."""
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L)
        for i, s in enumerate(seqs):
            out[i, L - len(s):] = torch.tensor(s, dtype=torch.float)
        return out

    X_genre  = torch.tensor(ctxs, dtype=torch.float32).to(device)
    hist_ids = pad_ids(hists).to(device)
    liked_t  = pad_ids(likeds).to(device)
    dis_t    = pad_ids(disikeds).to(device)
    hist_wts = pad_w(rats).to(device)
    ts_batch = ts.expand(B) if ts.numel() == 1 else ts

    return model.user_embedding(X_genre, hist_ids, liked_t, dis_t, hist_wts, ts_batch)

class BatchServer:
    def __init__(self, model, fs, all_embs, ts, device, max_batch, max_wait_s, top_k):
        self.model, self.fs, self.all_embs = model, fs, all_embs
        self.ts, self.device = ts, device
        self.max_batch, self.max_wait_s, self.top_k = max_batch, max_wait_s, top_k
        self.q = queue.Queue()
        self._stop = threading.Event()
        self.batch_sizes = []
        self.compute_ms  = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)

    def submit(self, liked_with_w, disliked) -> Future:
        fut = Future()
        self.q.put((liked_with_w, disliked, fut))
        return fut

    def _loop(self):
        while not self._stop.is_set():
            try:
                first = self.q.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            # 첫 요청 도착 후 max_wait 동안 최대 max_batch 까지 추가 수집
            deadline = time.perf_counter() + self.max_wait_s
            while len(batch) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break
            self._run(batch)

    def _run(self, batch):
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                U = build_user_embedding_batch(
                    self.model, self.fs,
                    [b[0] for b in batch], [b[1] for b in batch], self.ts, self.device)
                # (n_movies, 128) @ (128, B) → (n_movies, B)
                scores = self.all_embs @ U.T
                top    = torch.topk(scores, self.top_k, dim=0).indices.T.cpu().tolist()
            dt = time.perf_counter() - t0
            self.batch_sizes.append(len(batch))
            self.compute_ms.append(dt * 1000)
            for (_, _, fut), t in zip(batch, top):
                fut.set_result(t)
        except Exception as e:
            for (_, _, fut) in batch:
                fut.set_exception(e)

def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float('nan')

def report(tag, latencies_ms, lags_ms, wall, n, extra=None):
    print(f"\n── 결과 [{tag}] ─────────────────────────────────────────────")
    print(f"  완료 쿼리    : {n:,}")
    print(f"  전체 처리시간: {wall:.3f} 초")
    print(f"  QPS          : {n / wall:.2f}   (= {n:,} / {wall:.3f}s)")
    print()
    print(f"  지연 median  : {statistics.median(latencies_ms):.3f} ms")
    print(f"  지연 p99     : {pct(latencies_ms, 99):.3f} ms")
    print(f"  (참고) p95   : {pct(latencies_ms, 95):.3f} ms   "
          f"mean {statistics.mean(latencies_ms):.3f}   "
          f"min {min(latencies_ms):.3f}   max {max(latencies_ms):.3f}")
    print(f"  발사 지연    : median {statistics.median(lags_ms):.3f}  p99 {pct(lags_ms, 99):.3f} ms")
    if pct(lags_ms, 99) > 50:
        print(" 트래픽 생성기가 목표 레이트를 못 따라갔습니다.")
    if extra:
        for k, v in extra.items():
            print(f"  {k:<13}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=('cpu', 'xpu'), default='cpu')
    ap.add_argument('--rate', type=float, default=8.0, help='초당 발사할 쿼리 수')
    ap.add_argument('--duration', type=float, default=30.0, help='부하 지속 시간(초)')
    ap.add_argument('--movies', type=int, default=3, help='쿼리당 무작위 영화 수')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--max-batch', type=int, default=32, help='한 forward 에 묶을 최대 쿼리 수')
    ap.add_argument('--max-wait-ms', type=float, default=5.0, help='배치 수집 최대 대기(ms)')
    ap.add_argument('--workers', type=int, default=64, help='클라이언트 스레드 풀 크기')
    ap.add_argument('--no-batch', action='store_true', help='배칭 끄고 스레드별 배치=1 (비교군)')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--weight', type=float, default=5.0)
    ap.add_argument('--torch-threads', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--verify', action='store_true',
                    help='배치 경로 결과를 src/inference.py 단일 경로와 대조하고 종료')
    args = ap.parse_args()

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)

    device = resolve_device(args.device)
    rng    = random.Random(args.seed)

    print("serving/ 로드 중 ...")
    fs, model, all_ids, all_embs, ts = load_serving(device)
    n_csv, pool = sample_pool(fs)

    # ── 정확성 검증 모드 ──────────────────────────────────────────────────────
    if args.verify:
        B = 16
        batch_liked = [[(t, args.weight) for t in rng.sample(pool, args.movies)] for _ in range(B)]
        batch_dis   = [[] for _ in range(B)]
        with torch.no_grad():
            U_batch = build_user_embedding_batch(model, fs, batch_liked, batch_dis, ts, device)
            U_ref = torch.cat([build_user_embedding(model, fs, lw, [], ts)
                               for lw in batch_liked], dim=0)
        d = (U_batch - U_ref).abs().max().item()
        print(f"\n배치 유저 임베딩 {tuple(U_batch.shape)}  vs  단일 경로 {tuple(U_ref.shape)}")
        print(f"최대 절대 오차: {d:.3e}   →  {'일치 (float32 오차 범위)' if d < 1e-4 else '★불일치★'}")
        # 랭킹까지 같은지도 확인 — 임베딩이 같으면 자명하지만 스코어링 경로도 함께 검증한다.
        s_b = (all_embs @ U_batch.T).topk(10, dim=0).indices.T.cpu()
        s_r = torch.stack([(all_embs @ U_ref[i:i+1].T).squeeze(-1).topk(10).indices.cpu()
                           for i in range(B)])
        print(f"top-10 랭킹 일치: {bool((s_b == s_r).all())}")
        return

    total    = int(round(args.rate * args.duration))
    interval = 1.0 / args.rate
    mode     = 'no-batch (스레드별 배치=1)' if args.no_batch else \
               f'dynamic batching (max {args.max_batch}, wait {args.max_wait_ms:g}ms)'

    print(f"\n── 설정 ─────────────────────────────────────────────────────")
    print(f"  디바이스     : {device}"
          + (f"  ({torch.xpu.get_device_name(0)})" if device.type == 'xpu' else ""))
    print(f"  모드         : {mode}")
    print(f"  torch 스레드 : {torch.get_num_threads()}  (intra-op)")
    print(f"  샘플 풀      : movies.csv {n_csv:,}편 ∩ 코퍼스 → {len(pool):,}편")
    print(f"  부하         : {args.rate:g} QPS × {args.duration:g}초 = {total:,} 쿼리 "
          f"(쿼리당 {args.movies}편, top-{args.top_k})")
    print(f"  클라이언트   : 스레드 {args.workers}  |  워밍업 {args.warmup}회")

    queries = [rng.sample(pool, args.movies) for _ in range(total)]
    server, extra = None, {}

    if args.no_batch:
        # 비교군: 클라이언트 스레드가 각자 배치=1 forward. 모델은 eval+no_grad 로 읽기 전용 공유.
        def call(liked, scheduled_at):
            start = time.perf_counter()
            with torch.no_grad():
                u = build_user_embedding(model, fs, [(t, args.weight) for t in liked], [], ts)
                s = (all_embs @ u.T).squeeze(-1)
                _ = torch.topk(s, args.top_k).indices.cpu().tolist()
            end = time.perf_counter()
            return end - start, start - scheduled_at
    else:
        server = BatchServer(model, fs, all_embs, ts, device,
                             args.max_batch, args.max_wait_ms / 1000.0, args.top_k).start()

        def call(liked, scheduled_at):
            start = time.perf_counter()
            fut = server.submit([(t, args.weight) for t in liked], [])
            fut.result()                       # 큐 대기 + 배치 계산까지 포함한 클라이언트 관측 지연
            end = time.perf_counter()
            return end - start, start - scheduled_at

    # ── 워밍업 ────────────────────────────────────────────────────────────────
    for _ in range(args.warmup):
        call(rng.sample(pool, args.movies), time.perf_counter())
    if server:
        server.batch_sizes.clear(); server.compute_ms.clear()
    # sync(device)

    # ── 부하 발생 (open-loop) ─────────────────────────────────────────────────
    print(f"\n부하 시작 ... ({args.duration:g}초)")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        t0 = time.perf_counter()
        futures = []
        for i, liked in enumerate(queries):
            target = t0 + i * interval
            now    = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            futures.append(ex.submit(call, liked, target))
        results = [f.result() for f in futures]
        wall = time.perf_counter() - t0

    lat_ms = [r[0] * 1000 for r in results]
    lag_ms = [r[1] * 1000 for r in results]

    if server:
        bs = server.batch_sizes
        extra = {
            'forward 호출': f"{len(bs):,} 회  (쿼리 {len(results):,} → 배치당 평균 "
                            f"{statistics.mean(bs):.2f}, 최대 {max(bs)})",
            '배치 계산시간': f"median {statistics.median(server.compute_ms):.3f}  "
                            f"p99 {pct(server.compute_ms, 99):.3f} ms",
        }
        server.stop()

    report(mode, lat_ms, lag_ms, wall, len(results), extra)

    if args.out:
        payload = {
            'device': str(device), 'mode': mode, 'rate': args.rate, 'duration': args.duration,
            'max_batch': args.max_batch, 'max_wait_ms': args.max_wait_ms,
            'workers': args.workers, 'torch_threads': torch.get_num_threads(),
            'n_queries': len(results), 'wall_seconds': wall, 'qps': len(results) / wall,
            'latency_ms': {'median': statistics.median(lat_ms), 'p95': pct(lat_ms, 95),
                           'p99': pct(lat_ms, 99), 'mean': statistics.mean(lat_ms),
                           'min': min(lat_ms), 'max': max(lat_ms)},
            'scheduling_lag_ms': {'median': statistics.median(lag_ms), 'p99': pct(lag_ms, 99)},
            'batch_sizes': (server.batch_sizes if server else None),
            'latencies_ms': lat_ms,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\n  → {args.out} 저장")

if __name__ == '__main__':
    main()