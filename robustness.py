"""How forgiving is the halfway bar? Predictive-greedy policies with a corrupted landing table / no bomb veto."""
import importlib.util, random, statistics, sys
spec = importlib.util.spec_from_file_location("rs", sys.argv[1]); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)

def corrupted(frac, seed):
    rng = random.Random(seed)
    T = dict(rs.LAND)
    for key, (k, lc) in rs.LAND.items():
        if rng.random() < frac:
            T[key] = (k, rng.choice([c for c in range(rs.W) if c != lc]))
    return T

def make_pg(T, veto=True):
    def pol(s):
        unsafe = {T[(r, c, v)][1] for kind, r, c, v, _ in s.objs if kind == rs.BOMB and r == 6} if veto else set()
        safe = [a for a in (0, 1, 2) if rs.clip(s.paddle + rs.DELTA[a]) not in unsafe] or [0]
        cands = [(k, abs(lc - s.paddle), lc) for kind, r, c, v, _ in s.objs if kind == rs.GEM
                 for k, lc in [T[(r, c, v)]] if abs(lc - s.paddle) <= k]
        if not cands:
            return safe[0]
        t = min(cands)[2]
        return min(safe, key=lambda a: (abs(rs.clip(s.paddle + rs.DELTA[a]) - t), a))
    return pol

cfg = rs.Cfg(); tapes = [rs.make_tape(s, cfg) for s in range(300000, 300256)]
def score(pol):
    res = [rs.run_episode(pol, t, cfg) for t in tapes]
    return statistics.mean(r["score"] for r in res), sum(r["bomb"] for r in res) / len(res)
print("bar ~31.8 (trivial 6.2, reference 57.4)")
m, b = score(make_pg(rs.LAND, veto=False)); print(f"exact table, NO bomb veto            mean {m:5.1f}  bomb-death {b:5.1%}")
for frac in (0.05, 0.10, 0.20, 0.35, 0.50):
    ms, bs = zip(*(score(make_pg(corrupted(frac, rep))) for rep in range(3)))
    print(f"{frac:4.0%} of 168 landing entries wrong (3 draws)  mean {statistics.mean(ms):5.1f}  "
          f"range {min(ms):5.1f}-{max(ms):5.1f}  bomb-death {statistics.mean(bs):5.1%}")
