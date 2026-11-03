#!/usr/bin/env python3
"""Pure-Python check of the Ricochet Catch v0.1 spec: mechanics fixtures and baseline calibration.

Review aid only. Not part of any prompt bundle.
"""
import argparse
import random
import statistics
from collections import namedtuple

W = 8
PADDLE_ROW = 7
GEM, BOMB = "gem", "bomb"
DELTA = (0, -1, 1)  # stay, left, right


def clip(c):
    return 0 if c < 0 else (W - 1 if c > W - 1 else c)


def step_obj(col, vx):
    nc = col + vx
    if nc < 0 or nc > W - 1:
        vx = -vx
        nc = col + vx
    return nc, vx


# LAND[(row, col, vx)] -> (ticks until it reaches row 7, column there)
LAND = {}
for r in range(PADDLE_ROW):
    for c in range(W):
        for v in (-1, 0, 1):
            col, vv = c, v
            for _ in range(PADDLE_ROW - r):
                col, vv = step_obj(col, vv)
            LAND[(r, c, v)] = (PADDLE_ROW - r, col)

# REFL[(spawn_col, vx)] -> whether the 7-move fall from row 0 includes a wall reflection
REFL = {}
for c in range(W):
    for v in (-1, 0, 1):
        col, vv, refl = c, v, False
        for _ in range(PADDLE_ROW):
            ncol, nvv = step_obj(col, vv)
            refl |= nvv != vv
            col, vv = ncol, nvv
        REFL[(c, v)] = refl


class Cfg:
    def __init__(self, spawn_p=0.45, bomb_p=0.25, vel_p=(1 / 3, 1 / 3, 1 / 3), horizon=256):
        self.spawn_p, self.bomb_p, self.vel_p, self.horizon = spawn_p, bomb_p, vel_p, horizon


def make_tape(seed, cfg):
    """Fixed number of draws per tick, in a fixed order, whether or not anything spawns."""
    rng = random.Random(seed)
    tape = []
    for _ in range(cfg.horizon):
        u_spawn, u_col, u_vel, u_type = rng.random(), rng.random(), rng.random(), rng.random()
        if u_spawn >= cfg.spawn_p:
            tape.append(None)
            continue
        col = min(W - 1, int(u_col * W))
        vx = -1 if u_vel < cfg.vel_p[0] else (0 if u_vel < cfg.vel_p[0] + cfg.vel_p[1] else 1)
        kind = BOMB if u_type < cfg.bomb_p else GEM
        tape.append((kind, col, vx))
    return tape


State = namedtuple("State", "paddle objs remaining")  # all a policy may see


class Env:
    def __init__(self, tape, cfg, objs=(), paddle=3, remaining=None, spawning=True):
        self.tape, self.cfg, self.spawning = tape, cfg, spawning
        self.paddle = paddle
        # object = (kind, row, col, vx, reflects_on_way_down)
        self.objs = [o if len(o) == 5 else (*o, False) for o in objs]
        rows = [o[1] for o in self.objs]
        assert len(rows) == len(set(rows)) and all(0 <= r < PADDLE_ROW for r in rows), "unreachable state"
        self.remaining = cfg.horizon if remaining is None else remaining
        self.tick, self.score, self.done, self.bomb_hit = 0, 0, False, False
        self.landed_gems = self.landed_refl = self.caught_refl = 0
        self.spawned = []  # (tick, entry) revealed spawns, for tape-independence checks

    def state(self):
        return State(self.paddle, tuple(self.objs), self.remaining)

    def step(self, a):
        if self.done:
            raise RuntimeError("step() on finished episode")
        if type(a) is not int or a not in (0, 1, 2):
            raise ValueError(f"bad action {a!r}")
        self.paddle = clip(self.paddle + DELTA[a])
        reward, hit, survivors = 0, False, []
        for kind, row, col, vx, refl in self.objs:
            col, vx = step_obj(col, vx)
            row += 1
            if row < PADDLE_ROW:
                survivors.append((kind, row, col, vx, refl))
                continue
            if kind == GEM:
                self.landed_gems += 1
                self.landed_refl += refl
                if col == self.paddle:
                    reward += 1
                    self.caught_refl += refl
            elif col == self.paddle:
                hit = True
        self.objs = survivors
        self.tick += 1
        self.remaining -= 1
        self.score += reward
        if hit:
            self.done = self.bomb_hit = True
            return reward, True, False
        if self.remaining == 0:
            self.done = True
            return reward, False, True
        if self.spawning:
            e = self.tape[self.tick - 1]
            if e is not None:
                self.objs.append((e[0], 0, e[1], e[2], REFL[(e[1], e[2])]))
                self.spawned.append((self.tick, e))
        return reward, False, False


def observe(s):
    x = [0.0] * 449
    x[(PADDLE_ROW * W + s.paddle) * 7] = 1.0
    for kind, row, col, vx, _ in s.objs:
        idx = (row * W + col) * 7 + (1 if kind == GEM else 4) + (vx + 1)
        assert x[idx] == 0.0
        x[idx] = 1.0
    x[448] = s.remaining / 256
    return x


# ---------------------------------------------------------------- policies (state only, never the tape)

def pol_stay(s):
    return 0


def make_random(seed):
    rng = random.Random(seed)
    return lambda s: rng.randrange(3)


def toward(p, t):
    return 0 if t == p else (1 if t < p else 2)


def soonest_gem(s):
    gems = [(PADDLE_ROW - row, abs(col - s.paddle), col) for kind, row, col, vx, _ in s.objs if kind == GEM]
    return min(gems) if gems else None


def safe_actions(s):
    unsafe = {LAND[(row, col, vx)][1] for kind, row, col, vx, _ in s.objs if kind == BOMB and row == PADDLE_ROW - 1}
    return [a for a in (0, 1, 2) if clip(s.paddle + DELTA[a]) not in unsafe]


def pol_chase(s):
    g = soonest_gem(s)
    return 0 if g is None else toward(s.paddle, g[2])


def pol_chase_safe(s):
    safe = safe_actions(s)
    if not safe:
        return 0
    g = soonest_gem(s)
    if g is None:
        return safe[0]
    return min(safe, key=lambda a: (abs(clip(s.paddle + DELTA[a]) - g[2]), a))


def pol_reference(s, horizon=7):
    """Receding-horizon DP over visible objects; bombs forbidden; terminal value 0; ties -> smallest action."""
    h = min(horizon, s.remaining)
    gem = [[0] * W for _ in range(h + 1)]
    bomb = [[False] * W for _ in range(h + 1)]
    for kind, row, col, vx, _ in s.objs:
        k, lc = LAND[(row, col, vx)]
        if k <= h:
            if kind == GEM:
                gem[k][lc] += 1
            else:
                bomb[k][lc] = True
    neg = -10 ** 9
    V = [0] * W  # value-to-go after tick h
    for k in range(h, 1, -1):  # V becomes value-to-go after tick k-1
        nv = []
        for p in range(W):
            best = neg
            for d in DELTA:
                q = clip(p + d)
                if not bomb[k][q] and gem[k][q] + V[q] > best:
                    best = gem[k][q] + V[q]
            nv.append(best)
        V = nv
    best_a, best_v = 0, neg
    for a in (0, 1, 2):
        q = clip(s.paddle + DELTA[a])
        if not bomb[1][q] and gem[1][q] + V[q] > best_v:
            best_a, best_v = a, gem[1][q] + V[q]
    return best_a


# Extra diagnostics (not in the proposal): what simpler policy classes score.

def pol_dodge_only(s):
    safe = safe_actions(s)
    return safe[0] if safe else 0


def pol_predict_greedy_safe(s):
    """Soonest reachable gem by *predicted landing column*, with the same 1-step bomb veto."""
    safe = safe_actions(s)
    if not safe:
        return 0
    cands = []
    for kind, row, col, vx, _ in s.objs:
        if kind == GEM:
            k, lc = LAND[(row, col, vx)]
            if abs(lc - s.paddle) <= k:
                cands.append((k, abs(lc - s.paddle), lc))
    if not cands:
        return safe[0]
    t = min(cands)[2]
    return min(safe, key=lambda a: (abs(clip(s.paddle + DELTA[a]) - t), a))


POLICIES = {
    "stay": lambda i: pol_stay,
    "random": lambda i: make_random(10_000_019 + i),
    "chase": lambda i: pol_chase,
    "chase_safe": lambda i: pol_chase_safe,
    "reference": lambda i: pol_reference,
    "dodge_only": lambda i: pol_dodge_only,
    "predict_greedy_safe": lambda i: pol_predict_greedy_safe,
}


def run_episode(policy, tape, cfg):
    env = Env(tape, cfg)
    while not env.done:
        env.step(policy(env.state()))
    return dict(score=env.score, bomb=env.bomb_hit, length=env.tick,
                landed_gems=env.landed_gems, landed_refl=env.landed_refl, caught_refl=env.caught_refl)


def evaluate(names, seeds, cfg):
    tapes = [make_tape(s, cfg) for s in seeds]
    return {n: [run_episode(POLICIES[n](i), t, cfg) for i, t in enumerate(tapes)] for n in names}


def bootstrap_lower(diffs, n_boot=10_000, seed=12345, q=0.05):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(q * n_boot)]


# ---------------------------------------------------------------- fixtures

def run_fixtures():
    cfg = Cfg()
    nil = [None] * 256

    def one(objs, actions, paddle=3, remaining=100):
        env = Env(nil, cfg, objs=objs, paddle=paddle, remaining=remaining, spawning=False)
        out = [env.step(a) for a in actions]
        return env, out

    env, _ = one([(GEM, 3, 7, 1)], [0])
    assert env.objs[0][:4] == (GEM, 4, 6, -1), env.objs
    env, _ = one([(BOMB, 3, 0, -1)], [0])
    assert env.objs[0][:4] == (BOMB, 4, 1, 1), env.objs
    env, _ = one([(GEM, 3, 6, 1)], [0])
    assert env.objs[0][:4] == (GEM, 4, 7, 1), env.objs
    env.step(0)
    assert env.objs[0][:4] == (GEM, 5, 6, -1), env.objs
    env, out = one([(GEM, 6, 3, 0)], [0])
    assert out == [(1, False, False)] and env.score == 1 and env.objs == []
    env, out = one([(BOMB, 6, 3, 0)], [0])
    assert out == [(0, True, False)] and env.done and env.bomb_hit
    env, out = one([(BOMB, 6, 3, 0)], [1])
    assert out == [(0, False, False)] and env.objs == [] and not env.done
    env, _ = one([], [1], paddle=0)
    assert env.paddle == 0
    env, _ = one([], [2], paddle=7)
    assert env.paddle == 7
    env, out = one([(GEM, 6, 3, 0)], [0], remaining=1)
    assert out == [(1, False, True)]
    env, out = one([(BOMB, 6, 3, 0)], [0], remaining=1)
    assert out == [(0, True, False)]
    tape = [(GEM, 2, 1)] + [None] * 255
    env = Env(tape, cfg)
    env.step(0)
    assert env.objs == [(GEM, 0, 2, 1, REFL[(2, 1)])], env.objs
    # prediction fixture
    env, out = one([(GEM, 4, 6, 1)], [2, 2, 0])
    assert env.score == 1, "R,R,S should catch"
    env, out = one([(GEM, 4, 6, 1)], [2, 2, 2])
    assert env.score == 0, "R,R,R should miss"
    # conflicting-objective fixture: every 2-step sequence that would catch the gem hits the bomb first
    for a1 in (0, 1, 2):
        for a2 in (0, 1, 2):
            env = Env(nil, cfg, objs=[(BOMB, 6, 4, 0), (GEM, 5, 5, 0)], paddle=3, remaining=100, spawning=False)
            env.step(a1)
            if env.done:
                continue
            env.step(a2)
            assert env.score == 0, (a1, a2)
    # determinism and tape independence
    tape = make_tape(7, cfg)
    e1, e2 = Env(tape, cfg), Env(tape, cfg)
    rng1, rng2 = random.Random(1), random.Random(2)
    while not (e1.done or e2.done):
        e1.step(rng1.randrange(3))
        e2.step(rng2.randrange(3))
    n = min(len(e1.spawned), len(e2.spawned))
    common_ticks = min(e1.tick, e2.tick)
    assert [s for s in e1.spawned if s[0] <= common_ticks] == [s for s in e2.spawned if s[0] <= common_ticks]
    # observation sanity on random reachable states
    for seed in range(20):
        env = Env(make_tape(seed, cfg), cfg)
        rng = random.Random(seed)
        while not env.done:
            s = env.state()
            x = observe(s)
            assert len(x) == 449 and sum(x[:448]) == 1 + len(s.objs) and len(s.objs) <= 7
            env.step(rng.randrange(3))
    print("all fixtures pass (incl. prediction + conflict fixtures, tape independence, observation invariants)")


def summarize(name, res):
    sc = [r["score"] for r in res]
    lg = sum(r["landed_gems"] for r in res)
    lr = sum(r["landed_refl"] for r in res)
    cr = sum(r["caught_refl"] for r in res)
    return (f"{name:20s} mean {statistics.mean(sc):6.2f}  sd {statistics.pstdev(sc):5.2f}  "
            f"median {statistics.median(sc):5.1f}  bomb-death {sum(r['bomb'] for r in res) / len(res):5.1%}  "
            f"len {statistics.mean(r['length'] for r in res):6.1f}  catch/landed {sum(sc) / max(lg, 1):5.1%}  "
            f"reflected catch/landed {cr / max(lr, 1):5.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--spawn-p", type=float, default=0.45)
    ap.add_argument("--bomb-p", type=float, default=0.25)
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--policies", default=",".join(POLICIES))
    args = ap.parse_args()
    if args.fixtures:
        run_fixtures()
        return
    cfg = Cfg(spawn_p=args.spawn_p, bomb_p=args.bomb_p)
    names = args.policies.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    R = evaluate(names, seeds, cfg)
    print(f"spawn_p={cfg.spawn_p} bomb_p={cfg.bomb_p} episodes={len(seeds)}")
    for n in names:
        print(summarize(n, R[n]))
    if not {"stay", "random", "chase", "chase_safe", "reference"} <= set(names):
        return
    mean = {n: statistics.mean(r["score"] for r in R[n]) for n in names}
    triv = max(("stay", "random", "chase"), key=lambda n: (mean[n], -("stay", "random", "chase").index(n)))
    ref = mean["reference"]
    print(f"\ntrivial comparator = {triv} ({mean[triv]:.2f}); reference = {ref:.2f}; "
          f"halfway bar = {(ref + mean[triv]) / 2:.2f}")
    print(f"check ref >= 10: {ref >= 10} | ref >= 1.2*chase_safe ({1.2 * mean['chase_safe']:.2f}): "
          f"{ref >= 1.2 * mean['chase_safe']} | ref >= trivial+10: {ref >= mean[triv] + 10}")
    print("\nIf each policy below were the submitted network, would the halfway criterion pass?")
    for n in names:
        if n in ("reference",):
            continue
        d = [R[n][i]["score"] - 0.5 * (R["reference"][i]["score"] + R[triv][i]["score"]) for i in range(len(seeds))]
        lb = bootstrap_lower(d)
        print(f"  {n:20s} mean D {statistics.mean(d):7.2f}   one-sided 95% lower bound {lb:7.2f}   pass={lb > 0}")


if __name__ == "__main__":
    main()
