"""Turn a palette into a mixing guide for the mod's 12 palette spots.

How the palette behaves, per BasePalette and PaletteUtil in the mod source:

  * a dye dropped into a spot adds one entry; the spot shows the floored mean of
    its entries
  * dragging one spot onto another carries only its resulting color, which lands
    as ONE entry
  * water resets a spot

The second point rules out branching from a copy. A copied 4-drop mix weighs like
a single drop, so extending the copy produces a different color: replaying a plan
that branched through copies against the mod's own arithmetic got 79 of 104
colors wrong, median dE 9.5. The only exact way to reuse work is to grow one spot
in place.

So the guide is a set of chains. A spot grows c1 < c2 < ... < ck, costs |ck|
drops, and serves every color along the way. Choosing chains to minimize total
drops is a minimum-weight path cover of the sub-multiset order, which is solved
exactly as an assignment problem. On real palettes that saves 8-16% of drops; the
bigger win is the step-by-step card, not the arithmetic.

Every plan is replayed through SpotMix, a transcription of the mod's CustomColor,
and `mismatches` lists any color that would not come out exactly.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .colors import BASE_COLORS, hex_to_rgb, rgb_to_hex
from .palette_layout import EMPTY_SPOT

MAX_SPOTS = 12


class SpotMix:
    """PaletteUtil.CustomColor, transcribed: totals, a count, and the integer gain."""

    def __init__(self):
        self.r = self.g = self.b = self.peak = self.n = 0

    def add(self, rgb):
        self.r += rgb[0]
        self.g += rgb[1]
        self.b += rgb[2]
        self.peak += max(rgb)
        self.n += 1

    def color(self):
        if self.n == 0:
            return EMPTY_SPOT
        ar, ag, ab = self.r // self.n, self.g // self.n, self.b // self.n
        top = max(ar, ag, ab)
        gain = 0 if top == 0 else (self.peak // self.n) // top
        return (ar * gain, ag * gain, ab * gain)


@dataclass
class Step:
    action: str              # 'add' | 'wash'
    spot: int
    dye: str = ""            # for 'add'
    hex: str = ""            # the color this step completes, if any
    target: bool = False
    spots: tuple = ()        # every spot's hex after this tep, None when blank

    def describe(self):
        if self.action == 'wash':
            return "Wash the highlighted spot with water"
        return f"Add {self.dye} to the highlighted spot"


@dataclass
class MixPlan:
    steps: list = field(default_factory=list)
    drops: int = 0
    washes: int = 0
    chains: int = 0
    scratch_actions: int = 0     # each color mixed in its own spot, washing to reuse
    mismatches: list = field(default_factory=list)

    @property
    def actions(self):
        return self.drops + self.washes

    @property
    def targets(self):
        return [i for i, s in enumerate(self.steps) if s.target]

    def summary(self):
        if not self.scratch_actions:
            return "No colors to mix."
        saved = self.scratch_actions - self.actions
        pct = 100.0 * saved / self.scratch_actions
        return (f"{self.actions} actions ({self.drops} drops, {self.washes} washes) "
                f"instead of {self.scratch_actions}, {pct:.0f}% less\n"
                f"{self.chains} mixes, each grown in place")

    def group(self, step_index):
        """(first, last) steps of the color that `step_index` works toward."""
        targets = self.targets
        if not targets:
            return 0, len(self.steps) - 1
        prev = -1
        for t in targets:
            if step_index <= t:
                return prev + 1, t
            prev = t
        return targets[-1] + 1, len(self.steps) - 1

    def step_for(self, hex_color):
        for i, s in enumerate(self.steps):
            if s.target and s.hex.lower() == hex_color.lower():
                return i
        return None


def _chains(recipes):
    """Minimum-weight path cover of the strict sub-multiset order.
    
    A color that gets a successor stops being the top of its chain, saving its own
    drop count, so maximizing the total size of matched predecessors minimizes the
    drops spent.
    """
    n = len(recipes)
    counts = [Counter(r) for r in recipes]
    sizes = [len(r) for r in recipes]
    big = 1e9
    cost = np.full((n, n), big)
    for i in range(n):
        for j in range(n):
            if sizes[i] < sizes[j] and all(counts[j][d] >= c for d, c in counts[i].items()):
                cost[i, j] = -sizes[i]
    nxt = {}
    if n:
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] < big / 2:
                nxt[i] = j
    starts = [i for i in range(n) if i not in set(nxt.values())]
    chains = []
    for s in starts:
        chain = [s]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    # Longest first, so the work that pays off most is at the top of the guide.
    chains.sort(key=lambda c: (-len(c), recipes[c[0]]))
    return chains


def plan_mixes(targets, max_spots=MAX_SPOTS):
    """Build the guide. `targets` is [(hex, recipe), ...] for the colors in use."""
    seen = {}
    for h, r in targets:
        if r and tuple(sorted(r)) not in seen:
            seen[tuple(sorted(r))] = h
    recipes = list(seen)
    hexes = [seen[r] for r in recipes]
    plan = MixPlan()
    n = len(recipes)
    plan.scratch_actions = sum(len(r) for r in recipes) + max(0, n - max_spots)
    if not n:
        return plan

    dye_rgb = {name: hex_to_rgb(h) for name, h in BASE_COLORS.items()}
    spots = [SpotMix() for _ in range(max_spots)]
    used = [False] * max_spots
    chains = _chains(recipes)
    plan.chains = len(chains)

    def snapshot():
        return tuple(rgb_to_hex(s.color()) if s.n else None for s in spots)

    for k, chain in enumerate(chains):
        spot = k % max_spots
        if used[spot]:
            spots[spot] = SpotMix()
            plan.washes += 1
            plan.steps.append(Step('wash', spot, spots=snapshot()))
        used[spot] = True
        have = Counter()
        for node in chain:
            need = Counter(recipes[node]) - have
            for dye in sorted(need.elements()):
                spots[spot].add(dye_rgb[dye])
                plan.drops += 1
                plan.steps.append(Step('add', spot, dye=dye, spots=snapshot()))
            have = Counter(recipes[node])
            last = plan.steps[-1]
            last.hex = hexes[node]
            last.target = True
            got = rgb_to_hex(spots[spot].color())
            if got.lower() != hexes[node].lower():
                plan.mismatches.append((hexes[node], got))
    return plan
