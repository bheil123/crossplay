# ACCURACY.md -- Move Evaluation Accuracy Analysis

## How equity is computed

The engine evaluates moves in three phases, each adding information:

### Phase 1: Static evaluation (all moves)

```
prelim_equity = score + leave_value
```

- **Score**: exact (well-tested scoring engine)
- **Leave value**: hand-tuned formula + bingo probability DB lookup
- Used to rank and select top N candidates for deeper analysis

### Phase 2: Risk + positional analysis (top ~115 moves)

```
equity = score + leave_value - expected_risk + blocking_bonus + dls_penalty + dd_bonus + turnover_bonus
positional_adj = blocking_bonus - expected_risk + dls_penalty + dd_bonus + turnover_bonus
```

- **Expected risk**: hypergeometric probability-weighted opponent threat damage
- **Blocking bonus**: credit for covering bonus squares
- **DLS penalty**: exposure of high-value tiles near double-letter squares
- **DD bonus/penalty**: double-double lane creation/exposure
- **Turnover bonus**: small credit for using more rack tiles (more draws)
- `positional_adj` is carried forward to MC for dampened inclusion

### Phase 3: Monte Carlo 2-ply (top N=40 moves)

```
mc_equity = score - mc_avg_opp
total_equity = mc_equity + leave_value + positional_adj * 0.5
```

- **mc_avg_opp**: average opponent best-response score across K simulations
- **leave_value**: same formula+bingo evaluation as Phase 1
- **positional_adj * 0.5**: Phase 2 adjustment dampened to 50% (assumes MC
  captures ~50% of positional effects through opponent response sampling)
- **risk_adj_equity**: total_equity minus Phase 2 expected risk (undampened).
  Shows what equity looks like when threat analysis is fully weighted.

### Phase 4: 3-ply endgame (when bag <= 12)

Deterministic 3-ply: your move, opponent best, your best response.
Replaces MC equity when available (exact vs sampled).

---

## Accuracy factors ranked by impact

### 1. Leave evaluation (biggest gap)

**Current approach:** Hand-tuned formula + bingo probability DB.

The formula assigns fixed bonuses and penalties:
- Blank: +8, S: +6, good tiles (E,R,T,N,A,I): +1.5
- Awkward tiles (Q,V,J,X,Z,K): -3.0
- Vowel/consonant imbalance: -2 to -8
- Duplicate penalty: configurable per extra copy
- Bingo stem synergy: partial match bonus
- Bingo DB: P(bingo) * 0.5 * 77.0 expected bingo score

**Why it matters:** Leave value is added directly to MC equity. A 3-5
point systematic error shifts move rankings. Every move evaluation
depends on this.

**Known weaknesses:**
- Formula weights are hand-tuned, not trained on game outcomes
- BINGO_WEIGHT (0.5) and EXPECTED_BINGO_SCORE (77.0) are static guesses
- Bingo DB key is sorted tuple -- loses tile interaction information
- No phase awareness: same weights early game vs late game
- Vowel/consonant ideal ranges are hardcoded, not validated

**Fix path:** Train leave values via self-play (SuperLeaves approach).
Run millions of AI-vs-AI games, measure actual equity contribution of
each leave empirically. This is how Quackle and Maven achieve their
leave accuracy.

### 2. MC blank capping

**Current approach:** Opponent racks capped at 2 blanks for move
generation speed. Post-hoc correction factor applied only when all 3
blanks are unseen, using hypergeometric ratios (RATIO_0v2=0.470,
RATIO_1v2=0.687, RATIO_3v2=1.036). Cap=2 was chosen because it
handles the vast majority of draws exactly (only 3-blank draws need
correction, ~0.3% of draws at unseen=44), while cap=3 is ~2.7x slower.

**Why it matters:** With cap=2, the correction is much smaller than the
old cap=1 approach (~1.01-1.04x vs ~1.05-1.15x). The remaining error
is from the rare 3-blank draws, which are nearly negligible in most
game states.

**Known weaknesses:**
- Correction ratios are global, not per-board-density
- Ratios were calibrated on a single Crossplay board state
- Correction only matters when all 3 blanks are unseen (early-mid game)

**Fix path:** Already substantially improved by moving from cap=1 to
cap=2. Remaining correction is small enough that per-density indexing
has diminishing returns. Could still calibrate at multiple board states
for marginal accuracy.

### 3. Phase 2 positional heuristics

**Current approach:** DLS exposure penalty, double-double lane detection,
tile turnover bonus. All hand-tuned. Applied with 0.5 dampening in MC
to avoid double-counting (MC opponent response already captures some
positional effects).

**Why it matters:** The 0.5 dampening factor is a guess. If MC captures
70% of positional effects, dampening should be 0.3. If MC captures 30%,
dampening should be 0.7. Wrong dampening biases all move rankings.

**Known weaknesses:**
- DLS exposure only checks 1-2 squares away; misses longer parallel plays
- Double-double detection checks DWS proximity without actual lane connectivity
- Hookability table is hand-tuned, not validated against game data
- Turnover bonus uses static 1% per tile probability, ignoring pool density
- All heuristics are phase-independent (same weights throughout game)

**Fix path:** Validate dampening factor via self-play A/B testing. Or
remove heuristics entirely -- MC is now fast enough (2-10s) to be the
primary signal. Heuristics were more valuable when MC took 21-27s and
evaluated fewer candidates.

### 4. Risk analysis integration

**Current approach:** Phase 2 computes expected_risk per move. This
flows into `positional_adj` which is dampened by 0.5 in MC. Risk is
also displayed separately as "Risk (exp/max)" in Phase 2 output.

The new `risk_adj_equity` field shows MC equity minus full (undampened)
expected risk, giving a more conservative estimate that fully accounts
for opened bonus square threats.

**Known weaknesses:**
- Risk analysis only checks words up to 7 letters (misses rare 8-letter
  bingo threats through bonus squares)
- Threat probability assumes uniform opponent drawing from unseen pool
  (no skill modeling)
- Risk display after MC shows Phase 2 risk values that may be partially
  redundant with MC opponent response sampling
- Top 6 threats by EV + top 3 by score may miss nuanced medium-EV threats

**Fix path:** Already partially addressed by `risk_adj_equity`. Could
also compute risk delta (risk after move minus baseline risk) to show
incremental risk exposure.

### 5. N=40 candidate selection

**Current approach:** Phase 1 ranks by `prelim_equity = score +
leave_value` and takes top 40 for MC.

**Why it matters:** Moves that look bad in static eval but are strong
in 2-ply (blocking moves, setup plays, exchanges) can be missed if they
rank below #40. N=40 is a significant improvement over the old N=4-12,
but the underlying issue is prelim_equity's accuracy.

**Known weaknesses:**
- Blocking moves that sacrifice score for position rank low in Phase 1
- Exchange candidates are evaluated separately (good) but could interact
  with regular move selection

**Fix path:** Better leave evaluation (factor 1) would improve Phase 1
ranking and reduce the chance of missing good moves. Could also add a
"diversity" pass that includes moves from different strategic categories.

### 6. MC variance and early stopping

**Current approach:** K=2000 ceiling with convergence-based early
stopping (SE < 1.0, check every 10 sims, min 100). Actual sims per
candidate: 150-530 avg.

**Impact:** Different candidates get different numbers of simulations,
so their equity estimates have different confidence intervals. A move
that converges quickly (150 sims, tight CI) is ranked equally with a
move that needed 530 sims (wider CI). Close calls between moves within
2-3 equity points are unreliable.

**Known weaknesses:**
- Early stopping creates unequal confidence across candidates
- 95% CI (+/-2 pts) may not distinguish moves within 4 equity points
- Volatile moves (high opp_std) need more sims but may stop at 100

**Fix path:** Use confidence intervals for ranking tiebreaks. When two
moves have similar equity, prefer the one with tighter CI. Or implement
progressive widening: start with all candidates at low K, then
progressively add sims to candidates that are still in contention.

---

## Equity formula summary

```
Phase 2 equity  = score + leave - expected_risk + blocking + dls + dd + turnover
MC total_equity = score - mc_avg_opp + leave + positional_adj * 0.5
risk_adj_equity = total_equity - expected_risk
```

Where `positional_adj = blocking - expected_risk + dls + dd + turnover`.

Note: `expected_risk` appears twice in `risk_adj_equity` -- once dampened
inside `total_equity` (via `positional_adj * 0.5`) and once undampened as
the subtracted term. This is intentional: `risk_adj_equity` represents
the conservative view where risk is fully counted, not half-counted.

---

## Improvement roadmap

1. **Trained leave model** (biggest single improvement) -- self-play
   to learn empirical leave values. Replaces hand-tuned formula.
   See "SuperLeaves training plan" section below.
2. **Risk-adjusted equity display** (done) -- shows undampened risk
   so user can see conservative estimate alongside MC equity.
3. **Per-density blank correction** -- calibrate correction ratios
   at multiple board densities instead of using global constants.
4. **Positional heuristic validation** -- A/B test dampening factor
   via self-play. Consider removing heuristics if MC captures them.
5. **Confidence-aware ranking** -- use CI width for tiebreaking
   among moves within 2-3 equity points of each other.

---

## NYT CrossBot comparison (canjam_003)

Game replayed through the crossplay engine and compared against NYT
CrossBot analyzer recommendations. Final score: You 372 - Camjam 378
(loss by 6). 11 of your turns analyzed.

### Summary table

```
Turn  You Played  NYT Best   Crossplay #1  NYT vs CP  Notes
----  ----------  --------   ------------  ---------  -----
 1    PILAW(28)   PILAF(26)  PILAW(28)     disagree   NYT prefers leave, CP prefers score
 3    FOVEOLE(22) HOOF(27)   HOOF(27)      agree      Both say HOOF; you missed it
 5    HAKU(57)    HAKU(57)   HAKU(57)      agree      All three agree
 7    RHO(14)     ROIL(17)   ROIL(17)      agree      Both say ROIL; you played safe
 9    SIDH(21)    HAVIOR(28) SHOD(21)      3-way      NYT=HAVIOR, CP=SHOD, you=SIDH
11    BAAL(30)    BEAL(30)   OBOL(30)      3-way      Same score, different leave preferences
13    IRATE(39)   TRAY(44)   IRATE(39)     disagree   KEY: CP agrees with you, not NYT
15    PYRROLS(62) PYRROLS(62) PRAY(47)     disagree   CP says PRAY by MC; NYT/you say PYRROLS
17    STRONGER(55) STRONGER(55) STRONGER(55) agree    All three agree
19    FRITTS(27)  FRITTS(27) FRITTS(27)    agree      All three agree (near-endgame)
21    EN(17)      EN(21)     EN@R3C8V(17)  disagree   CP endgame solver proves R3C8V is best
```

Agreement rate: Crossplay agrees with NYT on 5/11, disagrees on 6/11.

### Key findings

**Finding 1: Turn 21 (EN) — Crossplay endgame solver is correct.**
NYT says EN at R8C3H for 21 pts. Crossplay's exact 2-ply endgame solver
(bag=0, opponent rack known: ADGSTV?) shows EN at R3C8V (17 pts) is
better: net -15 vs -22. The 4-point raw score difference is overwhelmed
by opponent's response (GAEN 32 vs DAVY 43). This is a case where exact
endgame search beats greedy scoring.

**Finding 2: Turn 13 (IRATE vs TRAY) — leave-value dependent.**
NYT prefers TRAY (44 pts, leave EIO) over IRATE (39 pts, leave YO).
Crossplay's MC 2-ply ranks IRATE #1 (equity -1.5) vs TRAY #5 (equity
-6.6). The gap is +5.1 equity points — driven by leave valuation.
SuperLeaves Gen2 training is narrowing this gap (OY-EIO spread went from
+11.36 deployed to +4.75 at Gen2-170K). This finding may flip with
more training.

**Finding 3: Turn 15 (PYRROLS vs PRAY) — risk/reward tradeoff.**
Both you and NYT chose PYRROLS (62 pts, sweep bonus). Crossplay MC
ranks PRAY #1 (equity +21.4) vs PYRROLS #4 (equity +12.8). PRAY scores
15 less but keeps SOR leave and gives opponent only 29.9 avg response
vs 46.3 for PYRROLS. The sweep bonus doesn't compensate for the opened
board. This is a legitimate disagreement about risk tolerance.

**Finding 4: Turn 9 (SIDH vs HAVIOR vs SHOD) — three-way split.**
NYT says HAVIOR (28 pts), you played SIDH (21 pts, ranked #5 by CP),
CP says SHOD (21 pts). HAVIOR ranks #20 in MC at equity -23.3, far
below SHOD at -14.0. HAVIOR scores 7 more but draws a massive avg opp
response (45.2 vs 37.0) and has high variance (std 24.1). NYT appears
to overweight raw score here.

### What this reveals about accuracy

1. **Endgame solver is the strongest component.** Turn 21 is the only
   case where we can prove correctness (deterministic, full information).
   The solver got it right when NYT's greedy approach didn't.

2. **Leave evaluation drives mid-game disagreements.** Turns 13 and 15
   both hinge on how much credit to give good leaves. SuperLeaves
   training will likely shift some of these rankings.

3. **MC opponent modeling catches risk.** Turns 9 and 15 show the MC
   correctly penalizing high-score moves that open the board. NYT
   appears to use a simpler model that rewards raw score more.

4. **Opening moves are noise.** Turn 1 (PILAW vs PILAF, 2-point
   difference) and Turn 11 (BAAL vs BEAL vs OBOL, same score) are
   within noise of any evaluation method.

---

## SuperLeaves training plan

### What are SuperLeaves?

A precomputed table giving the empirical equity value of every possible
1-to-6 tile leave. Instead of a hand-tuned formula with ~20 parameters,
SuperLeaves is a lookup table with ~921K entries trained via self-play.

**Mathematical definition:**
```
L(leave) = E[best_score(leave + draw)] - E[best_score(random_rack)]
```

The leave value is how much better your next turn will be compared to
having a completely random rack. Good leaves are positive, bad leaves
are negative, average leaves are near zero.

### How Quackle/Maven/Macondo do it

1. Start with initial estimates (hand-tuned formula or zeros)
2. Play millions of self-play games using a greedy bot (score + leave)
3. Record each leave and its equity contribution
4. Update leave values via running average or back-propagation
5. Repeat for 5-6 generations with head-to-head validation

Quackle's single-tile values (Scrabble/TWL): blank=+25.6, S=+8.0,
Z=+5.1, X=+3.3, R=+1.1, E=+0.4, U=-3.5, V=-5.8, Q=-12.5.

### Leave representation

- Keys: sorted tuples, e.g. `('?', 'A', 'E', 'R', 'S', 'T')`
- Blanks are a distinct tile type `'?'`, not expanded to 26 letters
- Crossplay unique leaves: ~921K (sizes 1-6, respecting tile distribution)
- Storage: ~70MB as pickled dict, ~15-20MB compressed

### Training pipeline

```
superleaves/
  leave_table.py     # LeaveTable class (sorted tuple -> float dict)
  fast_bot.py        # Greedy bot: max(score + leave_value), no MC
  self_play.py       # Fast game loop for training data generation
  trainer.py         # Generation loop with running-average updates
  bootstrap.py       # Initialize from existing evaluate_leave()
  validate.py        # Head-to-head validation between generations
```

**Parameters:**
- 10M games per generation, 5 generations = 50M total
- 6 workers on i7-8700 = ~50-100 games/sec
- Estimated training time: ~6-12 days total
- Bootstrap from existing hand-tuned formula (warm start)

### Crossplay-specific considerations

- **3 blanks**: blank-in-leave value will be lower per-blank than
  Scrabble (less scarce, diminishing marginal value). Leave `('?',)`
  likely ~20-22 (vs Quackle's 25.6 for Scrabble).
- **40-point bingo**: bingo-heavy leaves lose ~10 pts vs Scrabble.
  Score-focused tiles (Z, X, K) become relatively more valuable.
- **No tile penalty in endgame**: skip leave observations when bag=0
  (leave is meaningless in Crossplay endgame). Down-weight observations
  when bag < 7 by factor `bag_size / 7`.
- **Different tile values**: K=6 (vs 5), W=5 (vs 4), J=10 (vs 8).
  SuperLeaves training captures these automatically.

### Integration

After training, SuperLeaves replaces the formula in `evaluate_leave()`:

```python
def evaluate_leave(leave, bag_empty=False):
    if not leave or bag_empty:
        return 0.0
    leave_key = tuple(sorted(leave.upper()))
    if _SUPERLEAVES and leave_key in _SUPERLEAVES:
        return _SUPERLEAVES[leave_key]
    return _formula_evaluate(leave, bag_empty)  # fallback
```

### Validation

1. Train on 100K games first (~20 min) as a smoke test
2. Verify size-1 ordering: blank > S > Z > X > R > E > U > V > Q
3. Run 100K head-to-head games: SuperLeaves bot vs formula bot
4. Target: >51% win rate for SuperLeaves to confirm improvement
