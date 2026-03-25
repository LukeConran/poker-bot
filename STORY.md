# Poker Bot — The Story So Far

## Chapter 1: Start With the Math

Every poker decision comes down to one question: are you getting the right price?

We started there. Before thinking about bluffing frequencies, opponent tendencies, or game theory, we asked what the minimum viable poker bot looks like — one that makes decisions grounded in actual probability rather than gut feel or arbitrary rules.

The answer was a **probability-based bot** built on two core ideas:

**Hand equity** — given your hole cards, the community cards, and the number of opponents, what percentage of the time do you win? We calculate this via Monte Carlo simulation: deal out thousands of random runouts, count the wins, divide. It's not fast, but it's correct.

**Pot odds** — given that equity, is calling (or raising) a positive expected value decision? If the pot is offering you 3:1 and you win more than 25% of the time, you call. If not, you fold. Simple.

From those two primitives, a decision tree falls out naturally:

- No bet to face: if equity is strong, bet. Occasionally bluff. Otherwise check.
- Facing a bet: if equity beats the pot odds threshold and clears a minimum floor, call. If equity is dominant, raise. Otherwise fold.

Raise sizing is equally straightforward — 75% of the pot by default, a standard value bet size.

### What We Deliberately Left Out

This bot does not model opponents. It doesn't notice that one player raises every single hand, or that another only bets when they have the nuts. It doesn't look at bet sizing relative to pot — a 10x overbet and a half-pot bet are treated identically, just as a number to call. It doesn't track aggression history across streets. It has no concept of position beyond what's passed in manually.

These are real gaps. A human poker player adjusts constantly based on this information. Our bot ignores all of it and just does the math.

That was intentional. Get the foundation right first.

### The Stack

- `game_state.py` — dataclass holding hole cards, board, pot, stack, to_call, position, and opponent count
- `equity.py` — Monte Carlo equity estimator
- `hand_evaluator.py` — thin wrapper around [treys](https://github.com/ihendley/treys) for fast hand evaluation
- `pot_odds.py` — pot odds, EV, and raise sizing calculations
- `bot.py` — decision logic tying it all together
- `tests/` — 75+ unit tests covering hand rankings, equity bounds, pot odds math, and edge cases

---

## Chapter 2: You Can't Learn From One Hand

A bot that makes one decision in isolation isn't something you can improve. To get better, it needs to play — thousands of hands, against an opponent, tracking what happened.

That's where reinforcement learning enters the picture.

The idea is simple in principle: run the bot against itself repeatedly, reward decisions that led to winning chips, penalize decisions that didn't, and gradually shift the policy toward better play. The hard part is the infrastructure — you need a full game engine, a self-play loop, and a way to record what happened.

Rather than building a poker engine from scratch, we reached for **RLCard** [1] — a toolkit from Rice University's Data Analytics Lab designed exactly for this. RLCard provides a No-Limit Texas Hold'em environment with a clean agent interface: implement `step()` and `eval_step()`, set `use_raw = True` to get real card data, and the engine handles dealing, betting rounds, pot management, and payoffs.

### Bridging the Old and the New

The probability-based bot didn't get thrown away — it got wrapped. `rlcard_agent.py` is an adapter that:

1. Takes RLCard's raw observation (hole cards, board, chip counts)
2. Normalizes the card format (RLCard uses suit-first notation like `'HK'`; treys expects `'Kh'`)
3. Reconstructs a `GameState` our existing bot understands
4. Calls `Bot.decide()` to get a fold/call/raise decision
5. Maps that decision back to RLCard's action space (`FOLD`, `CHECK_CALL`, `RAISE_HALF_POT`, `RAISE_POT`, `ALL_IN`)

### The Training Loop

`train.py` runs two bot instances against each other for N hands using RLCard's environment, logging payoffs and trajectories to `results/hand_log.json`. Running two identical bots produces payoffs that oscillate around zero — expected, since it's a mirror match. The training loop is the scaffold; the learning agent comes next.

```
Hand   200 | P0 avg payoff (last 200): -2.21
Hand   400 | P0 avg payoff (last 200): -3.58
Hand   600 | P0 avg payoff (last 200): -1.46
Hand   800 | P0 avg payoff (last 200): +1.22
Hand  1000 | P0 avg payoff (last 200): +1.54
...
```

---

## What's Next

The foundation is in place. The natural next step is replacing one of the two mirror-match bots with a learning agent — RLCard's built-in DQN, or a custom policy network — and using the probability-based bot as the training opponent. As the learning agent improves, payoffs should diverge. That's when it gets interesting.

---

## References

[1] Zha, D., Lai, K. H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)*. https://github.com/datamllab/rlcard
