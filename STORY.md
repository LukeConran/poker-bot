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

## Chapter 3: Teaching a Bot to Learn

With the self-play scaffold in place, the mirror match gave way to something more interesting: a real learning agent.

We swapped one of the two identical bots out for RLCard's built-in **DQN (Deep Q-Network)** agent — a neural network that learns which actions lead to winning chips by playing thousands of hands and updating its weights based on the outcomes. The probability-based bot stayed in as the training opponent, giving the DQN a meaningful baseline to compete against rather than pure randomness.

### How DQN Works Here

The DQN takes a 54-dimensional encoded game state (52 cards one-hot encoded, plus two chip count features) and outputs Q-values for each of the five possible actions. It learns via the Bellman equation: if taking action A in state S eventually led to winning chips, increase the Q-value of A in S. Do this thousands of times and the network converges toward a policy that maximizes expected chip gain.

Key parameters:

- Two hidden layers of 64 neurons each
- Replay memory of 20,000 transitions — the agent samples randomly from past experience rather than learning from hands in order, which stabilizes training
- Epsilon-greedy exploration: starts at 100% random actions, decays to 10% over the course of training
- Separate target network that updates every 500 steps, preventing the Q-values from chasing a moving target

### The Wiring Problem

Getting DQN and our probability bot to coexist in the same environment wasn't trivial. RLCard supports mixed agent types, but each agent signals what kind of state it wants via a `use_raw` flag. DQN wants the encoded 54-dim array; our bot wants the raw card strings and chip counts.

The environment needs `allow_raw_data: True` in its config to serve both. We missed this initially — without it, our probability bot was receiving malformed state and effectively playing randomly during training. The DQN was learning to beat a broken opponent, then being evaluated against a working one. That explained the declining payoffs in our first run.

### First Results

The first training run (5,000 hands) showed declining performance — the DQN's average payoff against the baseline trended from -6 to -18 over the course of training. Two culprits: the `allow_raw_data` bug above, and 5,000 hands simply not being enough. DQN for poker needs tens of thousands of hands just to stabilize exploration before meaningful learning begins.

With the bug fixed and training extended to 20,000 hands, the epsilon decay now spans the full training window, giving the network time to explore broadly before committing to a policy.

### What Gets Saved

Checkpoints are saved to `results/checkpoints/` as `.pt` files (PyTorch format) every 5,000 hands, plus a final `dqn_final.pt` at the end. A hand log with per-hand payoffs is written to `results/hand_log.json` for analysis. To resume from a checkpoint or deploy the trained agent, load the weights back into a DQN instance with matching architecture.

---

## Chapter 4: DQN Doesn't Just Learn — It Needs to Crawl First

The first full DQN run finished. The results were not encouraging.

```
Hand  2000 | DQN avg payoff (last 500 eval hands): -10.18
Hand  4000 | DQN avg payoff (last 500 eval hands): -14.01
Hand  6000 | DQN avg payoff (last 500 eval hands): -11.50
Hand  8000 | DQN avg payoff (last 500 eval hands): -14.48
Hand 10000 | DQN avg payoff (last 500 eval hands): -11.79
Hand 12000 | DQN avg payoff (last 500 eval hands): -12.25
Hand 14000 | DQN avg payoff (last 500 eval hands):  -6.26
Hand 16000 | DQN avg payoff (last 500 eval hands):  -8.27
Hand 18000 | DQN avg payoff (last 500 eval hands): -10.67
Hand 20000 | DQN avg payoff (last 500 eval hands): -14.24

DQN win rate: 42.5%
```

Consistently negative. No upward trend. The loss never converged. After 20,000 hands and roughly 15 minutes of training, the DQN was performing worse than a coin flip.

### Why DQN Struggles at Poker

This isn't a bug — it's a known limitation of vanilla DQN applied to imperfect information games.

In a game like Atari (where DQN was designed), the agent sees the full game state. In poker, it doesn't. The opponent's hole cards are hidden, which means the Q-values the network is trying to learn are inherently noisy — the same board state with the same action can lead to wildly different outcomes depending on what the opponent is holding. The network can't distinguish between "I made a bad call" and "I made a correct call against bad luck."

On top of that, the reward signal is sparse. There are 4-8 actions per hand but only one reward at the end. Most transitions look like `(state, action, 0, next_state, False)` — no signal at all. The network has very little to learn from on a per-transition basis.

And 20,000 hands, while it sounds like a lot, is modest by poker AI standards. Production systems train on millions of hands.

### The Fix: Curriculum Learning

The core insight is that the DQN was thrown into the deep end immediately. Our probability-based bot, while simple, is a competent baseline opponent. A DQN starting from random weights has no chance against it — there's no gradient signal strong enough to pull it toward better play when it's losing every hand.

The solution is **curriculum learning**: start easy, then increase difficulty.

**Phase 1 — 10,000 hands vs a `RandomAgent`**
A random opponent folds, calls, and raises arbitrarily with no strategy. The DQN can beat this without much effort, which gives it a clear gradient signal to learn from. Basic concepts like "don't fold when you can check" and "raise with strong hands" should emerge quickly.

**Phase 2 — 20,000 hands vs `PokerBotAgent`**
Once the DQN has a working foundation, it fine-tunes against the probability bot. Crucially, it enters Phase 2 with existing weights rather than starting from scratch — so it's not learning poker from zero against a competent opponent, it's adjusting a policy that already beats random play.

The epsilon decay (exploration → exploitation) is spread across all 30,000 hands combined, so the DQN explores broadly in Phase 1 and gradually commits to its learned policy through Phase 2.

### What We Expect

Phase 1 payoffs should go positive — beating a random opponent isn't hard. The real test is whether Phase 2 payoffs improve over time rather than immediately collapsing, which is what happened when we threw DQN straight into the deep end.

---

## Chapter 5: Signs of Life

The curriculum run finished. 50,000 hands against a random agent, then 20,000 hands against the probability bot. Total training time: ~20 minutes.

**Phase 1 — vs RandomAgent (50,000 hands)**

```
Hand  2000 | avg payoff vs RandomAgent: +5.63
Hand  4000 | avg payoff vs RandomAgent: -0.75
Hand  6000 | avg payoff vs RandomAgent: +1.84
Hand 10000 | avg payoff vs RandomAgent: +5.50
Hand 20000 | avg payoff vs RandomAgent: +2.60
Hand 30000 | avg payoff vs RandomAgent: +3.37
Hand 38000 | avg payoff vs RandomAgent: +8.75
Hand 50000 | avg payoff vs RandomAgent: +3.59
```

Mostly positive throughout. The DQN learned to beat a random opponent — not a high bar, but a necessary one. It had something to work with going into Phase 2.

**Phase 2 — vs PokerBotAgent (20,000 hands)**

```
Hand  2000 | avg payoff vs PokerBotAgent: -14.12
Hand  4000 | avg payoff vs PokerBotAgent: -13.98
Hand  6000 | avg payoff vs PokerBotAgent: -15.31
Hand  8000 | avg payoff vs PokerBotAgent:  -9.86
Hand 10000 | avg payoff vs PokerBotAgent: -12.98
Hand 12000 | avg payoff vs PokerBotAgent: -10.00
Hand 14000 | avg payoff vs PokerBotAgent:  -7.65
Hand 16000 | avg payoff vs PokerBotAgent:  -7.07
Hand 18000 | avg payoff vs PokerBotAgent:  -9.45
Hand 20000 | avg payoff vs PokerBotAgent:  -5.70
```

Still negative — the DQN did not beat the probability bot. But the trend is real. It started at -14 and ended at -5.70. That's a 60% reduction in average loss over 20,000 hands. It's learning something.

**Overall win rate: 46.8%** (up from 42.5% without curriculum)

### What This Means

The gap is narrowing but hasn't closed. A few honest observations:

The DQN is still losing because the probability bot makes mathematically sound decisions on every hand, while the DQN is still partly exploring (epsilon hasn't fully decayed) and has only seen 20,000 hands against a competent opponent. The Q-values are noisy — poker outcomes are high variance, and each hand only yields one reward signal at the end.

The improvement from -14 to -5.70 across Phase 2 is meaningful. If the trend continued linearly, another 20,000 hands might bring it close to even. Whether it would ever go positive against this specific opponent is an open question — our probability bot is consistent and unexploitable in the sense that it always plays positive EV. A DQN can't exploit that; it can only try to match it.

### The Honest Ceiling

DQN for poker has a fundamental ceiling. It doesn't model the opponent, doesn't reason about hidden information, and can't represent mixed strategies. It's a reasonable starting point but not the end state. The next meaningful step up is NFSP — Neural Fictitious Self-Play — which learns an average strategy over time and is specifically designed for games where the opponent's cards are hidden.

---

## References

[1] Zha, D., Lai, K. H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)*. <https://github.com/datamllab/rlcard>
