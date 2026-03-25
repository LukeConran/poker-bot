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

## Chapter 6: The Right Tool for Hidden Information

The honest ceiling at the end of Chapter 5 wasn't a dead end — it was a diagnosis. DQN's problem isn't something you fix with more hands or better hyperparameters. It's structural.

DQN was designed for games where the agent sees the full state. In Atari, the pixels on screen are everything. In poker, they're not. Your opponent's hole cards are hidden, which means the same observable state — same board, same pot, same position — can lead to opposite correct decisions depending on what they're holding. DQN can't represent that uncertainty. It assigns one Q-value to each action in each observed state and optimizes it, which works fine when the state is complete and falls apart when it isn't.

### Taking Stock of the Options

RLCard ships four learning algorithms. Before committing to a direction, it was worth understanding what each one actually does and what it costs:

**Deep Monte-Carlo (DMC)** samples full game trajectories and trains a network on average outcomes. Simple, no game tree required, scales reasonably to multi-player. Good empirical results in RLCard's own benchmarks.

**Deep Q-Learning (DQN)** — where we are now. Learns Q-values, epsilon-greedy exploration, single network. Fast to train, easy to understand, but fundamentally unsuited to imperfect information without modification.

**Neural Fictitious Self-Play (NFSP)** layers game theory on top of DQN. The agent maintains two policies simultaneously: a best-response network (RL, like DQN) and an average strategy network (supervised). The average strategy is what converges — over many iterations, it approaches Nash equilibrium. Specifically designed for imperfect information games. Compute cost is roughly 2x DQN.

**Counterfactual Regret Minimization (CFR)** is the gold standard — the algorithm behind Libratus and Pluribus. It iteratively minimizes regret (the gap between what you did and what you should have done) and provably converges to Nash equilibrium. The catch: vanilla CFR doesn't scale to No-Limit Hold'em. Production CFR systems require game abstraction, which is a research problem in its own right. Without access to a high-performance compute cluster, CFR at meaningful scale is not realistic.

### The Format Problem

There's another unknown: the competition format. If the final evaluation is heads-up (1v1), NFSP's convergence guarantees apply directly — it's designed for two-player zero-sum games. If it's a pool of bots (3-6 players), Nash equilibrium becomes ill-defined; the better choice there is DMC, which doesn't rely on two-player theory and scales naturally.

With the format uncertain, NFSP is the safer bet. It handles 1v1 well by construction, and in a multi-player pool it will still learn solid poker — just without the theoretical guarantees. If the format clarifies as multi-player, the switch to DMC is a one-file change.

### How NFSP Actually Works

Every hand, the agent flips a biased coin. With probability η (typically 0.1), it plays its **best-response policy** — acting greedily to maximize chip gain against the current opponent. With probability 1−η, it plays its **average policy** — sampling from the distribution it has averaged over all its past best-response decisions.

The best-response policy is trained with RL (off-policy, like DQN) on a standard replay buffer. The average policy is trained with supervised learning on a reservoir buffer that stores every action the agent took while in best-response mode.

The result: the average policy is always "what would I do if I had to commit to one mixed strategy that covers all the ways I've played?" Over time, this converges to Nash equilibrium. An opponent can't exploit it because there's no fixed pattern to exploit.

### The Training Design

The new training loop (`train_nfsp.py`) uses two NFSP agents — `nfsp0` (the agent we care about) and `nfsp1` (its sparring partner) — and runs them through two phases:

**Phase 1 — 10,000 hands vs RandomAgent (warm-up)**
Two untrained agents playing each other from random weights can be slow to develop meaningful poker. A brief warm-up against a random opponent gives `nfsp0` a clear gradient signal to learn basic hand strength before self-play begins. Phase 1 is short — 10k hands, not the 100k we used for DQN curriculum — because NFSP self-play generates signal far more efficiently.

**Phase 2 — 200,000 hands of self-play**
Both agents train against each other. Every hand, both `nfsp0` and `nfsp1` get their transitions fed and their networks updated. The opponent keeps getting stronger as `nfsp0` does, which is the point — self-play is a natural curriculum where the difficulty scales with your ability.

The `PokerBotAgent` is no longer a training target. It becomes a benchmark: every 5,000 hands, we pause and measure `nfsp0`'s average payoff against it. A positive trend there is a signal that the agent is developing real poker intuition, not just learning to beat its own mirror image.

---

## References

[1] Zha, D., Lai, K. H., Cao, Y., Huang, S., Wei, R., Guo, J., & Hu, X. (2019). RLCard: A Toolkit for Reinforcement Learning in Card Games. *Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI-20)*. <https://github.com/datamllab/rlcard>
