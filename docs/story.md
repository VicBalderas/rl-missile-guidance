# The Story of the Guided Rocket — An RL Journey

---

## Chapter 1: Starting From Zero

It started with a simple decision: learn reinforcement learning properly, not by copying a tutorial, but by actually understanding it from the inside out.

The first session opened with the basics — what an agent is, what an environment is, what a policy looks like. LunarLander from OpenAI Gymnasium was chosen as the training ground. Before touching any library, the entire PPO algorithm was built by hand in PyTorch. That meant writing the **Policy Network** from scratch — a simple MLP that takes in observations and outputs action probabilities. Then the **Value Network**, which learns to estimate how good a given state is. Then the **Memory Buffer**, which collects experience episode by episode. Then the **GAE (Generalized Advantage Estimation)** computation, which determines how much credit to assign to each action. Then the full **PPO update loop**, complete with clipped surrogate loss, value loss, and entropy regularization.

Every piece was explained and reasoned through — why the ratio between new and old log-probs gets clipped, why entropy gets added as a penalty to prevent premature convergence, why advantages get normalized before being fed into the loss. Nothing was treated as a black box.

The scratch implementation trained. It peaked, and then it collapsed — classic policy gradient instability when hyperparameters aren't tuned. But that was fine. The point was understanding what was happening *inside* the algorithm, not winning the benchmark. That goal was achieved.

---

## Chapter 2: Moving to SB3 and Solving the Baseline

With a firm grasp of how PPO works mechanically, the decision was made to move to Stable-Baselines3 for everything that came next. The reasoning was clean: the goal isn't to develop new RL algorithms — it's to apply reinforcement learning to control problems. SB3's PPO is battle-tested, properly parallelized, and lets you iterate on environments and systems rather than fighting numerical instability.

LunarLander was solved quickly. `n_envs=8` parallel environments, `n_steps=2048`, evaluation every 10,000 steps. The agent reached +159 average reward on evaluation — well past the solved threshold of +200 on individual episodes. Two episodes crossed +200 cleanly.

That established the pipeline: custom Gymnasium environment → VecNormalize for observation normalization → EvalCallback for checkpointing → PPO training. Everything that followed would build on this scaffolding.

---

## Chapter 3: The Real Problem — Build a Rocket That Intercepts a Target

The ambition escalated. Rather than rocket landing, the problem became something harder and more interesting: **guided missile intercept**. Design a rocket that can chase and hit a moving, eventually evasive target, using only gimbal control — no side thrusters, no throttle. One control axis. The rocket fires continuously until it runs out of fuel. The only thing the policy controls is the rate at which the nozzle deflects.

This required building a **custom 2D Gymnasium environment from scratch** in Python with Pygame rendering. A 900×650 pixel arena. The rocket spawns on the left; the target spawns somewhere in a configurable region. The physics had to be hand-coded: thrust force, gravity, angular velocity, gimbal torque, fuel consumption. The observation space had to be designed from scratch: relative position to target, relative velocity, heading error, LOS angle, closing velocity, obstacle proximity, fuel remaining.

And this is where everything got hard.

---

## Chapter 4: The Physics Were Wrong

The first version of the environment looked reasonable on paper. The rocket moved. The agent trained. But something was visually off — the body of the rocket, the direction of the flame, and the actual velocity vector all pointed in different directions simultaneously.

The bug was traced down: **the gimbal angle was being added directly to the thrust direction**. That meant the agent could deflect the nozzle and literally redirect where thrust pointed — as if thrust was a steerable jet, not a body-axis force. In reality, gimbal deflection only creates **torque** that rotates the body; the thrust itself always points along the rocket's current body axis. The policy had learned to exploit this broken physics to steer without actually rotating — a kinematic cheat disguised as flight.

The fix was surgical:

```python
# Thrust always along body axis — this is TVC
thrust_fx = thrust * cos(rocket_angle)
thrust_fy = thrust * sin(rocket_angle)

# Gimbal creates torque only
torque = gimbal_coeff * sin(gimbal_angle) * (thrust / max_thrust)
angular_velocity += torque * dt
```

This was version 3. Versions 1 and 2 had the exploit baked in.

---

## Chapter 5: The Policy Found Another Exploit

Fixing the thrust-redirect bug revealed a second problem immediately. With correct physics, the policy discovered it could build lateral velocity through prior steering, then *coast sideways* while pointing somewhere misleading. The rocket was gliding like a crab — building momentum in one direction and then drifting toward the target without pointing at it.

In a vacuum, this would be physically valid. In atmosphere, it isn't. Air resists velocity components that don't align with the body axis — this is **crossflow drag**, the aerodynamic weathervaning effect that forces aircraft to point roughly where they're going.

Crossflow damping was added: every physics step, velocity components perpendicular to the rocket's body axis are attenuated by 12%. The rocket now weathervanes toward its velocity direction. The exploit closed. The policy had to actually fly like a missile.

---

## Chapter 6: The Agent Learned to Do Nothing

With the physics fixed, training resumed — and the agent learned to do absolutely nothing. Zero gimbal command. No movement toward the target. Just sitting there collecting the least-bad reward.

This was a **reward signal inversion** problem. The reward function had several penalties: a living cost per step, gimbal rate penalties for aggressive control, gimbal angle penalties for sustained deflection, and an LOS rate penalty for guidance geometry. The problem was that the progress signal — the reward for getting closer to the target — was *smaller in magnitude than the sum of penalties at the dominant operating point*. So the optimal policy was inaction: produce zero output, collect zero penalties, and let the episode time out.

Diagnosing this required looking at the `reward_components` dict in the info output — breaking the reward into its constituent pieces and seeing which terms dominated. The fix was making progress unambiguously dominant:

- Progress reward scaled from `8.0 × progress_norm` to `25.0 × progress_norm`
- Hit terminal bonus raised from `+12.0` to `+30.0`
- All penalties reduced to soft nudges (`0.005` range instead of `0.05`)

The gradient now pointed unambiguously at the target. Training unstuck.

---

## Chapter 7: The Evaluation Was Lying

Training converged. The EvalCallback was saving a "best model" checkpoint mid-training. Evaluation was loaded from that checkpoint. And then the agent behaved erratically — looping, pointing in wrong directions, making nonsensical corrections — despite the training curves looking clean.

The bug took time to find. It wasn't the policy. It was **VecNormalize**.

During training, SB3 normalizes observations using a running mean and variance tracked in the VecNormalize wrapper. The best model checkpoint gets saved at, say, 240k steps. But the `.pkl` stats file gets saved at the *end* of training — at 1M steps. The stats at 1M steps are *different* from the stats at 240k steps. Loading the best model weights but feeding it observations scaled by the wrong statistics distorted every input dimension. The policy was receiving garbage inputs.

The fix was a **`SaveVecNormalizeCallback`** — a custom SB3 callback that saves a paired `vec_normalize_best.pkl` file *every time* a new best model is checkpointed. Model and stats are now always synchronized. And the correct load pattern was established for good:

```python
# CORRECT pattern:
raw_env = DummyVecEnv([lambda: RocketInterceptEnv(...)])
env = VecNormalize.load("vec_normalize_best.pkl", raw_env)
env.training = False
env.norm_reward = False
```

Never normalize on top of a normalized env. Never load stats independently from the model that was trained with them.

---

## Chapter 8: Teaching the Rocket Physics to Stay Within Physics

Another failure mode emerged on steep targets. When the target spawned nearly directly above the rocket, the agent would attempt the right move — rotate the nose upward — but then overshoot past vertical. Once past 90°, the thrust vector starts pointing partially *downward*, and gravity compounds the problem. The rocket would tumble into the ground trying to correct.

Two things caused this. First, there was no penalty for angular velocity — so the agent could spin the rocket aggressively with no cost. Second, the spawn geometry was too unconstrained, placing targets in physically unreachable positions given the rotation budget available before the rocket flew past them.

Both were fixed: an angular velocity penalty was added, and spawn ranges were tightened so the worst-case heading error at episode start was within what the physics could actually correct.

---

## Chapter 9: Curriculum Learning — Staged Training Across Five Difficulties

Hitting a static target is a different problem than hitting a moving target. Hitting a moving target is a different problem than hitting an evasive target that's actively trying to avoid you. Trying to train all of this at once from scratch produces nothing — the reward signal is too sparse to find.

The solution was **curriculum learning**: a 5-stage progression where each stage trains a meaningfully harder version of the problem, using the previous stage's model as a starting point.

| Stage | Target Behavior               | Obstacles | Challenge                            |
|-------|-------------------------------|-----------|--------------------------------------|
| 0     | Static                        | 0         | Basic intercept geometry             |
| 1     | Static-then-drift             | 0         | Reacquisition after target moves     |
| 2     | Vertical bounce (predictable) | 0         | Track 1D periodic motion, lead angle |
| 3     | Evasive (perpendicular dodge) | 1         | Pursue while target actively evades  |
| 4     | Aggressive evasion            | 2         | Sharp dodges + obstacle field        |
| 5     | Evasion + obstacle luring     | 2–3       | Target uses obstacles as shields     |

**Mode A** trains fresh from scratch. **Mode B** fine-tunes from the previous stage's checkpoint — the hyperparameters stay conservative when the behavioral delta is small, and near-scratch when the policy needs to restructure its strategy rather than just refine it. Moving-target stages required near-scratch entropy coefficients because the policy's whole decision structure had to change, not just the magnitude of its output.

---

## Chapter 10: The Target Was Cheating Back

With stages 3–5 redesigned around evasion, a new bug appeared: the target was occasionally **freezing in place** when the missile approached. Hit rates looked artificially good because the agent was catching a stationary target.

The evasion logic was traced: the target computed its perpendicular dodge direction from the vector `rocket_position - target_position`. When the missile made small corrective zigzags during approach, this position-relative vector oscillated rapidly between slightly different angles. The computed perpendicular directions kept swapping, the forces canceled, and the target froze indecisively.

The fix was elegant: compute the dodge direction from the missile's **velocity vector**, not its position. Velocity stays stable even under micro-steering oscillations because the missile is fundamentally heading *toward* the target regardless of small perturbations.

```python
# WRONG: flips with zigzag
approach_x = (rocket_x - target_x) / dist

# RIGHT: stable under micro-steering
approach_x = rocket_vx / missile_speed
```

After this fix, the target dodges cleanly and decisively every time.

---

## Chapter 11: Physics-Grounded Observations and Proportional Navigation

The observation space evolved from a simple "where is the target" representation toward something grounded in how real guidance systems actually work.

Two key observations were added late in the project:

**Closing velocity** — the rate at which distance to target is decreasing. This is equivalent to what a Doppler radar seeker measures. It separates "I'm moving fast toward the target" from "the target is drifting away."

**LOS rate (Line of Sight rate)** — the angular velocity of the line connecting missile to target. If LOS rate is zero, the missile is on a pure collision course — this is the foundational signal used in **Proportional Navigation**, the same guidance law used in real interceptor missiles. A small LOS rate penalty in the reward incentivized the policy to discover PN-like behavior emergently, without being explicitly programmed with the control law.

A deliberate **"no cheating" sensor philosophy** was maintained throughout: every observation in the space had to be something a real onboard sensor could physically measure. No future target positions, no privileged target velocity readings — only what a seeker head and IMU would actually see.

---

## Chapter 12: Where It Stands

After four environment versions, dozens of training runs, and a curriculum that spans five increasingly difficult stages, the policy reaches **~76% hit rate on stage 3** (evasive target with one obstacle), with entropy still exploring. Stage 4 — aggressive evasion with two obstacles — is the next frontier, with entropy coefficient dropped to ~0.005 to push toward exploitation.

The most recent training logs show entropy trending slightly upward during stage 3, which signals the policy is still searching. Stage 4 will compress that search space and demand committed pursuit decisions over random exploration.

But the most meaningful benchmark didn't come from a metric. It came from plugging in a human operator.

The environment includes a human control mode where WASD keys drive the target directly — the ultimate stress test, because a human can read the missile's trajectory, anticipate its arc, and make deliberate evasive decisions that no scripted target behavior can replicate. The stage 3 policy was put up against a human pilot across repeated 10-episode runs. The policy landed at least one hit in every single run. It took somewhere between 13 and 15 full runs — **130 to 150 episodes** — before the human operator achieved complete evasion across an entire set.

That number communicates something a hit-rate percentage can't: the policy had developed genuine pursuit behavior, not pattern matching against a scripted evasion routine. A human with full visual information, deliberate intent, and real-time adaptation still needed over a hundred attempts to fully survive it. That's the clearest possible signal that the underlying guidance logic is sound.

The long-term goal is unchanged: transfer this into the NVIDIA Isaac Sim / Isaac Lab ecosystem, where the same guidance problem can be tested in a GPU-accelerated physics simulation used in real robotics research.

---

## Chapter 13: A Realization — And the Next Frontier

After the curriculum was deep into development, a question surfaced: *would it have been easier to give the policy full simulation advantages first — perfect, clean, privileged observations — and then add noise later to harden it for the real world?*

The answer is yes. That's a well-established technique called **domain randomization**, and it's standard practice in sim-to-real robotics RL. The idea is straightforward: train with clean inputs until the task is solved, then inject noise across the observation space — Gaussian noise on sensor readings, quantization error on position estimates, lag on angular velocity updates — so the policy trains across a *distribution* of sensor qualities. When the real system's noisy sensor produces a reading, it just looks like one more sample from a distribution the policy has already seen. The policy generalizes rather than breaking.

Applied here, that would have meant: give the policy a clean, exact LOS rate during training, then later corrupt it with realistic seeker head noise. The policy learns to pursue the target correctly under that noise, so when a real sensor provides a noisy LOS rate measurement, the policy handles it the same way.

**But the decision to use real signals from day one wasn't a mistake — it was a different kind of education.**

By restricting the observation space to physically obtainable signals from the very beginning, the design process forced the same questions a guidance systems engineer has to answer: what can a real seeker actually measure? What does LOS rate mean physically, and why does proportional navigation use it? What's the difference between closing velocity and raw speed? Those questions had to be answered correctly to design the observation space at all — and answering them built a kind of domain understanding that privileged simulation shortcuts would have bypassed entirely.

The "no cheating" constraint made the project harder. It also made the learning more genuine.

**What comes next** is a hybrid of both philosophies. The physically grounded observations stay — LOS rate, closing velocity, relative velocity, heading error. These are the right signals. But once the curriculum completes, a noise injection training stage gets added on top: realistic sensor noise applied to every observation, drawn from distributions that approximate what real hardware would produce. Gaussian noise on LOS rate. Quantization error on position. Lag on angular velocity. The policy trains across that noise distribution until it becomes robust to it.

At that point, the sim-to-real gap narrows to something manageable. The real seeker produces noisy readings — but noisy readings the policy has been prepared for. The same numbers the policy was trained on in simulation, just with the imperfection of the physical world layered on top.

That's the next chapter: finish the curriculum, then battle-harden what was built.

---

## What Was Learned

This project forced mastery of a set of RL skills that can't be learned from a benchmark:

- **Reward shaping is fragile.** A reward function that looks reasonable on paper can be catastrophically broken in practice. The inaction exploit happened because penalties collectively outweighed progress, and the only way to catch it was decomposing the reward term by term.
- **Evaluation infrastructure matters as much as training.** The VecNormalize mismatch was one of the subtler bugs in the project — and it would have made every evaluation result meaningless if it hadn't been caught. Model and stats must always travel together.
- **Physics bugs produce high metrics for the wrong reasons.** The thrust-redirect exploit and the crossflow glide both produced agents that looked capable during training but were encoding physically impossible behavior. Watching the rendered output — not just the reward curve — is what caught both.
- **Curriculum design philosophy:** start clean, then add complexity. Sensor noise, obstacles, and evasion were all introduced in progression, never all at once.
- **Moving-target stages require near-scratch training.** Fine-tuning conservatively from a static-target policy doesn't work when the task demands a completely restructured strategy. The policy's value estimates, not just its action outputs, need to be rebuilt.
- **The agent will always find the path of least resistance.** Every design decision — physics, reward, spawn geometry, target behavior — has to be examined for exploitable loopholes, because the agent will find them before you do.
- **Harder constraints produce deeper understanding.** Restricting observations to physically realizable signals from day one forced genuine engagement with the guidance problem domain — the same questions a real systems engineer has to answer. The difficulty wasn't wasted; it became knowledge.