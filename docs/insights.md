# Key Insights & Lessons Learned

These are things that came up while building this project — not rules pulled
from a textbook, but specific problems that broke training, cost time to diagnose,
and changed how I thought about the next decision. Some of them probably have
cleaner explanations in the literature. This is just what I ran into and what
seemed to fix it.

---

## 1. The Reward Function Can Look Fine and Still Be Wrong

The most disorienting failure in this project was a reward function that was
reasonable on paper but produced completely broken behavior in practice.

The specific problem was that the sum of penalty terms — living cost, gimbal
penalties, LOS rate penalty — ended up larger in magnitude than the progress
signal at the point where the agent spent most of its time. So the agent learned
to do nothing. Zero gimbal command, no movement, let the episode time out. It
was mathematically optimal given the reward structure.

The way I caught it was logging every reward component separately in the `info`
dict and just looking at the numbers during early training:

```python
info["reward_components"] = {
    "progress": 8.0 * progress_norm,
    "living_cost": -0.01,
    "gimbal_cmd": -0.01 * abs(gimbal_rate),
    "los_rate": -0.05 * abs(los_rate / pi),
}
```

Once I could see which terms dominated, the fix was obvious — scale progress up
until it clearly outweighs everything else combined. What I'd watch for going
forward: if the agent isn't moving toward the objective in the first few hundred
episodes, break the reward apart before assuming the algorithm is the problem.

---

## 2. Good Training Metrics Don't Mean the Behavior Is Right

Both major physics bugs in this project produced *high rewards*. The agent was
succeeding at the task. But watching the rendered output made it immediately
clear something was wrong — the body orientation, the flame direction, and the
actual velocity vector were all pointing in different directions at the same time.

**The thrust-redirect bug:** The gimbal angle was being added directly to the
thrust vector, which let the agent redirect force without actually rotating the
body. Physically that's not how TVC works — gimbal deflection creates torque,
and torque rotates the body, and the body rotation is what eventually changes
the thrust direction. The agent had essentially given itself free steering.

**The crossflow glide:** Once the thrust bug was fixed, the agent figured out
it could build lateral velocity through prior steering and then coast sideways
toward the target. In a vacuum that would arguably be valid. In atmosphere, it
isn't — air resists velocity components that don't align with the body axis.
Adding crossflow drag closed it.

I wouldn't have caught either of these from the reward curve alone. The numbers
looked fine. Rendering was what showed the problem. I don't know if there's a
general lesson there beyond: watch what the agent is actually doing, not just
whether the reward is going up.

---

## 3. The Normalization Stats and the Model Have to Stay in Sync

This one took a while to figure out because the symptom — erratic, broken-looking
behavior at evaluation — looked like a policy problem, not an infrastructure problem.

SB3's `VecNormalize` keeps a running mean and variance over observations throughout
training. `EvalCallback` saves the best model mid-training when it hits a new peak.
But the `.pkl` normalization stats file gets written at the *end* of training. If
those two points are far apart — say, best model saved at 240k steps, training ends
at 1M — the statistics are different. Loading the best model weights with end-of-training
stats means every observation the policy receives is scaled differently from what it
was trained on.

The fix was a custom callback that saves a paired stats file every time a new best
model checkpoint is written:

```python
class SaveVecNormalizeCallback(BaseCallback):
    def _on_step(self):
        if self.eval_callback.best_mean_reward > self._last_best:
            self._last_best = self.eval_callback.best_mean_reward
            self.model.get_vec_normalize_env().save("best_models/vec_normalize_best.pkl")
        return True
```

And the load pattern that actually works at evaluation:

```python
raw_env = DummyVecEnv([lambda: RocketInterceptEnv(...)])
env = VecNormalize.load("vec_normalize_best.pkl", raw_env)
env.training = False
env.norm_reward = False
```

The short version: model and stats are a pair, not independent files. They have
to be saved and loaded together.

---

## 4. The Agent Will Find Whatever Shortcut Exists

Every exploit in this project was discovered by the agent, not anticipated beforehand.
The thrust redirect, the lateral glide, the inaction strategy, the tumbling behavior
on steep targets, the environment bug where the evasive target would freeze — all
of them showed up in training before I knew they were possible.

The full list of things that had to be closed:
- **Thrust redirect** — gimbal deflecting thrust directly instead of creating torque
- **Crossflow glide** — coasting laterally without pointing toward the target
- **Inaction** — outputting zero to avoid penalties
- **Tumbling** — spinning aggressively to accidentally reach steep targets
- **Target freeze** — an environment bug where the evasive target's dodge direction became unstable under missile micro-steering

I don't think there's a way to fully anticipate these ahead of time. The agent
just finds them. The habit I developed was: when behavior looks weird or suspiciously
good, assume there's a shortcut being exploited and go looking for it rather than
assuming the policy is doing something clever.

---

## 5. Not All Curriculum Transitions Are Equal

Fine-tuning from a previous stage's checkpoint worked well when the new stage was
a harder version of the same task — the policy mostly needed to refine what it
already knew. It didn't work as well when the new stage required a different
strategy entirely.

The clearest case was moving from a static target to an evasive one. The static-target
policy had built its whole internal model around a target that doesn't change position.
Presenting it with a target that actively dodges perpendicular to the missile's
approach wasn't a refinement — it was a restructuring. Fine-tuning conservatively
from that checkpoint produced slow, unstable learning.

What seemed to help was treating those transitions more like fresh training —
near-scratch entropy coefficients, accepting that the policy is mostly starting over
with a slightly informed initialization rather than building on what came before.

I'm not confident this is a universal rule, but in this project the transitions that
required strategy changes needed more aggressive hyperparameters than the ones that
were just harder versions of the same problem.

---

## 6. What Goes Into the Observation Space Seems to Shape What the Agent Can Learn

This is something I understood abstractly going in but felt more concretely after
iterating on it. Two observations added later in the project seemed to make a real
difference:

**LOS rate** — the angular velocity of the line connecting the missile to the target.
I added this after reading about proportional navigation, which is a classical
guidance approach that uses the rate of change of the line-of-sight angle as its
core signal. I included LOS rate as an observation and added a small penalty for
nonzero LOS rate in the reward. After that the policy started producing curved
intercept trajectories that looked a lot like lead pursuit. Whether that's directly
because of the LOS rate addition or something else that changed at the same time,
I can't say with certainty — but it was a noticeable behavioral shift.

**Closing velocity** — the rate at which distance to the target is decreasing.
On a moving target, raw distance change is noisy — the target drifting sideways
changes your distance even if you made a correct control decision. Closing velocity
seemed to give the policy a cleaner signal about whether it was actually making
progress toward intercept.

The thing I noticed generally: when I looked at what real guidance systems actually
measure and tried to include analogues of those signals, training seemed to go
better than when I was just including whatever was convenient. I don't know how
much of that is specific to this problem versus something that generalizes, but
it's something I'd think about more carefully on a future project.

---

## 7. The "No Cheating" Constraint Was Harder but Worth It for a Different Reason

From the beginning, the observation space was restricted to signals a real onboard
sensor could physically measure — no privileged information, no future target
positions, no ground-truth target acceleration. Only what a seeker head and IMU
would actually see.

In hindsight, a more practical approach for sim-to-real transfer probably would
have been to train with full simulation privileges first — clean, perfect observations
— solve the task cleanly, and then add sensor noise progressively as a hardening
stage. This is roughly what domain randomization does: expose the policy to a
distribution of sensor qualities during training so the real noisy sensor looks
like one more sample it's already seen.

The "no cheating" approach made training harder and probably slower to converge.
But forcing myself to justify every observation from first principles — why does
LOS rate matter, what does closing velocity actually measure, what can a real seeker
physically see — meant I had to actually understand the guidance problem rather than
just feed the agent everything available and let it sort out what's useful.

I'm not saying one approach is strictly better. For pure training efficiency,
privileges first then noise hardening is probably cleaner. But the constraint did
produce a kind of domain understanding I'm not sure that I would have gotten the other way.

---

## 8. Human Evaluation Catches Things Metrics Don't

Hit rate against a scripted evasion target is useful but limited — the scripted
target runs the same pattern regardless of what the missile is doing. It can't
adapt, so a policy that learns to beat the script might not be doing anything
that generalizes.

Adding a human control mode — where WASD keys drive the target in real time —
gave a much more honest picture. A human can watch the missile's trajectory,
anticipate where it's going, and make deliberate evasive decisions based on what
it's actually doing. It's not a rigorous benchmark (small sample size, one person),
but it tests something the scripted evaluation can't: whether the pursuit behavior
holds up against someone actively trying to break it.

The stage 3 policy landed at least one hit in every 10-episode run I attempted.
It took somewhere between 13 and 15 full runs — roughly 130 to 150 episodes —
before I managed a complete run with zero hits. That result wouldn't have been
visible in the scripted hit rate number alone.

It's also a cheap thing to add — one extra branch in the environment's step
function — so worth doing early.

---

## 9. The Boring Infrastructure Stuff Actually Matters

A few things that ended up being more important than they looked going in:

- **Reward component logging** — being able to see each term separately in the
  `info` dict is what caught the inaction problem before it wasted a full training run
- **Synchronized VecNormalize checkpointing** — without the paired stats callback,
  every evaluation against the best model checkpoint was producing misleading results
- **Curriculum Mode A/B** — having a clean separation between fresh training and
  fine-tuning, with different hyperparameter defaults for each, made stage transitions
  less error-prone
- **Manual rollouts before training** — running the environment with random actions
  for a few episodes to visually sanity-check behavior before committing to hours of training
- **`check_env()`** — SB3's built-in environment validator catches interface issues
  before they cause silent errors during training

None of these are interesting algorithmically. All of them saved time at some point.
The ones I added reactively after a problem appeared cost more time than the ones
I had from the start.