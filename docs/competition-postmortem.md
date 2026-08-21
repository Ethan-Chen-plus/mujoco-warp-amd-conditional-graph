# AMD AI DevMaster Submission Postmortem

The public submission set does not expose a single official award list in the
repository. The useful comparison is therefore the evidence structure of the
strongest public PRs, not the amount of website content.

## Patterns worth carrying forward

- [RadeonHome, PR #64](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/pull/64): one mobile-manipulation story, task-level results, recovery behavior, source code, report, and a runnable package.
- [Radeon MJX LEAP Hand, PR #193](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/pull/193): long training, deterministic evaluation, a fixed checkpoint, and a complete version matrix.
- [Radeon Motion Lab, PR #265](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/pull/265): runtime and appearance matrices, ablations, retained failures, and a compact evidence package.
- [SafeHighway-ROCm, PR #222](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/pull/222): separates GPU kernel speedup from end-to-end speed and identifies the CPU bottleneck.

The shared pattern is simple: a narrow technical claim, a fixed protocol, a
reproducible artifact, and a result table that a reviewer can verify quickly.
The MuJoCo-Warp work follows the same structure: one graph capability, one
task family, one baseline pair, and explicit correctness and performance gates.

The official [submission README](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/blob/main/README.md)
requires the technical report, dedicated source repository, reproducibility
instructions, and a complete demo workflow. This PoC is an engineering
follow-up and is kept separate from the closed competition submission.
