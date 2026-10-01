# Andai documentation

| Document | For | What's inside |
|---|---|---|
| [features.md](features.md) | Users, evaluators | What Andai does, feature by feature, and what is still a preview |
| [andai-website/](andai-website/index.html) | Public | Product landing page (static HTML — open `index.html` in a browser) |
| [../README.md](../README.md) | Everyone | What Andai is, download, getting started |
| [development.md](development.md) | Developers | How a turn works, build, run, test, release, the WebKit details |
| [../AGENTS.md](../AGENTS.md) | Contributors & coding agents | Grounding rules, architecture, conventions, testing, releases, security (§9) |
| [platform-facts.md](platform-facts.md) | Contributors & coding agents | Hard-won, measured facts about WebKit, Tauri, Windows, ug, wllama, MLX and Laya: what surprised us and what the code does about it (moved from AGENTS.md §2) |
| [security.md](security.md) | Users, evaluators, security reviewers | Every security protection, what it means for you, how it's tested, and what isn't covered yet |
| [decider.md](decider.md) | Contributors & coding agents | How a regular model decides the agent's next action: the choice-based prompt, the logprob readout, and a copy-paste prompt to reimplement it elsewhere |
| [agentic-rag-improvements.md](agentic-rag-improvements.md) | Contributors & coding agents | Tracker for planned agent-loop accuracy and speed work, one item at a time |
| [performance.md](performance.md) | Contributors, evaluators | Performance baselines: what is measured, the current numbers, allowed drift, and how to re-record them |
| [../SECURITY.md](../SECURITY.md) | Everyone | Security model in brief and how to report a vulnerability |

Docs describe the app **as it ships today**. If a change alters behavior, the
docs change in the same commit (see AGENTS.md §8).
