# Subliminal transfer research plan

A buildless, responsive planning website focused on proposal questions 1 and 4.

Serve locally from this directory:

```bash
python -m http.server 8765 --directory dist
```

Open http://localhost:8765. You can also open `dist/index.html` directly: the observed data is embedded, so all interactions work without a server.

- `dist/index.html`: full protocols, decision gates, schedule, and paper allocation.
- `dist/app.js`: scenario plots, observed heatmap, browser-local decisions, CSV/Markdown exports.
- `dist/existing-results.json`: inspected branch summary artifacts; **not rerun**.
- `PLAN.md`: text version of the website, useful for subsequent experiment sessions.
- `.openai/hosting.json`: private Sites project identity and static output declaration.

All figure mockups are explicitly synthetic. The width heatmap and three summary metrics come from existing GitHub artifacts on `varun/mnist-subliminal`. No new training was run. UI choices do not execute experiments. The site was checked for source validity and interaction behavior; visual browser QA was unavailable. The user approved private publication. Hosting: https://subliminal-learning-research-plan.onwardpenguin3.chatgpt.site No installation or build is required.
