# Stage 7 Findings — Network Analysis

**Generated from:** `network_metrics.csv`, `super_attack_heatmap.png`, `playbook_communities.png`, `cooccurrence_graph.png`, `defense_priority_matrix.png`
**Graph:** 12 technique nodes (T01–T12), edges weighted by scored-prompt co-occurrence
**Communities:** Louvain, resolution tuned for 2 dense playbooks + 4 niche singletons

---

## 7A — Structural hubs

- **T01 Roleplay/Persona is the central hub of the attack surface.** Highest PageRank (0.2002) and highest weighted degree (2.80) — it co-occurs with more techniques and more often than any other node. 4,415 scored prompts (39% of the corpus) use it as their primary technique.

- **T04 Creative Writing punches above its weight on harm.** It is the #2 hub by PageRank (0.1563) but ranks #1 on the harm-weighted-PageRank / PageRank ratio (1.20). Translation: its edges carry disproportionately higher mean harm than its raw co-occurrence volume would predict. This is the clearest "block-first" target on the defense-priority matrix.

- **Authority Override (T07) rounds out the top-three hubs.** PageRank 0.1398, mean harm 0.152 — consistently harmful and structurally central.

---

## 7B — Playbook communities

- **Two dominant playbooks cover 97.5% of scored attacks.** Louvain recovers two large communities plus four singleton niches:
  - **Community 3 — Narrative Framing** (T01 Roleplay, T04 Creative Writing, T07 Authority Override, T12 Direct): 8,211 prompts (72.3%), weighted mean harm **0.164** — the larger, more harmful playbook.
  - **Community 2 — Reframing / Pressure** (T02 Hypothetical, T03 Academic, T11 Prompt Chaining, T08 Emotional Manipulation): 2,865 prompts (25.2%), weighted mean harm **0.114**.
  - **Singletons** (T05 Encoding, T06 Translation, T09 Multi-turn, T10 Technical/Code): 278 prompts (2.5%) combined — four low-frequency, isolated vectors that rarely co-occur with anything else.

- **The two playbooks are not equally dangerous.** Community 3 is ~44% more harmful per prompt than Community 2 (0.164 vs. 0.114). Defender priority ordering falls out of the structure itself.

---

## 7C — Bridges between playbooks

- **T08 Emotional Manipulation is the dominant bridge — by a wide margin.** Betweenness centrality **0.327** vs. next-highest 0.073 (T03 Academic). T08 has modest raw frequency (369 prompts) but sits on most shortest paths linking the Narrative-Framing and Reframing-Pressure communities.

- **Implication:** T08 is a high-leverage defender target. Disrupting it fragments cross-playbook recipes even though blocking it alone removes only ~3% of the corpus. The other 10 techniques have betweenness of 0 or near-0 — they are in-community players, not connectors.

---

## 7D — Super-attacks (pair-lift)

- **Attack recipes are category-specific.** The pair-lift heatmap (`super_attack_heatmap.png`) shows that the technique pair driving Hate is not the pair driving Self-harm or Sexual-Minors — each harm category has its own set of amplifying combinations.

- **Super-attacks cluster along the hub axis.** Bright cells (lift > 1.5) concentrate in rows/columns anchored on the Community-3 hubs (T01, T04, T07), consistent with their harm-weighted-PageRank dominance — the high-centrality techniques are also the ones whose pairings most reliably exceed the corpus baseline.

- **Defender framing:** blocking a super-attack pair is worth substantially more than blocking either technique alone, because the lift is in the *combination*. This is what the pair-lift view adds on top of the single-technique leaderboard.

---

## 7E — Defense priority matrix

- **Top-right quadrant (high hub-residual × high mean-harm) is small and actionable.** The 2×2 (`defense_priority_matrix.png`) isolates techniques that are *more central than their frequency predicts* AND harmful: **T04 Creative Writing** is the clearest occupant, with **T01 Roleplay** and **T07 Authority Override** close behind.

- **T12 Direct/Unframed is the opposite case.** Despite being in the large playbook, it has the lowest mean harm (0.061) and a harm-PR / PR ratio of 0.81 — i.e. refusals work on direct asks. T12 is a *detection-is-cheap* technique, not a priority to block.

- **Singletons are low-priority by structure.** T05 Encoding, T06 Translation, T09 Multi-turn, T10 Technical/Code are each their own community; they are niche and rarely compose with other techniques. Handle with targeted rules, not playbook-level defenses.

---

## Slide bullets (for presentation)

- **Two playbooks dominate.** Narrative-Framing (72% of attacks, mean harm 0.16) and Reframing-Pressure (25%, 0.11). Everything else is niche.
- **One hub matters most.** T01 Roleplay is the highest-PageRank node; T04 Creative Writing punches above its weight on harm. Block-first targets.
- **One bridge matters most.** T08 Emotional Manipulation has 4× the betweenness of the next-closest technique. Disrupting it fragments cross-playbook recipes.
- **Super-attacks are category-specific.** No universal "worst pair" — Hate, Self-harm, and Sexual-Minors each have their own amplifying combinations.

---

## Caveats

- **WildJailbreak bias.** 79% of scored prompts come from WildJailbreak's synthetic compositions, which over-represents Narrative-Framing. Pattern-level findings (hubs, bridges, community structure) generalize; absolute frequency counts should be read with this in mind.
- **Co-occurrence edges encode *sequential* technique use within prompts**, not session-level multi-turn behavior. T09 Multi-turn is a singleton here because the dataset is single-prompt rows.
- **T08 bridge finding rests on n=369.** Directionally robust (betweenness is 4× the runner-up) but not invariant to resampling.

---

## Deliverables

- [x] `network_metrics.csv` — per-technique PageRank, harm-weighted PageRank, betweenness, weighted degree, community assignment
- [x] `cooccurrence_graph.png` — force-directed render
- [x] `playbook_communities.png` — same graph coloured by Louvain community
- [x] `super_attack_heatmap.png` — pair-lift heatmap faceted by harm category
- [x] `defense_priority_matrix.png` — hub-residual × mean-harm 2×2
- [x] `network_findings.md`
