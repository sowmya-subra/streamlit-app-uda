I'm working on a graduate analytics project called GuardRail. The goal is to map the structural patterns in AI jailbreak attacks for a hypothetical Trust & Safety client. Due Apr 22, 2026, 10-min presentation.

The data pipeline is complete. I'm working with a master Parquet file (11,354 rows, 26 columns) of jailbreak prompts from three sources:
- `wildjailbreak` (9,000 rows): synthetic adversarial prompts from AI2's WildTeaming pipeline
- `toxicchat` (1,152 rows): real user inputs from the Vicuna chat demo, human-annotated
- `trustairlab` (1,202 rows): hand-curated jailbreak prompts from Reddit/Discord communities

Every row has:
- A normalized `prompt_text`
- A primary `technique_type` (T01-T12) classified by Claude Sonnet 4.6 — 92% human agreement on spot-check
  - T01 Roleplay/Persona, T02 Hypothetical, T03 Academic, T04 Creative Writing, T05 Encoding, T06 Translation, T07 Authority Override, T08 Emotional Manipulation, T09 Multi-turn, T10 Technical/Code, T11 Prompt Chaining, T12 Direct/Unframed Request
- An `owasp_category` (LLM01/05/06/07/09/10) from keyword rules
- 11 `harm_*` columns with OpenAI Moderation per-category scores (0.0–1.0). Coverage is 75% — 25% of rows have all-zero scores because the API refused to evaluate the most extreme prompts (this is itself a finding)
- `harm_max` and `harm_mean` summary columns
- `human_jailbreak_label` (0/1/NaN) for the Toxic-Chat and TrustAIRLab subsets
- `sentiment_compound`: VADER's composite emotional valence, range −1 (most negative) to +1 (most positive), computed per prompt. Full coverage (all 11,354 rows, zero NaN). 
`politeness_score`: A rule-based rate: count of polite tokens (please, thank, kindly, appreciate, could, would, sorry, etc.) divided by total tokens × 100. So it's "polite tokens per 100 words." Full coverage, zero NaN. 
`lda_topic`: Integer topic ID (0 through k−1, where k was selected from {6, 8, 10} by coherence score). Argmax of the LDA topic distribution, so each prompt gets its single dominant topic.

My specific task is: **Goal**: A 5–7-tab interactive demo that showcases the findings. Used during slide 9 of the presentation. 

Tabs (priority order — cut from the bottom if time runs short) 
    1. Attack Surface Map — pyvis network, colored by community, sized by PageRank 2. Technique × Policy Heatmap — harm score by (technique, harm category) 3. Super-Attack Finder — pair lift heatmap, filterable by harm category 4. Politeness Paradox — sentiment distribution by harm quartile 5. Topic Explorer — embed the pyLDAvis HTML 6. Jailbreak Risk Scorer — text input → predicted harm scores from Stage 8 model 7. Ask GuardRail — GraphRAG demo (skip if Stage 9 is cut) 

Implementation 
    - Single app.py with streamlit imports 
    - Load master.parquet and any derived files at startup with @st.cache_data 
    - Each tab is a function 
    - Deploy to Streamlit Cloud or run locally for the demo

Constraints I should know about:
- For any harm-score-dependent analysis, filter to `df[df['harm_max'] > 0]` (8,487 rows) — the rest have no scores
- For confidence-sensitive analysis, filter to `df[df['technique_confidence'] == 'high']` (9,629 rows)
- T12 isn't a "garbage" category — it's a real finding (direct unframed harmful requests, concentrated in real-user data)
- WildJailbreak is synthetic compositions, so findings about specific phrasing don't generalize but findings about tactic patterns do
- Don't modify the master parquet; write derived outputs to a separate file

Please help me analyze the current app.py set up and set up a framework to showcase the project.