# Campaign Ideation AI — Plain-English README

## One-paragraph summary for slides

<!-- <video width="640" height="480" controls>
  <source src="data/campaign-2025-08-29_00.53.32.mp4" type="video/mp4">
</video> -->
![alt text](data/campaign-2025-08-29_00.53.32.gif "CampaignIdeationAI")

We built an AI assistant that converts a short marketing brief into activation-ready campaign concepts tailored to Thailand’s channels (e.g., LINE OA, TikTok). It outputs big idea, channel plan, budget split, timeline, and KPIs, in English or Thai, with validation and safety checks. This accelerates strategy work, standardizes deliverables, and keeps outputs compatible with CRM/loyalty activation—so teams move from idea to execution faster, with consistent quality.

-----

## 0) How to use
first install libs
```
pip install -r requirements.txt
```
then, download LORA weight from ggdrive and place to path `outputs/lora-llama31-8b`

https://drive.google.com/drive/folders/1eH0GBCtWiZQ8IkelrP_EKk5nozLAfPnQ?usp=drive_link

dont forget to input huggingface access token to be able to use the AI
### UI case
```python
streamlit run deploy/app.py 
```
### API case
```python
python3 deploy/api_app.py
```
port are set to be 8000 for localhost for testing run the following cmd
```python
python3 deploy/test_api.py
```
-----

## 1) What this is (in one line)

An AI assistant that turns a short marketing brief into ready-to-run campaign ideas—with channels, budget split, timeline, and KPIs—so teams can go from concept to activation faster.

------

## 2) Who it’s for

Marketing managers / brand owners who need polished ideas that fit Thailand’s channels (e.g., LINE OA, TikTok, Meta).

CRM & Loyalty teams who want campaigns that plug into a CDP/loyalty stack.

Account & strategy teams who need multiple options quickly to discuss with clients.

------

## 3) What you can do with it

Generate 1–3 complete campaign concepts from a brief (industry, audience, budget, objective).

Get a budget split, 6–8 week timeline, and measurement plan (KPIs).

Ensure ideas are activation-ready for CRM/omni-channel (e.g., LINE OA stamp card, coupon, broadcast).

Produce plans in English or Thai (same structure, translated copy).

Example (condensed):

- Concept title: Crave & Create
- Big idea: Turn snack breaks into creativity boosts
- Channels: TikTok UGC challenge; LINE OA stamp card + coupon
- Budget split: Media 60% | Creators 25% | Production 15%
- Timeline: 6 weeks (tease → UGC push → recap)
- KPIs: Reach 5M; Engagement 6%; OA Signups 15k; Redemptions 12k

------

## 4) How it works (simple)

You enter a brief

Industry, audience (e.g., “TH, 18–24”), budget (THB), objective (awareness/acquisition/loyalty), and any rules (must include LINE, banned channels, tone).

AI plans the campaign

The model uses examples and guardrails to output a single, structured plan (not free-form text).

Output is validated

We check the structure (fields exist, numbers make sense); then you can export/share.

Under the hood, the “AI engine” is a large language model fine-tuned on curated campaign examples plus synthetic scenarios. If internet-restricted, we can run it fully offline on a local server.

------

## 5) What states/outputs you’ll see

Concept(s) — title, one-liner, big idea

Channels — why this channel, how to activate (e.g., LINE OA coupon gate, TikTok creator collab)

Assets — video lengths, key visuals, copy items

Budget split — Media / Creators / Production (adds up to ~100%)

Timeline — weeks and milestones

KPIs — numbers matched to the objective (awareness ⇒ reach, promotion ⇒ redemption, etc.)

------

## 6) KPIs glossary (quick)

Reach — unique people who saw the content (not impressions).

CTR (Click-Through Rate) — clicks ÷ impressions × 100%.

CPL (Cost per Lead) — ad spend ÷ leads collected.

Redemption — number (or rate) of coupons actually used.

Membership_signup — new members added to CRM/loyalty.

(Thai hints): Reach = การเข้าถึง, Redemption = อัตราการใช้คูปอง, Membership signup = สมัครสมาชิกใหม่

------

## 7) Why this helps

Speed: Get solid starting ideas in minutes, not days.

Fit for TH market: Favors channels common in Thailand (e.g., LINE OA) and retail tie-ins.

Activation-ready: Output is already mapped to CRM/loyalty actions.

Consistency: Every plan contains the same key fields, so it’s easy to compare.

------

## 8) Data & privacy (safe by design)

The system does not require PII (personally identifiable information).

Briefs should be brand-level, not customer-level.

If we reference a CDP/CRM, we do it as capabilities (e.g., “has stamp card”)—not by loading customer data.

All generations are stored with audit logs so results are traceable.

------

## 9) Quality & safeguards

Schema validation: The plan must contain required sections (channels, budget split, KPIs).

Sanity checks: Budget splits sum to ~100%; timelines are reasonable; channels are available in TH.

Content safety: The generator follows acceptable-use rules (brand safety, no harmful content).

Human-in-the-loop: Final approval stays with the marketer; AI is a co-pilot, not an auto-publish tool.

------

## 10) Current limitations

Estimates, not guarantees: KPIs are planning targets, not actual results.

Creative samples: The tool proposes assets (e.g., “15s vertical video”) but does not design visuals.

Niche rules: Very specific brand/legal constraints may need manual editing.

Access-gated models: Some top models require approval; we provide open-model fallbacks so work never stops.

------

## 11) What’s been built already

A generation engine (AI model) guided by a business-friendly prompt + schema.

A dataset of structured campaign examples (mix of curated and scenario-based) to teach the model the format and tone.

Validation & guardrails so outputs are consistent and safe.

A simple API (so a web form or workflow tool can call it).

Bilingual support (English/Thai) with the same structure.

------

## 12) How you’ll use it (non-dev flow)

Open a brief form (web page or internal tool).

Fill: industry, audience, budget (THB), objective, tone, and any must-include/banned channels.

Click Generate → review 1–3 concepts.

Pick one, adjust copy if needed, and export (doc/slides/JSON for ops).

Hand off to creative/media teams or push to activation tools.

(If you don’t see the web form yet, the API is ready—your internal tools team can wire the form in a day.)

------

## 13) Examples of good briefs

“FMCG snacks | TH, Gen-Z (18–24) | 1,000,000 THB | Objective: awareness + trial | Must include: LINE OA; Tone: playful”

“Banking | TH, 25–45 | 2,000,000 THB | Objective: loyalty (card usage) | Ban: Twitter/X; Tone: trustworthy”

------

## 14) Roadmap (what’s next)

Co-creation loop: “Regenerate with stricter budget”, “Swap TikTok for YouTube”, etc.

Auto-costing: smarter budget suggestions based on historical media mixes.

Activation export: one-click handoff (e.g., LINE OA broadcast template, coupon rules).

Learning from outcomes: use post-campaign results (if available) to improve future suggestions.

------

## 15) FAQ

Q: Does it replace the strategist?
A: No. It speeds up the first 70% (structure, options), leaving humans to refine insights and creative.

Q: Can we control brand voice?
A: Yes—via “tone” settings and a small brand guide snippet; the model learns preferred phrasing.

Q: Does it need internet?
A: It can run fully offline on an internal server once the model files are installed.

Q: What if we need LINE-only or retail-only ideas?
A: Add that as a constraint; the generator will bias toward those channels.

------

## 16) Contacts / Ownership

Product owner: Visarut Trairattanapa

AI/Engineering: Lookmoo (Senior ML Engineer)

Support: mosenter@hotmail.com

-----