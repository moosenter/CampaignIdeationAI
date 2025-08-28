import os, json, traceback
import streamlit as st

from config import MODEL_ID, DEFAULT_SCHEMA, CHANNEL_CATALOG, SYSTEM_PROMPT
from prompts import build_user_prompt
from utils import extract_first_json_block, normalize_budget_split, safe_load_json, json_after_assistant, align_plan_to_schema
from validators import validate_plan
from model_loader import load_llama
from generator import generate_json_plan

st.set_page_config(page_title="Campaign Ideation AI (Llama 3.1 8B)", page_icon="🧠", layout="wide")
st.markdown("<h1>🧠 Campaign Ideation AI</h1><p>Meta-Llama-3.1-8B-Instruct only.</p>", unsafe_allow_html=True)

def to_markdown(plan: dict) -> str:
    lines = []
    lines.append(f"# {plan.get('concept_title','(no title)')}")
    lines.append(plan.get("big_idea",""))
    lines.append(f"**Key message:** {plan.get('key_message','')}")
    lines.append("\n## Channels & Activations")
    for ch in plan.get("channels", []):
        name = ch.get("name","")
        act = ch.get("activation","")
        kpi = ", ".join([f"{k}: {v}" for k,v in (ch.get("kpis") or {}).items()])
        lines.append(f"- **{name}** — {act}  \n  KPIs: {kpi}")
    lines.append("\n## Assets")
    for a in plan.get("assets", []):
        lines.append(f"- {a}")
    lines.append("\n## Budget Split")
    for item in plan.get("budget_split", []):
        if isinstance(item, list) and len(item)==2:
            lines.append(f"- {item[0]}: {int(item[1]*100)}%")
    lines.append(f"\n## Timeline\n- {plan.get('timeline_weeks','?')} weeks")
    lines.append("\n## KPIs")
    for k,v in (plan.get("kpis") or {}).items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input("Local model path (optional)", value="", help="Leave empty to load from Hugging Face (requires HF token and access).")
    local_only = st.checkbox("Local files only (offline)", value=bool(model_dir))
    hf_token = st.text_input("HF token (needed for gated repo)", type="password", value=os.getenv("HF_TOKEN",""))
    max_new_tokens = st.slider("Max new tokens", 256, 2048, 1024, 64)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    st.markdown("---")
    st.caption("JSON schema (edit as needed)")
    schema_str = st.text_area("Schema", value=json.dumps(DEFAULT_SCHEMA, ensure_ascii=False, indent=2), height=240)

st.subheader("Brief")
with st.form("brief_form"):
    c1, c2, c3, c4 = st.columns([1.2,1,1,1])
    industry = c1.text_input("Industry", "FMCG snacks")
    geo = c2.text_input("Geo", "TH")
    age = c3.text_input("Age band", "18-24")
    budget = c4.number_input("Budget (THB)", min_value=50000, step=50000, value=1000000)

    c5, c6 = st.columns([1,1])
    objective = c5.selectbox("Objective", ["awareness","acquisition","retention","loyalty","upsell"], index=0)
    tone = c6.selectbox("Brand tone", ["playful","premium","trustworthy","innovative","minimal"], index=0)

    c7, c8 = st.columns([1,1])
    mandatory = c7.multiselect("Mandatory channels", CHANNEL_CATALOG, default=["LINE OA"])
    banned = c8.multiselect("Banned channels", CHANNEL_CATALOG, default=[])

    submitted = st.form_submit_button("Generate plan")

if submitted:
    with st.spinner("Loading model and generating..."):
        # Parse schema
        try:
            schema = json.loads(schema_str)
        except Exception as e:
            st.error(f"Invalid schema JSON: {e}")
            st.stop()

        # Build brief and prompt
        brief = {
            "industry": industry,
            "audience": {"geo": geo, "age": age},
            "budget_thb": float(budget),
            "objective": objective,
            "constraints": {"brand_tone": tone, "mandatory_channels": mandatory, "banned_channels": banned}
        }
        user_prompt = build_user_prompt(brief)

        # Load model (Llama 3.1 8B only)
        try:
            mdl_dir = model_dir or None
            tok, mdl = load_llama(model_dir=mdl_dir, local_files_only=local_only, hf_token=hf_token or None)
        except Exception as e:
            st.error(f"Model load error: {e}")
            st.stop()

        # Generate raw text
        try:
            raw = generate_json_plan(tok, mdl, SYSTEM_PROMPT, user_prompt, max_new_tokens, temperature, top_p)
        except Exception as e:
            st.error("Generation error")
            st.code(traceback.format_exc())
            st.stop()

        # Extract JSON
        print(raw)
        # cand = extract_first_json_block(raw) or raw
        # plan = safe_load_json(cand)
        plan = align_plan_to_schema(json_after_assistant(raw))
        if not plan:
            st.warning("Could not parse a clean JSON block; showing raw text.")
            st.code(raw)
            st.stop()

        # Normalize + validate
        normalize_budget_split(plan)
        ok, err = validate_plan(plan, schema)
        if not ok:
            st.warning("Plan generated but failed schema validation:")
            st.code(err)

        # Render
        c1, c2, c3 = st.columns([2,3,2])
        with c1:
            st.subheader(plan.get("concept_title","(no title)"))
            st.write(plan.get("big_idea",""))
            st.caption(f"Key message: {plan.get('key_message','')}")
        with c2:
            st.markdown("**Channels & Activations**")
            rows = []
            for ch in plan.get("channels", []):
                name = ch.get("name","")
                act = ch.get("activation","")
                kpi = ", ".join([f"{k}: {v}" for k,v in (ch.get("kpis") or {}).items()])
                rows.append(f"- **{name}** — {act}  \n  KPIs: {kpi}")
            st.markdown("\n".join(rows) or "_No channels_")
        with c3:
            st.markdown("**Budget split**")
            for item in plan.get("budget_split", []):
                if isinstance(item, list) and len(item)==2:
                    st.write(f"- {item[0]}: {int(item[1]*100)}%")
            st.write(f"**Timeline:** {plan.get('timeline_weeks','?')} weeks")
            kpis = plan.get("kpis", {})
            if kpis:
                st.markdown("**KPIs**")
                for k, v in kpis.items():
                    st.write(f"- {k}: {v}")

        with st.expander("Full JSON"):
            st.code(json.dumps(plan, ensure_ascii=False, indent=2), language="json")

        md = to_markdown(plan)
        st.download_button("Download JSON", data=json.dumps(plan, ensure_ascii=False, indent=2), file_name="campaign_plan.json", mime="application/json")
        st.download_button("Download Markdown", data=md, file_name="campaign_plan.md", mime="text/markdown")