import os
from datetime import datetime, timedelta
from urllib.parse import urlparse
import json
import random

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ------------------------
# ENV & OPENAI CLIENT
# ------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in environment or .env.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------------
# BASIC CONFIG
# ------------------------

st.set_page_config(
    page_title="Cebron Outreach Engine",
    page_icon="🧠",
    layout="wide",
)

DEFAULT_PROFILE_OWNER = "Oliver Lewis"


# ------------------------
# PROMPT TEMPLATES
# ------------------------

SYSTEM_INSTRUCTIONS_COMPANY = """
You are an AI research assistant helping identify companies and acquisition-relevant contacts
for a mid-market M&A advisory firm called Cebron Group.

Your job in THIS MODE:
1. Given a short description or niche, identify 5–20 relevant companies.
2. For each company, find:
   - Name
   - Website
   - Industry
   - Approximate size or revenue (if you can)
   - City/region/country (if possible)
   - A key contact who is MOST likely to be responsible for:
       * M&A
       * Corporate development
       * Strategic partnerships
       * Or high-level financial / strategic decisions (CFO, Head of Strategy, etc.)
   - That contact's:
       * Name
       * Title
       * LinkedIn URL
       * Email (ONLY if it is confidently available from a public, legitimate source – otherwise leave blank).
3. Return results in JSON strictly following the schema.
"""


SYSTEM_INSTRUCTIONS_MESSAGE = """
You are an AI assistant helping generate hyper-targeted, concise LinkedIn connection notes
for senior executives about strategic acquisitions.

Your job:
1. Read the company profile and the contact's background.
2. Write a short, natural, non-salesy LinkedIn connection message that:
   - references something specific about the company or contact's role
   - briefly indicates that we focus on strategic acquisitions / roll-ups
   - suggests that it may be relevant to them WITHOUT sounding like a canned pitch
   - is 280 characters or less
   - has no emojis, no exclamation marks overused, and no fake flattery.

Tone:
- Professional, calm, and slightly curious.
- No hard CTAs like "book a call" – more like "open to comparing notes" or
  "happy to share what we're seeing in your space."
"""


SYSTEM_INSTRUCTIONS_SUMMARIZE = """
You are a concise summarizer. Given a bunch of text or structured info,
rewrite it into a clean, factual, short summary used to personalize outreach.
Max 1-2 sentences. No fluff, no emojis.
"""


SYSTEM_INSTRUCTIONS_HISTORY_RECAP = """
You are acting as an internal analyst reading a history of outreach and results.
Given a dataset of companies and contacts that have been previously researched,
you produce a short bullet-point summary of:
- Industries
- Typical contact roles/titles
- Average company size
- Any pattern about which titles *tend* to be chosen as the acquisitions decision-maker.
"""


# ------------------------
# UTILS
# ------------------------

def safe_get(d, key, default=""):
    if not isinstance(d, dict):
        return default
    val = d.get(key, default)
    return val if val is not None else default


def parse_domain(website_or_url: str) -> str:
    """
    Extract a clean domain from a URL like https://www.example.com/path
    -> example.com
    """
    if not website_or_url:
        return ""
    try:
        parsed = urlparse(website_or_url)
        netloc = parsed.netloc
        if not netloc:
            # maybe the input is just 'example.com'
            netloc = website_or_url.split("/")[0]
        # strip 'www.'
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return website_or_url


def canonical_company_key(name: str, website: str) -> str:
    """
    Canonical key for deduping companies across runs.
    """
    name = (name or "").strip().lower()
    domain = parse_domain(website or "").lower()
    return f"{name}|{domain}"


def load_json_file(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_json_file(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving {path}: {e}")


# ------------------------
# OPENAI HELPERS
# ------------------------

def openai_chat_json(
    system: str,
    user: str,
    response_format: dict = None,
    temperature: float = 0.4,
):
    """
    Generic helper for OpenAI ChatCompletions returning JSON-like structures,
    using the new 'response_format' mechanism.
    """
    kwargs = {
        "model": "gpt-5.1",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }

    if response_format:
        kwargs["response_format"] = response_format

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    content = choice.message.content

    if response_format:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            st.warning("AI returned non-JSON for a JSON-expected call; returning raw text.")
            return content
    else:
        return content


# ------------------------
# AI: COMPANY + CONTACT SEARCH
# ------------------------

COMPANY_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "website": {"type": "string"},
                    "industry": {"type": "string"},
                    "location": {"type": "string"},
                    "revenue": {"type": "string"},
                    "linkedin_url": {"type": "string"},
                    "contact": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "title": {"type": "string"},
                            "linkedin_url": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["name", "title"],
                    },
                },
                "required": ["name", "website", "contact"],
            },
        }
    },
    "required": ["companies"],
}


def ai_company_search(query: str, num_companies: int = 10):
    """
    Use OpenAI + web search to:
      - find relevant companies
      - identify key acquisitions-relevant contact for each
    Returns list of companies with contact embedded.
    """
    user_prompt = f"""
Find approximately {num_companies} companies that match this query or niche:

\"{query}\"

For each company:
- Provide the fields required by the JSON schema (companies[].name, website, industry, location, revenue, linkedin_url, contact).

Focus strongly on B2B operators that might fit a mid-market roll-up strategy
$10M - $500M revenue, or similar scale, when possible.
"""
    schema = {"type": "json_schema", "json_schema": {"name": "company_search_result", "schema": COMPANY_SEARCH_SCHEMA}}

    resp_data = openai_chat_json(
        system=SYSTEM_INSTRUCTIONS_COMPANY,
        user=user_prompt,
        response_format=schema,
        temperature=0.4,
    )

    if isinstance(resp_data, dict) and "companies" in resp_data:
        return resp_data["companies"]
    return []


# ------------------------
# AI: MESSAGE GENERATION
# ------------------------

def ai_generate_message(company: dict, profile_owner: str = DEFAULT_PROFILE_OWNER):
    """
    Generate a short LinkedIn connection note using the AI-based
    system instructions and company/contact info.
    """
    name = company.get("name", "")
    industry = company.get("industry", "")
    location = company.get("location", "")
    revenue = company.get("revenue", "")
    contact_name = company.get("contact_name", "")
    contact_title = company.get("contact_title", "")

    user_prompt = f"""
We are {profile_owner} at Cebron Group, a mid-market M&A advisory and roll-up platform builder.

Company:
- Name: {name}
- Industry: {industry}
- Location: {location}
- Approx. revenue/size: {revenue}

Contact:
- Name: {contact_name}
- Title: {contact_title}

Write a concise LinkedIn connection note (under 280 characters, no emojis) that:
- Mentions something specific about the company or their role
- Indicates that we do strategic acquisitions / roll-up work
- Suggests it could be relevant for them
- Does not sound like a mass blast or pushy sales pitch
"""

    msg = openai_chat_json(
        system=SYSTEM_INSTRUCTIONS_MESSAGE,
        user=user_prompt,
        response_format=None,
        temperature=0.6,
    )

    msg = (msg or "").strip()
    if msg.startswith('"') and msg.endswith('"'):
        msg = msg[1:-1].strip()
    if msg.startswith("'") and msg.endswith("'"):
        msg = msg[1:-1].strip()

    if len(msg) > 280:
        msg = msg[:270].rstrip() + "…"

    msg = " ".join(msg.split())
    msg = msg.replace("...", ".")
    while ".." in msg:
        msg = msg.replace("..", ".")

    msg = msg.replace("?", ".")

    for token in ["http://", "https://", "www."]:
        if token in msg:
            msg = msg.split(token)[0].strip()
    if "@" in msg:
        msg = msg.split("@")[0].strip()

    return msg


def generate_message(company: dict, profile_owner: str = DEFAULT_PROFILE_OWNER):
    """
    Fallback simple template-based message if AI is disabled or errors out.
    """
    name = company.get("name", "")
    industry = company.get("industry", "")
    contact_name = company.get("contact_name", "")
    contact_title = company.get("contact_title", "")

    pieces = []
    if contact_name:
        pieces.append(f"Hi {contact_name},")
    else:
        pieces.append("Hi there,")

    base = f"I'm {profile_owner} at Cebron Group. We help operators in {industry or 'your space'} explore strategic acquisitions and roll-up strategies."
    pieces.append(base)

    role_hint = contact_title.lower() if contact_title else ""
    if any(k in role_hint for k in ["corp dev", "m&a", "mergers", "acquisitions", "strategy"]):
        pieces.append("Given your role, I thought it might make sense to connect and share what we're seeing in the market.")
    else:
        pieces.append("Thought it might make sense to connect and share what we're seeing in the market.")

    msg = " ".join(pieces)
    return msg


# ------------------------
# AI: OPTIONAL HISTORY RECAP
# ------------------------

def ai_history_recap(all_results: list):
    """
    Summarizes the existing master list in a short bullet set.
    """
    df = pd.DataFrame(all_results)
    sample_text = df.to_string(max_rows=50)

    prompt = f"""
Here is a sample of the existing records of companies and contacts we've researched:

{sample_text}

Please provide 4-6 bullet points that summarize:
- common industries
- typical titles / roles we are targeting
- approximate scale of companies
- any patterns about who tends to own M&A or acquisitions.

Be concrete, but concise.
"""
    resp = openai_chat_json(
        system=SYSTEM_INSTRUCTIONS_HISTORY_RECAP,
        user=prompt,
        response_format=None,
        temperature=0.3,
    )
    return resp


# ------------------------
# INDUSTRY HELPER
# ------------------------

def normalize_industry(i: str) -> str:
    """
    Very rough normalization into the categories we're generally interested in.
    """
    if not i:
        return ""
    s = i.lower()
    if "cyber" in s or "security" in s:
        return "cybersecurity"
    if "software" in s or "saas" in s:
        return "software / SaaS"
    if "health" in s or "med" in s or "life science" in s:
        return "health / medtech / life sciences"
    if "manufactur" in s or "industrial" in s:
        return "industrial / manufacturing"
    if "logistics" in s or "supply chain" in s:
        return "logistics / supply chain"
    if "professional services" in s or "consulting" in s:
        return "professional services"
    return i


def industry_to_outreach_angle(industry: str) -> str:
    s = (industry or "").lower()
    if "cyber" in s:
        return "cybersecurity platform and managed security operators"
    if "software" in s or "saas" in s:
        return "software and SaaS platforms"
    if "health" in s or "med" in s:
        return "health and medtech operators"
    if "manufactur" in s:
        return "industrial and manufacturing operators"

    return f"operators in {industry}"


# ------------------------
# CONTACT SELECTION HEURISTICS
# ------------------------

def _score_title_for_acquisitions(title: str) -> int:
    """
    Higher score = more likely to be involved in acquisitions / M&A / corp dev.
    """
    t = (title or "").lower()

    score = 0
    if any(k in t for k in ["corp dev", "corporate development", "corporate strategy"]):
        score += 60
    if any(k in t for k in ["m&a", "mergers", "acquisitions"]):
        score += 50
    if any(k in t for k in ["vp", "vice president", "director", "head of"]):
        score += 20
    if "chief" in t or "cfo" in t or "finance" in t:
        score += 30
    if "strategy" in t:
        score += 20
    if "ceo" in t or "founder" in t or "coo" in t or "president" in t:
        score += 10

    if "assistant" in t or "associate" in t or "analyst" in t:
        score -= 10
    if "sales" in t or "marketing" in t and "strategy" not in t:
        score -= 5
    if "intern" in t:
        score -= 50

    return score


def _choose_best_contact(company: dict):
    """
    If we have multiple candidates for a company, pick the best by title score.
    Right now we expect a single "contact" field, but we can extend to
    contact_candidates later.
    """
    if not company:
        return None

    candidates = []
    primary = company.get("contact") or {}
    if primary:
        candidates.append(primary)

    contact_candidates = company.get("contact_candidates") or []
    for c in contact_candidates:
        candidates.append(c)

    if not candidates:
        return None

    scored = []
    for c in candidates:
        title = safe_get(c, "title", "")
        score = _score_title_for_acquisitions(title)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_contact = scored[0]
    return best_contact, best_score, [c for _, c in scored]


# ------------------------
# INITIAL DATA LOAD FROM JSON FILES
# ------------------------

def load_initial_data():
    """
    Load existing companies + contacts + outreach results from JSON files into all_results.
    - Base: cebron_companies.json
    - Enrich contacts: cebron_contacts.json
    - Add extra companies from cebron_outreach_results.json if not present
    """
    base_path = "/mnt/data"
    companies_path = os.path.join(base_path, "cebron_companies.json")
    contacts_path = os.path.join(base_path, "cebron_contacts.json")
    outreach_path = os.path.join(base_path, "cebron_outreach_results.json")

    all_results = []

    companies_data = load_json_file(companies_path) or []
    contacts_data = load_json_file(contacts_path) or []
    outreach_data = load_json_file(outreach_path) or []

    companies_map = {}
    for c in companies_data:
        name = safe_get(c, "name")
        website = safe_get(c, "website")
        ck = canonical_company_key(name, website)
        if ck not in companies_map:
            companies_map[ck] = {
                "name": name,
                "website": website,
                "industry": safe_get(c, "industry"),
                "location": safe_get(c, "location"),
                "revenue": safe_get(c, "revenue"),
                "company_linkedin_url": safe_get(c, "linkedin_url"),
                "source": "companies_json",
                "added_at": safe_get(c, "added_at") or datetime.utcnow().isoformat(),
            }

    for contact in contacts_data:
        company_name = safe_get(contact, "company")
        website = safe_get(contact, "website")
        ck = canonical_company_key(company_name, website)

        if ck not in companies_map:
            companies_map[ck] = {
                "name": company_name,
                "website": website,
                "industry": safe_get(contact, "industry"),
                "location": safe_get(contact, "location"),
                "revenue": safe_get(contact, "revenue"),
                "company_linkedin_url": safe_get(contact, "company_linkedin_url"),
                "source": "contacts_json",
                "added_at": safe_get(contact, "added_at") or datetime.utcnow().isoformat(),
            }

        if "contact_name" not in companies_map[ck] or not companies_map[ck]["contact_name"]:
            companies_map[ck]["contact_name"] = safe_get(contact, "name")
            companies_map[ck]["contact_title"] = safe_get(contact, "title")
            companies_map[ck]["linkedin_url"] = safe_get(contact, "linkedin_url")
            companies_map[ck]["phone"] = safe_get(contact, "phone")
            companies_map[ck]["contact_email"] = safe_get(contact, "email")

    for rec in outreach_data:
        company_name = safe_get(rec, "company_name")
        website = safe_get(rec, "website")
        ck = canonical_company_key(company_name, website)

        if ck not in companies_map:
            companies_map[ck] = {
                "name": company_name,
                "website": website,
                "industry": safe_get(rec, "industry"),
                "location": safe_get(rec, "location"),
                "revenue": safe_get(rec, "revenue"),
                "company_linkedin_url": safe_get(rec, "company_linkedin_url"),
                "source": "outreach_json",
                "added_at": safe_get(rec, "added_at") or datetime.utcnow().isoformat(),
            }

        if "contact_name" not in companies_map[ck] or not companies_map[ck]["contact_name"]:
            companies_map[ck]["contact_name"] = safe_get(rec, "contact_name")
            companies_map[ck]["contact_title"] = safe_get(rec, "contact_title")
            companies_map[ck]["linkedin_url"] = safe_get(rec, "linkedin_url")
            companies_map[ck]["phone"] = safe_get(rec, "phone")
            companies_map[ck]["contact_email"] = safe_get(rec, "contact_email")

    for ck, val in companies_map.items():
        all_results.append(val)

    return all_results


# ------------------------
# STREAMLIT SESSION INIT
# ------------------------

if "all_results" not in st.session_state:
    st.session_state.all_results = load_initial_data() or []

if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = []

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "selected" not in st.session_state:
    st.session_state.selected = {}

if "profile_owner" not in st.session_state:
    st.session_state.profile_owner = DEFAULT_PROFILE_OWNER


# ------------------------
# STREAMLIT UI
# ------------------------

st.title("🧠 Cebron Outreach Engine (v1 – LinkedIn Focus)")

st.markdown(
    """
This app:
1. Lets you define or select a **niche / ICP** for acquisition targets.
2. Uses AI + web search to find **real companies** and **contacts** likely responsible for M&A / strategy / finance.
3. Generates **short LinkedIn connection notes**.
4. Lets you **select** contacts and **export** them as a CSV for tools like HeyReach.
"""
)

with st.sidebar:
    st.header("Settings")
    profile_owner = st.text_input(
        "Profile owner name (used in messages)",
        value=st.session_state.profile_owner,
    )
    st.session_state.profile_owner = profile_owner

    st.markdown("---")
    st.markdown("**AI options**")
    use_ai_messages = st.checkbox(
        "Use AI to generate connection messages",
        value=True,
        help="If unchecked, will use a simpler template-based message.",
    )

    st.markdown("---")
    st.markdown("**History recap**")
    if st.button("Summarize existing master list"):
        if not st.session_state.all_results:
            st.info("No master list data yet.")
        else:
            recap = ai_history_recap(st.session_state.all_results)
            st.markdown("### Master List Summary")
            st.write(recap)


# ------------------------
# ICP / NICHE SELECTION
# ------------------------

st.markdown("## 1️⃣ Select Your ICP / Niche")

cyber_icp = """
US-based cybersecurity operators between $10M–$150M in annual revenue.
Focus on: MSSPs, MDR providers, SOC-as-a-Service, incident response firms,
penetration testing companies, vCISO providers, threat intelligence platforms,
identity and access management (IAM), OT security, and managed EDR/XDR.

Target firms should have:
- established enterprise or mid-market customer base
- recurring revenue model (managed services or SaaS security)
- limited geographic footprint (consolidation upside)
- specialized IP, tooling, or certifications (SOC 2 / ISO 27001 / PCI DSS)
- opportunities for platform expansion or tuck-in integration

Exclude:
- consumer antivirus
- micro agencies
- early-stage startups (<$3M revenue)
"""

medtech_icp = """
US-based MedTech and HealthTech companies between $10M–$150M in annual revenue.
Focus on: medical devices, wearables, digital diagnostics, telehealth platforms,
remote patient monitoring (RPM), imaging software, surgical tools,
biometric data platforms, robotics-assisted systems, and clinical workflow SaaS.

Target firms should have:
- FDA-cleared or FDA pathway products (510k, De Novo, PMA)
- strong reimbursement alignment or CPT coding advantage
- established hospital/clinic customer base
- recurring device, consumable, or SaaS revenue

Exclude:
- pre-FDA startups
- biotech research firms
- pharma-only companies
"""

icp_choice = st.radio(
    "Choose an industry:",
    ["Cybersecurity", "MedTech", "Custom"],
    horizontal=True,
)

if icp_choice == "Cybersecurity":
    query = cyber_icp.strip()
elif icp_choice == "MedTech":
    query = medtech_icp.strip()
else:
    query = st.text_area(
        "Enter your ICP / niche (industry, size, geography):",
        placeholder="Describe your custom target niche...",
        height=140,
    )

st.text_area("Your ICP definition:", query, height=180, disabled=True)

num_companies = st.slider("Approx. number of companies to find:", 5, 30, 10)

col_search, col_clear = st.columns([3, 1])

with col_search:
    run_search = st.button("🔍 Search Companies + Contacts")
with col_clear:
    clear_results = st.button("Clear Last Search Results")


if clear_results:
    st.session_state.last_search_results = []
    st.session_state.selected = {}
    st.session_state.messages = {}
    st.experimental_rerun()


# ------------------------
# SEARCH EXECUTION
# ------------------------

if run_search and query.strip():
    with st.spinner("Running AI search for companies and contacts..."):
        companies = ai_company_search(query, num_companies=num_companies)

    results = []
    now_str = datetime.utcnow().isoformat()

    for c in companies:
        name = safe_get(c, "name")
        website = safe_get(c, "website")
        industry = normalize_industry(safe_get(c, "industry"))
        location = safe_get(c, "location")
        revenue = safe_get(c, "revenue")
        linkedin_url = safe_get(c, "linkedin_url")

        contact_obj, score, candidates = _choose_best_contact(c)

        contact_name = safe_get(contact_obj, "name")
        contact_title = safe_get(contact_obj, "title")
        contact_linkedin = safe_get(contact_obj, "linkedin_url")
        contact_email = safe_get(contact_obj, "email")

        result_rec = {
            "name": name,
            "website": website,
            "industry": industry,
            "location": location,
            "revenue": revenue,
            "company_linkedin_url": linkedin_url,
            "contact_name": contact_name,
            "contact_title": contact_title,
            "linkedin_url": contact_linkedin,
            "contact_email": contact_email,
            "score": score,
            "source": "ai_search",
            "added_at": now_str,
            "candidates": candidates,
        }

        st.session_state.all_results.append(result_rec)
        results.append(result_rec)

    st.session_state.last_search_results = results
    st.session_state.selected = {}
    st.session_state.messages = {}
    st.success(f"Found {len(results)} companies. See below.")
elif not query.strip() and run_search:
    st.warning("Please enter or select a description / niche to search for.")


# ------------------------
# RESULTS REVIEW
# ------------------------

st.markdown("---")
st.markdown("## 2️⃣ Review & Refine Companies + Contacts")

results = st.session_state.last_search_results

if not results:
    st.info("No search results yet. Select your ICP above and click 'Search Companies + Contacts'.")
else:
    for i, company in enumerate(results):
        with st.expander(
            f"{i+1}. {company.get('name', 'Unknown Company')}  "
            f"({company.get('industry', 'Unknown industry')}, "
            f"{company.get('location', 'Unknown location')})"
        ):
            col_main, col_side = st.columns([3, 2])

            with col_main:
                st.markdown("**Company Info**")
                st.write(f"Website: {company.get('website', 'N/A')}")
                st.write(f"Industry: {company.get('industry', 'N/A')}")
                st.write(f"Location: {company.get('location', 'N/A')}")
                st.write(f"Revenue (approx): {company.get('revenue', 'N/A')}")
                st.write(f"Source: {company.get('source', 'N/A')}")

                st.markdown("**Primary Contact (Chosen for Acquisitions Relevance)**")
                contact_name = company.get("contact_name", "")
                contact_title = company.get("contact_title", "")
                score = company.get("score", 0)

                st.write(f"Name: {contact_name or 'N/A'}")
                st.write(f"Title: {contact_title or 'N/A'}")
                st.write(f"Acquisitions relevance score: {score}")

                if company.get("linkedin_url"):
                    st.write(f"Contact LinkedIn: {company.get('linkedin_url')}")
                if company.get("company_linkedin_url"):
                    st.write(f"Company LinkedIn: {company.get('company_linkedin_url')}")

                st.markdown("**Candidate Contacts (for transparency)**")
                candidates = company.get("candidates") or []
                if not candidates:
                    st.write("(No additional candidates recorded.)")
                else:
                    st.markdown("**Candidate contacts (raw from AI, scored for acquisitions relevance):**")
                    for idx_c, cand in enumerate(candidates, start=1):
                        title = cand.get("title") or ""
                        name = cand.get("name") or ""
                        li = cand.get("linkedin_url") or ""
                        email = cand.get("email") or ""
                        cand_score = _score_title_for_acquisitions(title)

                        st.write(f"{idx_c}. {name} — {title} (score: {cand_score})")
                        if li:
                            st.write(f" • LinkedIn: {li}")
                        if email:
                            st.write(f" • Email (if confidently found): {email}")

            with col_side:
                selected_key = f"select_{i}"
                selected = st.checkbox(
                    "Select for export",
                    value=st.session_state.selected.get(i, False),
                    key=selected_key,
                )
                st.session_state.selected[i] = selected

                st.markdown("**Connection Message**")
                if i not in st.session_state.messages:
                    if use_ai_messages:
                        msg = ai_generate_message(company, profile_owner=profile_owner)
                    else:
                        msg = generate_message(company, profile_owner=profile_owner)
                    st.session_state.messages[i] = msg

                msg_key = f"message_{i}"
                new_msg = st.text_area(
                    "Edit message as needed:",
                    value=st.session_state.messages[i],
                    key=msg_key,
                    height=100,
                )
                st.session_state.messages[i] = new_msg


# ------------------------
# EXPORT
# ------------------------

st.markdown("---")
st.markdown("## 3️⃣ Export")

if results:
    st.markdown("### Export Selected for HeyReach")

    export_rows = []
    for i, company in enumerate(results):
        if st.session_state.selected.get(i) and i in st.session_state.messages:
            export_rows.append({
                "profile_owner": profile_owner,
                "company_name": company.get("name", ""),
                "contact_name": company.get("contact_name", ""),
                "contact_title": company.get("contact_title", ""),
                "linkedin_url": company.get("linkedin_url", ""),
                "company_linkedin_url": company.get("company_linkedin_url", ""),
                "phone": company.get("phone", ""),
                "contact_email": company.get("contact_email", ""),
                "message": st.session_state.messages[i],
                "website": company.get("website", ""),
                "industry": company.get("industry", ""),
                "location": company.get("location", ""),
                "revenue": company.get("revenue", ""),
                "added_at": company.get("added_at", ""),
                "exported_at": datetime.utcnow().isoformat()
            })

    if not export_rows:
        st.warning("No selected contacts with generated messages yet.")
    else:
        export_df = pd.DataFrame(export_rows)
        csv_data = export_df.to_csv(index=False).encode("utf-8")

        st.success(f"Ready to export **{len(export_rows)}** contacts.")
        st.download_button(
            label="⬇️ Download CSV for HeyReach (current batch)",
            data=csv_data,
            file_name="cebron_linkedin_outreach.csv",
            mime="text/csv",
        )

        # Minimal LinkedIn CSV export for generic LinkedIn tools
        minimal_rows = []
        for row in export_rows:
            full_name = row.get("contact_name", "") or ""
            parts = full_name.split()
            first_name = parts[0] if parts else ""
            last_name = " ".join(parts[1:]) if len(parts) > 1 else ""
            minimal_rows.append({
                "company_name": row.get("company_name", ""),
                "company_website": row.get("website", ""),
                "company_linkedin_url": row.get("company_linkedin_url", ""),
                "contact_first_name": first_name,
                "contact_last_name": last_name,
                "contact_full_name": full_name,
                "contact_title": row.get("contact_title", ""),
                "contact_linkedin_url": row.get("linkedin_url", ""),
                "location": row.get("location", ""),
                "acquisition_score": _score_title_for_acquisitions(row.get("contact_title", "") or ""),
                "connection_message": row.get("message", ""),
            })

        if minimal_rows:
            minimal_df = pd.DataFrame(minimal_rows)
            minimal_csv_data = minimal_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Minimal LinkedIn CSV (generic format)",
                data=minimal_csv_data,
                file_name="cebron_linkedin_minimal.csv",
                mime="text/csv",
            )


# ------------------------
# HISTORY / MASTER LIST VIEW
# ------------------------

st.markdown("---")
st.markdown("## Existing Master List (History)")

all_results = st.session_state.all_results

if not all_results:
    st.write("No master results yet. Run a search or import a CSV to populate this.")
else:
    df_all = pd.DataFrame(all_results)

    if "added_at" not in df_all.columns:
        df_all["added_at"] = datetime.utcnow().isoformat()

    timeframe = st.radio(
        "Show records from:",
        ["Today", "Last 7 days", "Last 30 days", "All time"],
        horizontal=True,
        key="history_timeframe",
    )

    now = datetime.utcnow()

    if timeframe == "Today":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif timeframe == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif timeframe == "Last 30 days":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = None

    if cutoff:
        df_all["added_at_dt"] = pd.to_datetime(df_all["added_at"], errors="coerce")
        df_filtered = df_all[df_all["added_at_dt"] >= cutoff].copy()
    else:
        df_filtered = df_all.copy()

    st.write(f"Showing {len(df_filtered)} companies out of {len(df_all)} total.")

    if not df_filtered.empty:
        st.dataframe(
            df_filtered[
                [
                    "name",
                    "industry",
                    "location",
                    "revenue",
                    "contact_name",
                    "contact_title",
                    "company_linkedin_url",
                    "phone",
                    "added_at",
                ]
            ],
            use_container_width=True,
        )
