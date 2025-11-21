import streamlit as st
import pandas as pd
from datetime import datetime

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(
    page_title="Cebron Outreach Assistant",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cebron Outreach Assistant")
st.caption("Curated LinkedIn outreach builder for Oliver, Brandon & team.")


# ------------------------
# PLACEHOLDER: CORE LOGIC HOOKS
# ------------------------

def search_companies(keyword, industry, location, min_revenue, max_revenue, limit):
    """
    TODO: Replace this stub with the real search logic you and Hermes already built.
    This function should return a list of dicts with at least:
      - name
      - website
      - industry
      - location
      - revenue
      - contact_name
      - contact_title
      - linkedin_url
    """

    # Dummy data for now so the UI works immediately
    results = [
        {
            "name": "ExampleTech Solutions",
            "website": "https://exampletech.com",
            "industry": industry or "IT Services",
            "location": location or "US",
            "revenue": "25M",
            "contact_name": "Jane Doe",
            "contact_title": "CEO",
            "linkedin_url": "https://linkedin.com/in/jane-doe-example"
        },
        {
            "name": "Acme Cyber Defense",
            "website": "https://acmecyber.com",
            "industry": industry or "Cybersecurity",
            "location": location or "US",
            "revenue": "40M",
            "contact_name": "John Smith",
            "contact_title": "Founder & CEO",
            "linkedin_url": "https://linkedin.com/in/john-smith-example"
        },
    ]

    return results[:limit]


def generate_message(profile_owner, company, context_note):
    """
    TODO: Replace this stub with the Hermes-driven message generator
    you and I were using before (OpenAI call or your local logic).

    `company` will be a dict from search_companies().
    """
    name = company.get("contact_name", "there")
    company_name = company.get("name", "your company")
    industry = company.get("industry", "your space")

    base = (
        f"Hey {name}...\n\n"
        f"Oliver here. I’ve been looking at {company_name} and the way you’ve grown in {industry}.\n\n"
        f"We help founders use structured acquisitions to break through the scale ceiling "
        f"without giving up control or culture. "
        f"Curious if acquisitions are part of your playbook, or something you’ve considered but "
        f"haven’t had the right partner for.\n\n"
        f"Open to a brief conversation?"
    )

    if context_note:
        base += f"\n\n(Quick note from our internal review: {context_note})"

    return base


# ------------------------
# SESSION STATE
# ------------------------

if "results" not in st.session_state:
    st.session_state.results = []

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "selected" not in st.session_state:
    st.session_state.selected = {}


# ------------------------
# SIDEBAR FILTERS
# ------------------------

st.sidebar.header("Search Filters")

profile_owner = st.sidebar.selectbox(
    "Which profile is sending?",
    ["Oliver Lewis", "Brandon", "Other"],
)

keyword = st.sidebar.text_input("Keyword (sector / niche)", value="cybersecurity")
industry = st.sidebar.text_input("Industry", value="")
location = st.sidebar.text_input("Location", value="United States")

col_rev1, col_rev2 = st.sidebar.columns(2)
with col_rev1:
    min_revenue = st.text_input("Min revenue (e.g. 10M)", value="10M")
with col_rev2:
    max_revenue = st.text_input("Max revenue (e.g. 100M)", value="100M")

limit = st.sidebar.slider("Max companies to fetch", min_value=5, max_value=50, value=10, step=5)

st.sidebar.markdown("---")
st.sidebar.write("Click **Search** to pull a fresh list of companies.")


# ------------------------
# MAIN LAYOUT
# ------------------------

top_col1, top_col2 = st.columns([1, 2])

with top_col1:
    if st.button("🔍 Search Companies"):
        results = search_companies(keyword, industry, location, min_revenue, max_revenue, limit)
        st.session_state.results = results
        st.session_state.messages = {}
        st.session_state.selected = {i: False for i in range(len(results))}

with top_col2:
    st.info(
        "Workflow: 1) Set filters → 2) Search → 3) Approve companies → 4) Generate messages → 5) Export CSV for HeyReach."
    )

st.markdown("## Results & Message Builder")

results = st.session_state.results

if not results:
    st.write("No companies yet. Adjust filters and click **Search Companies**.")
else:
    df_preview = pd.DataFrame(results)[
        ["name", "industry", "location", "revenue", "contact_name", "contact_title"]
    ]
    st.dataframe(df_preview, use_container_width=True)

    st.markdown("---")
    st.markdown("### Review & Generate Messages")

    for i, company in enumerate(results):
        with st.expander(f"{i+1}. {company['name']} — {company.get('contact_name', '')} ({company.get('contact_title', '')})"):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"**Website:** {company.get('website', '—')}")
                st.write(f"**Industry:** {company.get('industry', '—')}")
                st.write(f"**Location:** {company.get('location', '—')}")
            with col_info2:
                st.write(f"**Revenue:** {company.get('revenue', '—')}")
                st.write(f"**Contact:** {company.get('contact_name', '—')}")
                st.write(f"**Title:** {company.get('contact_title', '—')}")
                if company.get("linkedin_url"):
                    st.write(f"[LinkedIn]({company['linkedin_url']})")

            selected = st.checkbox(
                "Include this contact in export",
                key=f"select_{i}",
                value=st.session_state.selected.get(i, False),
            )
            st.session_state.selected[i] = selected

            context_note = st.text_input(
                "Internal/context note (optional, not sent to prospect)",
                key=f"context_{i}",
                value="",
            )

            if st.button("✍️ Generate personalized message", key=f"gen_msg_{i}"):
                msg = generate_message(profile_owner, company, context_note)
                st.session_state.messages[i] = msg

            existing_msg = st.session_state.messages.get(i)
            if existing_msg:
                st.text_area(
                    "Generated LinkedIn message",
                    value=existing_msg,
                    height=220,
                    key=f"msg_area_{i}",
                )


    st.markdown("---")
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
                "message": st.session_state.messages[i],
                "website": company.get("website", ""),
                "industry": company.get("industry", ""),
                "location": company.get("location", ""),
                "revenue": company.get("revenue", ""),
                "exported_at": datetime.utcnow().isoformat()
            })

    if not export_rows:
        st.warning("No selected contacts with generated messages yet.")
    else:
        export_df = pd.DataFrame(export_rows)
        csv_data = export_df.to_csv(index=False).encode("utf-8")

        st.success(f"Ready to export **{len(export_rows)}** contacts.")
        st.download_button(
            label="⬇️ Download CSV for HeyReach",
            data=csv_data,
            file_name="cebron_linkedin_outreach.csv",
            mime="text/csv",
        )
