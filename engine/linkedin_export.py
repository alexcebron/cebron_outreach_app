import csv
from dataclasses import dataclass
from typing import List, Optional


# =========================
# DATA MODELS
# =========================

@dataclass
class Company:
    name: str
    website: Optional[str] = None
    linkedin_url: Optional[str] = None
    location: Optional[str] = None          # City / region / country (if known)
    industry: Optional[str] = None          # e.g. "Cybersecurity", "MedTech"
    description: Optional[str] = None       # Short blurb from website/LI
    size_range: Optional[str] = None        # e.g. "51-200 employees"
    founded_year: Optional[int] = None      # e.g. 2017


@dataclass
class Prospect:
    # Person
    first_name: str
    last_name: str
    full_name: str
    job_title: str
    linkedin_url: str
    location: Optional[str]

    # Fit / scoring
    acquisition_relevance_score: int        # 0-100
    why_chosen: str

    # Company reference
    company: Company

    # Output message for LinkedIn connection note
    connection_message: str


# =========================
# STEP A: COMPANY INTELLIGENCE (STUB)
# =========================

def fetch_company_data(raw_input: str) -> Company:
    """
    Step A: Given a company name / URL / LinkedIn URL,
    return a normalized Company object.

    For now this is a stub with simple heuristics.
    Later you'll plug in:
      - Google/SerpAPI search
      - Website scraping
      - LinkedIn company lookup
    """

    website = None
    name = raw_input

    if raw_input.startswith("http"):
        website = raw_input
        domain = raw_input.replace("https://", "").replace("http://", "")
        domain = domain.split("/")[0]
        name = domain.split(".")[0].replace("-", " ").title()

    company = Company(
        name=name,
        website=website,
        linkedin_url=None,
        location=None,
        industry=None,
        description=None,
        size_range=None,
        founded_year=None,
    )

    return company


# =========================
# STEP B: PROSPECT INTELLIGENCE (STUB)
# =========================

def find_relevant_prospects(company: Company) -> List[Prospect]:
    """
    Step B: Given a Company (with at least name, ideally linkedin_url),
    identify 1–3 best prospects for acquisitions / finance / strategy.

    This is a stub right now.
    You'll later:
      - Query LinkedIn 'People' tab for this company
      - Pars
