#!/usr/bin/env python3
"""
Configuration for multi-relation extraction pipeline
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class RelationConfig:
    name: str
    source_entity_type: str
    target_entity_type: str
    relationship_type: str
    section_keywords: str
    chunk_keywords: str
    extraction_prompt_template: str
    entity_parser: Callable[..., Optional[Dict]]
    # Optional extra kwargs forwarded to entity_parser (e.g. main_company for OPERATES_IN)
    entity_parser_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Per-relation retrieval tuning (override global defaults)
    n_sections: int = 2
    n_chunks_per_section: int = 3
    # If non-empty, one Qdrant query is issued per entry; results are union-deduplicated.
    # Overrides chunk_keywords when set.
    chunk_keywords_list: List[str] = field(default_factory=list)


# ============================================================================
# ENTITY PARSERS
# ============================================================================

def parse_person_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    person = str(entity.get('person', '')).strip()
    role = str(entity.get('role', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company
    is_current = entity.get('is_current', False)

    if not person or len(person) <= 2 or not is_current:
        return None

    name_parts = person.split()
    if len(name_parts) < 2:
        return None

    role_lower = role.lower()
    if 'ceo' in role_lower or 'chief executive' in role_lower:
        rel_type = 'CEO_OF'
    elif 'cfo' in role_lower or 'chief financial' in role_lower:
        rel_type = 'CFO_OF'
    elif 'board' in role_lower or 'director' in role_lower:
        rel_type = 'BOARD_MEMBER_OF'
    else:
        rel_type = 'WORKS_AT'

    return {
        'source': {'type': 'Person', 'name': person},
        'target': {'type': 'Organization', 'name': org},
        'relationship': rel_type,
        'properties': {'role': role}
    }


def parse_metric_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    metric = str(entity.get('metric', '')).strip()
    value = str(entity.get('value', '')).strip()
    unit = str(entity.get('unit', 'ratio')).strip()
    year = str(entity.get('year', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not metric or not value:
        return None

    import re
    # Strip commas, currency symbols, and trailing 'x'/'times'
    value_clean = value.replace(',', '').strip()
    value_clean = re.sub(r'[₹#$€£¥\s]', '', value_clean)
    value_clean = re.sub(r'[xX]$', '', value_clean).strip()

    try:
        float(value_clean)
    except ValueError:
        return None

    metric_name = f"{metric} ({year})" if year else metric

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Metric',
            'name': metric_name,
            'properties': {
                'value': value_clean,
                'unit': unit,
                'year': year,
                'metric_type': metric
            }
        },
        'relationship': 'HAS_METRIC',
        'properties': {}
    }


def parse_risk_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    risk_type = str(entity.get('risk_type', '')).strip()
    description = str(entity.get('description', '')).strip()
    severity = str(entity.get('severity', 'Unknown')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not risk_type:
        return None

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Risk',
            'name': risk_type,
            'properties': {
                'description': description,
                'severity': severity
            }
        },
        'relationship': 'FACES_RISK',
        'properties': {}
    }


def parse_industry_entity(entity: Dict, main_company: str = 'the Company') -> Dict:
    industry = str(entity.get('industry', '')).strip()
    sector = str(entity.get('sector', '')).strip()
    # Allow entity itself to carry the company name as a fallback
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not industry:
        return None

    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Industry',
            'name': industry,
            'properties': {'sector': sector}
        },
        'relationship': 'OPERATES_IN',
        'properties': {}
    }


# ============================================================================
# RELATION CONFIGURATIONS
# ============================================================================

RELATION_CONFIGS: Dict[str, RelationConfig] = {
    'CEO': RelationConfig(
        name='CEO',
        source_entity_type='Person',
        target_entity_type='Organization',
        relationship_type='CEO_OF',
        section_keywords='corporate governance overview introduction',
        chunk_keywords='president and ceo chief executive officer ceo president chief executive',
        extraction_prompt_template="""Extract ONLY the CURRENT CEO from this text. Ignore board members, CFOs, former executives.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"person": "Full Name", "role": "CEO", "organization": "{main_company}", "is_current": true}}
]

STRICT Rules:
- Extract ONLY the person explicitly called "President and CEO", "Chief Executive Officer", or "CEO" who currently holds the position.
- Ignore anyone with titles like "Chairman", "Director", "CFO", "Executive Vice President", or past tense.
- organization: ALWAYS use "{main_company}".
- Extract exactly ONE person.
- Return empty array [] if no current CEO is clearly identified.
""",
        entity_parser=parse_person_entity,
        entity_parser_kwargs={}
    ),

    'HAS_METRIC': RelationConfig(
        name='HAS_METRIC',
        source_entity_type='Company',
        target_entity_type='Metric',
        relationship_type='HAS_METRIC',
        section_keywords='financial performance results earnings statements',
        chunk_keywords='',  # Overridden by chunk_keywords_list below
        chunk_keywords_list=[
            'gearing ratio net debt total equity leverage debt ratio',
            'free cash flow operating cash flow capital expenditure investment',
            'EBITDA earnings before interest tax depreciation amortization operating profit',
            'ROACE return on average capital employed return on equity',
            'net income profit loss attributable earnings per share',
            'revenue total sales income turnover',
            'interest coverage ratio EBIT debt service fixed charge',
            'current ratio quick ratio liquidity cash equivalents short-term',
            'total debt borrowings long-term debt bonds notes payable',
        ],
        n_sections=3,
        n_chunks_per_section=3,
        extraction_prompt_template="""Extract ALL financial metrics that appear with explicit numeric values in the text.

Use these standard metric names where applicable (but also extract any other clearly stated financial metric):
  - Gearing Ratio            (net debt / total equity, as %)
  - Net Debt                 (total borrowings minus cash)
  - Free Cash Flow           (operating cash flow minus capex)
  - EBITDA                   (earnings before interest, tax, depreciation & amortization)
  - ROACE                    (return on average capital employed)
  - Capital Expenditure      (capex / investment spending)
  - Net Income               (profit attributable to shareholders)
  - Revenue                  (total sales / turnover)
  - Interest Coverage Ratio  (EBIT / interest expense)
  - Total Debt               (total borrowings / financial liabilities)
  - Cash and Equivalents     (cash on hand / short-term liquidity)
  - Debt-to-Equity Ratio     (total debt / equity)
  - Current Ratio            (current assets / current liabilities)

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"metric": "Gearing Ratio", "value": "12.3", "unit": "%", "year": "2024", "organization": "{main_company}"}}
]

STRICT Rules:
- Only extract a metric if its numeric value is EXPLICITLY stated in the text — no calculation, no inference.
- metric: Use the standard name from the list above if it matches; otherwise use the exact name as written in the text.
- value: The raw number only (e.g. "12.3", "454.3"). Strip commas and currency symbols.
- unit: use "%" for percentages/ratios, "USD billion", "SAR billion", or "USD million" based on context.
- year: extract from context (e.g. "year ended December 31, 2024" → "2024"). Leave "" if not found.
- organization: ALWAYS use "{main_company}".
- Return [] if no metrics with a clear numeric value appear in the text.
- DO NOT use external knowledge — only what is written in the text.
""",
        entity_parser=parse_metric_entity,
        entity_parser_kwargs={}
    ),

    'FACES_RISK': RelationConfig(
        name='FACES_RISK',
        source_entity_type='Company',
        target_entity_type='Risk',
        relationship_type='FACES_RISK',
        section_keywords='risk management exposure factors financial operational',
        chunk_keywords='could materially adversely affect business financial condition operations results',
        n_sections=3,
        n_chunks_per_section=5,
        extraction_prompt_template="""Extract ALL risks explicitly described in this text. Classify each into one of the categories below.

Risk categories:
  - Credit_Risk       : risk of counterparty or borrower default, impairment, receivables deterioration
  - Liquidity_Risk    : insufficient cash, funding gaps, inability to meet short-term obligations
  - Market_Risk       : interest rate, foreign exchange, or commodity price movements
  - Operational_Risk  : system failures, process breakdowns, fraud, human error, cyber threats
  - Regulatory_Risk   : regulatory changes, sanctions, legal/compliance requirements
  - Geopolitical_Risk : political instability, armed conflict, war, sanctions, country risk
  - Strategic_Risk    : competition, business model disruption, M&A integration, reputational damage
  - Environmental_Risk: climate change, natural disasters, environmental liability

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"risk_type": "Geopolitical_Risk", "description": "concise description from the text, 80-160 chars", "severity": "High", "organization": "{main_company}"}}
]

STRICT Rules:
- risk_type: Must be exactly one of the eight categories above.
- description: Paraphrase the text's own wording — 80-160 characters. No generic filler.
- severity:
    "High"   → text uses "material adverse", "significant", "could materially affect", "severely"
    "Medium" → text uses "could affect", "may impact", "potential impact", "may result in"
    "Low"    → text uses "possible", "minor", "limited", "unlikely"
    "Unknown"→ severity not stated
- organization: ALWAYS use "{main_company}".
- Extract each distinct risk as a separate object. Merge near-duplicates into one.
- Return [] if no risk is explicitly described in the text.
- DO NOT use external knowledge — only what is written.
""",
        entity_parser=parse_risk_entity,
        entity_parser_kwargs={}
    ),

    'OPERATES_IN': RelationConfig(
        name='OPERATES_IN',
        source_entity_type='Company',
        target_entity_type='Industry',
        relationship_type='OPERATES_IN',
        section_keywords='overview strategy introduction',
        chunk_keywords='oil gas energy refining petrochemicals chemicals upstream downstream renewable energy marketing distribution production exploration',
        extraction_prompt_template="""Extract the PRIMARY industry of {main_company} ONLY.

Text: {text}

Return ONLY a valid JSON array:
[
  {{"industry": "Oil & Gas", "sector": "Energy"}}
]

Rules:
- ONLY extract the industry of {main_company} — ignore all other companies, subsidiaries, or partners
- Return exactly ONE entry
- If unclear, return []
""",
        entity_parser=parse_industry_entity,
        entity_parser_kwargs={}
    )
}


def get_relation_config(relation_name: str) -> RelationConfig:
    """Get configuration for a specific relation type"""
    return RELATION_CONFIGS.get(relation_name.upper())


def set_main_company(company_name: str):
    """Inject the main company name into all relation configs at runtime."""
    for cfg in RELATION_CONFIGS.values():
        cfg.entity_parser_kwargs['main_company'] = company_name


def list_available_relations() -> List[str]:
    """List all available relation types"""
    return list(RELATION_CONFIGS.keys())
