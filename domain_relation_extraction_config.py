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
    unit = str(entity.get('unit', '')).strip()
    currency = str(entity.get('currency', 'USD')).strip()
    year = str(entity.get('year', '')).strip()
    org = str(entity.get('organization', main_company)).strip() or main_company

    if not metric or not value:
        return None
    
    # Clean value: remove commas, handle currency symbols
    import re
    value_clean = value.replace(',', '').strip()
    
    # Remove any currency symbols from value
    value_clean = re.sub(r'[₹#$€£¥\s]', '', value_clean)
    
    # Validate it's a number
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
                'currency': currency,
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
        section_keywords='corporate governance board directors leadership executive management senior executives officers',
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
        section_keywords='results performance financial highlights key metrics revenue income ebit cash flow dividends capital expenditures',
        chunk_keywords='net income revenue ebit free cash flow capital expenditures dividends roace profit earnings cash flow',
        extraction_prompt_template="""Extract ONLY financial metrics that have EXPLICIT numeric values in this text. DO NOT invent or infer numbers.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"metric": "Free Cash Flow", "value": "85333", "unit": "million", "currency": "USD", "year": "2024", "organization": "{main_company}"}}
]

CRITICAL Rules:
- Extract ONLY metrics where you can see the EXACT number in the text
- Handle currency symbols: ₹ (SAR), # (SAR), $ (USD) - extract the number after the symbol
- For dual currency like "₹319,998 ($85,333)", prefer the USD value in parentheses
- Remove commas from numbers: "319,998" becomes "319998"
- Metrics: Net Income, Revenue, EBIT, Free Cash Flow, Capital Expenditures, Dividends, ROACE, Basic EPS
- unit: "billion", "million", "thousand", "percent" (infer from context like "billion" or "M" or "B")
- currency: "USD" or "SAR" based on symbol ($ = USD, ₹ or # = SAR)
- year: extract from text (e.g., "in 2024", "for 2024")
- organization: ALWAYS use "{main_company}"
- If you cannot find a clear number in the text, return empty array []
- DO NOT use your knowledge of the company - only extract what's written
""",
        entity_parser=parse_metric_entity,
        entity_parser_kwargs={}
    ),

    'FACES_RISK': RelationConfig(
        name='FACES_RISK',
        source_entity_type='Company',
        target_entity_type='Risk',
        relationship_type='FACES_RISK',
        section_keywords='risk factors risk management principal risks uncertainties threats challenges exposures',
        chunk_keywords='risk factors geopolitical commodity price climate change operational regulatory cyber cybersecurity hazard litigation market volatility',
        extraction_prompt_template="""Extract ONLY risk factors that are EXPLICITLY named or described in this text. DO NOT infer generic risks.

Text: {text}

The company in this document is: {main_company}

Return ONLY a valid JSON array (no other text):
[
  {{"risk_type": "Specific Risk Name from Text", "description": "exact description from text, paraphrased to 120-180 chars", "severity": "High", "organization": "{main_company}"}}
]

CRITICAL Rules:
- risk_type: Must be a specific risk explicitly mentioned (e.g., "Commodity Price Volatility", "Geopolitical Instability in Middle East")
- DO NOT extract generic risks like "Oil Price Volatility" unless those exact words appear
- description: Use the actual wording from the text, paraphrased to fit 120-180 characters
- severity: 
  * "High" if text says "material adverse effect", "significant impact", "could materially affect"
  * "Medium" if text says "could affect", "may impact"
  * "Low" if text says "potential", "possible"
- organization: ALWAYS use "{main_company}"
- If the text only mentions "risks" generically without naming specific ones, return empty array []
- DO NOT use your knowledge of typical industry risks - only extract what's written
""",
        entity_parser=parse_risk_entity,
        entity_parser_kwargs={}
    ),

    'OPERATES_IN': RelationConfig(
        name='OPERATES_IN',
        source_entity_type='Company',
        target_entity_type='Industry',
        relationship_type='OPERATES_IN',
        section_keywords='overview strategy business operations segments activities portfolio upstream downstream refining petrochemicals',
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