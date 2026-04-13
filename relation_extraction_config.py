#!/usr/bin/env python3
"""
Configuration for multi-relation extraction pipeline
Defines relation types with their section/chunk keywords and extraction logic
"""

from typing import Dict, List, Any, Callable
from dataclasses import dataclass


@dataclass
class RelationConfig:
    """Configuration for a single relation type"""
    name: str
    source_entity_type: str
    target_entity_type: str
    relationship_type: str
    section_keywords: str  # Keywords for section-level filtering
    chunk_keywords: str    # Keywords for chunk-level semantic search
    extraction_prompt_template: str  # LLM prompt template
    entity_parser: Callable[[Dict], Dict]  # Function to parse LLM output to entity dict


# ============================================================================
# ENTITY PARSERS - Convert LLM output to standardized entity format
# ============================================================================

def parse_person_entity(entity: Dict) -> Dict:
    """Parse person/executive entity from LLM output"""
    person = entity.get('person', '').strip()
    role = entity.get('role', '').strip()
    org = entity.get('organization', 'Saudi Aramco').strip()
    is_current = entity.get('is_current', False)
    
    if not person or len(person) <= 2 or not is_current:
        return None
    
    # Additional validation: person should have at least first and last name
    name_parts = person.split()
    if len(name_parts) < 2:
        return None
    
    # Map role to relationship type
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


def parse_competitor_entity(entity: Dict) -> Dict:
    """Parse competitor entity from LLM output"""
    company = entity.get('company', '').strip()
    competitor = entity.get('competitor', '').strip()
    context = entity.get('context', '').strip()
    
    if not company or not competitor:
        return None
    
    return {
        'source': {'type': 'Company', 'name': company},
        'target': {'type': 'Company', 'name': competitor},
        'relationship': 'COMPETES_WITH',
        'properties': {'context': context}
    }


def parse_metric_entity(entity: Dict) -> Dict:
    """Parse financial metric entity from LLM output"""
    metric = entity.get('metric', '').strip() if isinstance(entity.get('metric'), str) else str(entity.get('metric', ''))
    value = str(entity.get('value', '')).strip()
    unit = entity.get('unit', '').strip() if isinstance(entity.get('unit'), str) else str(entity.get('unit', ''))
    currency = entity.get('currency', 'USD').strip() if isinstance(entity.get('currency'), str) else 'USD'
    year = str(entity.get('year', '')).strip()
    org = entity.get('organization', 'Saudi Aramco').strip() if isinstance(entity.get('organization'), str) else 'Saudi Aramco'
    
    if not metric or not value:
        return None
    
    # Create metric node name with value for uniqueness
    metric_name = f"{metric} ({year})" if year else metric
    
    return {
        'source': {'type': 'Company', 'name': org},
        'target': {
            'type': 'Metric',
            'name': metric_name,
            'properties': {
                'value': value,
                'unit': unit,
                'currency': currency,
                'year': year,
                'metric_type': metric
            }
        },
        'relationship': 'HAS_METRIC',
        'properties': {}
    }


def parse_risk_entity(entity: Dict) -> Dict:
    """Parse risk factor entity from LLM output"""
    risk_type = entity.get('risk_type', '').strip()
    description = entity.get('description', '').strip()
    severity = entity.get('severity', 'Unknown').strip()
    org = entity.get('organization', 'Saudi Aramco').strip()
    
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


def parse_industry_entity(entity: Dict) -> Dict:
    """Parse industry/sector entity from LLM output"""
    company = entity.get('company', '').strip()
    industry = entity.get('industry', '').strip()
    sector = entity.get('sector', '').strip()
    
    if not company or not industry:
        return None
    
    return {
        'source': {'type': 'Company', 'name': company},
        'target': {
            'type': 'Industry',
            'name': industry,
            'properties': {'sector': sector}
        },
        'relationship': 'OPERATES_IN',
        'properties': {}
    }


def parse_product_entity(entity: Dict) -> Dict:
    """Parse product/service entity from LLM output"""
    company = entity.get('company', '').strip()
    product = entity.get('product', '').strip()
    category = entity.get('category', '').strip()
    description = entity.get('description', '').strip()
    
    if not company or not product:
        return None
    
    return {
        'source': {'type': 'Company', 'name': company},
        'target': {
            'type': 'Product',
            'name': product,
            'properties': {
                'category': category,
                'description': description
            }
        },
        'relationship': 'OFFERS',
        'properties': {}
    }


# ============================================================================
# RELATION CONFIGURATIONS
# ============================================================================

RELATION_CONFIGS = {
    'CEO': RelationConfig(
        name='CEO',
        source_entity_type='Person',
        target_entity_type='Organization',
        relationship_type='CEO_OF',
        section_keywords='overview board governance directors leadership',
        chunk_keywords='CEO chief executive officer',
        extraction_prompt_template="""Extract CURRENT CEO information from this text. Ignore past/former positions and board members.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"person": "Full Name", "role": "CEO", "organization": "Company Name", "is_current": true}}
]

CRITICAL Rules:
- Extract ONLY the person who is CURRENTLY the CEO or Chief Executive Officer
- DO NOT extract board members, directors, or other executives unless they are explicitly the CEO
- Ignore "Previously", "has served", "from X to Y", "was", past tense verbs
- Ignore people with titles like "Board Member", "Director", "Chairman" unless they are also CEO
- Look for explicit phrases: "President and CEO", "Chief Executive Officer", "CEO"
- Set is_current to true ONLY if they CURRENTLY hold the CEO position
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- Return empty array [] if no CURRENT CEO found
- ONLY extract ONE person - the actual CEO, not the entire board
""",
        entity_parser=parse_person_entity
    ),
    
    'COMPETES_WITH': RelationConfig(
        name='COMPETES_WITH',
        source_entity_type='Company',
        target_entity_type='Company',
        relationship_type='COMPETES_WITH',
        section_keywords='competition competitive landscape business market',
        chunk_keywords='compete competitors rivals competitive',
        extraction_prompt_template="""Extract competitor information from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"company": "Saudi Aramco", "competitor": "Competitor Name", "context": "brief context"}}
]

Rules:
- Extract companies that are mentioned as competitors or rivals
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as company
- Provide brief context about the competitive relationship (max 100 chars)
- Return empty array [] if no competitors found
""",
        entity_parser=parse_competitor_entity
    ),
    
    'HAS_METRIC': RelationConfig(
        name='HAS_METRIC',
        source_entity_type='Company',
        target_entity_type='Metric',
        relationship_type='HAS_METRIC',
        section_keywords='financial highlights results operations performance',
        chunk_keywords='revenue EBITDA net income profit growth',
        extraction_prompt_template="""Extract financial metrics WITH SPECIFIC NUMBERS from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"metric": "Revenue", "value": "123.4", "unit": "billion", "currency": "USD", "year": "2024", "organization": "Saudi Aramco"}}
]

Rules:
- Extract financial metrics like: Revenue, Net Income, Profit, Sales, EBITDA, Growth Rate, Operating Income, Cash Flow
- MUST include the ACTUAL NUMERIC VALUE (e.g., "123.4", "45.2", "8.5")
- unit should be: billion, million, thousand, or percent (for growth rates)
- currency should be: USD, SAR, EUR, etc. (default to USD if not specified)
- year should be the fiscal year (e.g., "2024", "2023")
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- ONLY extract metrics where you can find the actual number
- Return empty array [] if no financial data with numbers found
""",
        entity_parser=parse_metric_entity
    ),
    
    'FACES_RISK': RelationConfig(
        name='FACES_RISK',
        source_entity_type='Company',
        target_entity_type='Risk',
        relationship_type='FACES_RISK',
        section_keywords='risk factors uncertainties',
        chunk_keywords='risk uncertainty adversely affect',
        extraction_prompt_template="""Extract detailed risk factors from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"risk_type": "Specific Risk Name", "description": "detailed description of the risk", "severity": "High", "organization": "Saudi Aramco"}}
]

Rules:
- Extract SPECIFIC risk types with descriptive names (e.g., "Oil Price Volatility", "Geopolitical Instability", "Climate Change Regulations")
- Provide a DETAILED description (100-200 chars) explaining what the risk is and its potential impact
- severity should be: High, Medium, Low (assess based on language like "significant", "material", "could adversely affect")
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as organization
- Extract ALL distinct risks mentioned in the text
- Be specific - avoid generic terms like "Market Risk", instead use "Oil Price Volatility Risk"
- Return empty array [] if no risks found
""",
        entity_parser=parse_risk_entity
    ),
    
    'OPERATES_IN': RelationConfig(
        name='OPERATES_IN',
        source_entity_type='Company',
        target_entity_type='Industry',
        relationship_type='OPERATES_IN',
        section_keywords='business overview segments operations',
        chunk_keywords='industry sector market',
        extraction_prompt_template="""Extract industry and sector information from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"company": "Saudi Aramco", "industry": "Oil & Gas", "sector": "Energy"}}
]

Rules:
- Extract the industry and sector the company operates in
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as company
- Be specific about industry (e.g., "Oil & Gas", "Renewable Energy", "Petrochemicals")
- sector is broader (e.g., "Energy", "Technology", "Healthcare")
- Return empty array [] if no industry information found
""",
        entity_parser=parse_industry_entity
    ),
    
    'OFFERS': RelationConfig(
        name='OFFERS',
        source_entity_type='Company',
        target_entity_type='Product',
        relationship_type='OFFERS',
        section_keywords='products services solutions segments',
        chunk_keywords='offer provide product service',
        extraction_prompt_template="""Extract products and services from this text.

Text: {text}

Return ONLY a JSON array with this exact format (no other text):
[
  {{"company": "Saudi Aramco", "product": "Product/Service Name", "category": "category", "description": "brief description"}}
]

Rules:
- Extract products, services, or solutions offered by the company
- If text mentions Saudi Aramco or "the Company", use "Saudi Aramco" as company
- category can be: Product, Service, Solution, etc.
- Provide brief description (max 100 chars)
- Return empty array [] if no products/services found
""",
        entity_parser=parse_product_entity
    )
}


def get_relation_config(relation_name: str) -> RelationConfig:
    """Get configuration for a specific relation type"""
    return RELATION_CONFIGS.get(relation_name.upper())


def list_available_relations() -> List[str]:
    """List all available relation types"""
    return list(RELATION_CONFIGS.keys())
