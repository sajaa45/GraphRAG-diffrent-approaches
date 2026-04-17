# Knowledge Graph Extraction Fixes

## Problems Identified and Fixed

### 1. CEO Extraction - Semantic Validation Instead of Similarity Scores

**Problem:** The system selected Seth M. Klarman (similarity 0.515) over Amin H. Nasser (similarity 0.504) because it only used embedding similarity scores. The chunk with Klarman had higher similarity but wrong context.

**Fix:** Implemented semantic scoring system that validates CEO candidates based on:
- **Positive signals (+100 points):** "President and CEO", "President & CEO"
- **Strong signals (+80 points):** "Chief Executive Officer"
- **Context signals (+30 points):** "was appointed", "serves as", "is the"
- **Negative signals (-50 points):** "board member", "director", "chairman", "cfo", "formerly", "previous"
- **Context validation:** Checks if CEO keywords appear within 100 characters of the person's name

The system now scores candidates semantically and picks the one with the highest semantic score (not just embedding similarity).

### 2. Metric Hallucination - Strict Text Grounding

**Problem:** LLM invented metrics like "Net Income: $106.2B" and "Revenue: $103.1B" from chunks that contained no such numbers. It was using parametric knowledge instead of extracting from text.

**Fix:** 
- **Updated prompt:** Changed from "Extract EVERY financial metric" to "Extract ONLY financial metrics that have EXPLICIT numeric values in this text. DO NOT invent or infer numbers."
- **Added validation:** `_validate_entity_in_text()` now checks that the numeric value actually appears in the source text using regex patterns
- **Validation logic:** Searches for the number with various formats (with/without currency symbols, commas, etc.)
- **Rejects hallucinations:** If the number isn't found in text, the entity is rejected with a warning message

### 3. Currency Symbol Parsing - Handle ₹ and Dual-Currency Formats

**Problem:** Chunks with "₹319,998 ($85,333M)" were returning "✗ No entities extracted" because the parser couldn't handle the ₹ symbol and dual-currency format.

**Fix:**
- **Updated prompt:** Added explicit instructions: "Handle currency symbols: ₹ (SAR), # (SAR), $ (USD)" and "For dual currency like '₹319,998 ($85,333)', prefer the USD value in parentheses"
- **Improved parser:** `parse_metric_entity()` now strips currency symbols using regex: `re.sub(r'[₹#$€£¥\s]', '', value_clean)`
- **Validation:** Validates the cleaned value is a valid number before accepting
- **Flexible matching:** Validation checks for numbers with various currency symbol patterns

### 4. Risk Extraction - Ground in Actual Text

**Problem:** Extracted risks like "Oil Price Volatility", "Cybersecurity", "Regulatory" were too generic and not grounded in the actual chunk text.

**Fix:**
- **Updated prompt:** Changed from "Extract ALL distinct risk factors" to "Extract ONLY risk factors that are EXPLICITLY named or described in this text. DO NOT infer generic risks."
- **Added validation:** Checks that at least 2 keywords from the risk name appear in the source text
- **Stricter requirements:** Risk type must be specifically mentioned, not just implied
- **Rejects generic risks:** If risk keywords don't match text, entity is rejected with warning

## Key Changes to Code

### `multi_relation_kg_builder.py`

1. **Added `_validate_entity_in_text()` method** - Validates extracted entities against source text before parsing
2. **Enhanced CEO selection logic** - Semantic scoring instead of similarity-based selection
3. **Improved logging** - Better visibility into validation failures
4. **Added `import re`** - For regex-based validation

### `relation_extraction_config.py`

1. **Updated `HAS_METRIC` prompt** - Strict grounding requirements, currency symbol handling
2. **Updated `FACES_RISK` prompt** - Explicit risk naming requirements
3. **Enhanced `parse_metric_entity()`** - Currency symbol stripping and numeric validation

## Testing Recommendations

Run the extraction again with:
```bash
docker-compose up multi-relation-kg
```

Expected improvements:
- ✓ CEO: Should select "Amin H. Nasser" with high semantic score
- ✓ Metrics: Should extract "Free Cash Flow: ₹319,998 ($85,333M)" and "EBIT: ₹772,296 ($205,946M)"
- ✓ Metrics: Should NOT extract hallucinated Net Income/Revenue values
- ✓ Risks: Should only extract risks explicitly named in the text, or return empty if none found

## Generalization for Other Annual Reports

These fixes are designed to work across different annual reports:
- **No hard-coded company names** - Uses "the Company" or extracts from text
- **Flexible currency handling** - Supports $, ₹, #, €, £, ¥
- **Semantic validation** - Works for any CEO/executive extraction
- **Text grounding** - Prevents hallucination regardless of company
- **Configurable thresholds** - Can adjust validation strictness via config
