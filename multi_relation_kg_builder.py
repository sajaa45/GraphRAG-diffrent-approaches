#!/usr/bin/env python3
"""
Generalized Multi-Relation Knowledge Graph Builder
Supports multiple relation types using embedding-based hierarchical retrieval
"""

import os
import json
import argparse
import requests
import time
import re
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from relation_extraction_config import (
    RelationConfig, 
    get_relation_config, 
    list_available_relations,
    set_main_company
)
from industry_to_sic import get_sic_code

# ============================================================================
# CONFIGURATION
# ============================================================================
TOP_N_SECTIONS = 2
TOP_N_CHUNKS_PER_SECTION = 3
SECTION_SIMILARITY_THRESHOLD = 0.25  # Lowered from 0.35 for better recall
CHUNK_SIMILARITY_THRESHOLD = 0.3  # Lowered from 0.4 — metrics/risks often score lower
OLLAMA_URL = "http://host.docker.internal:11434"
OLLAMA_MODEL = "mistral:latest"
OLLAMA_TIMEOUT = None  # No timeout - wait indefinitely
OLLAMA_MAX_RETRIES = 1  # No retries needed without timeout
# ============================================================================


class MultiRelationKGBuilder:
    """Build Neo4j knowledge graph with multiple relation types"""
    
    def __init__(self, 
                 neo4j_uri: str, 
                 neo4j_user: str, 
                 neo4j_password: str,
                 collection_name: str = "financial_docs",
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 ollama_url: str = OLLAMA_URL,
                 ollama_model: str = OLLAMA_MODEL,
                 output_dir: str = "/app/output",
                 main_company: str = "the Company"):
        
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"✓ Connected to Neo4j at {neo4j_uri}")
        
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        print(f"✓ Using Ollama at {ollama_url} with model {ollama_model}")
        
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✓ Loaded embedding model: {embedding_model}")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(name=collection_name)
        print(f"✓ Connected to ChromaDB collection: {collection_name}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.main_company = main_company
        if main_company == "the Company":
            self.main_company = self.detect_main_company()
        set_main_company(self.main_company)
        print(f"✓ Main company: {self.main_company}")

        self._sic_cache: Dict[str, str] = {}  # sector -> SIC code cache
        
        timestamp = int(time.time())
        self.log_file = os.path.join(self.output_dir, f"extraction_log_{timestamp}.txt")
        self.log_buffer = []
        print(f"✓ Logging to: {self.log_file}")
    
    def close(self):
        self.driver.close()
        self._save_log()
    
    def _log(self, message: str):
        self.log_buffer.append(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def _save_log(self):
        if self.log_buffer:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log_buffer))
            print(f"\n✓ Extraction log saved to: {self.log_file}")
    
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✓ Cleared existing graph")

    def _lookup_sic(self, sector: str) -> str:
        """Return SIC code for a sector string, cached to avoid repeat API calls."""
        if not sector:
            return None
        key = sector.strip().lower()
        if key not in self._sic_cache:
            try:
                code = get_sic_code(sector)
                self._sic_cache[key] = str(code).strip()
                print(f"    ✓ SIC lookup: '{sector}' → {self._sic_cache[key]}")
            except Exception as e:
                print(f"    ⚠ SIC lookup failed for '{sector}': {e}")
                self._sic_cache[key] = None
        return self._sic_cache[key]

    def detect_main_company(self) -> str:
        """
        Use Gemini to identify the main company from document chunks.
        Falls back to regex if Gemini is unavailable.
        """
        print("  Auto-detecting main company from vector store...")

        try:
            results = self.collection.query(
                query_embeddings=[self.embedding_model.encode(["annual report company overview"])[0].tolist()],
                n_results=4,
                where={"type": "chunk"}
            )
            docs = results['documents'][0]
        except Exception:
            self.company_aliases = set()
            return "the Company"

        context = "\n\n---\n\n".join(docs)

        if os.getenv("GOOGLE_API_KEY"):
            try:
                prompt = (
                    "Read the following excerpts from an annual report and return ONLY "
                    "the full legal name of the main company this report is about. "
                    "No explanation, just the name.\n\n" + context
                )
                name = self._call_llm(prompt).strip().strip('"').strip("'")
                if name and len(name) > 3:
                    print(f"  ✓ Detected main company: {name}")
                    self.company_aliases = {name}
                    return name
            except Exception as e:
                print(f"  ⚠ Gemini detection failed ({e}), falling back to regex")

        # Regex fallback
        candidates: Dict[str, int] = {}
        ORG_PATTERNS = [
            r'\b((?:[A-Z][A-Za-z0-9&\'\-\.]+\s+){1,8}(?:Corporation|Incorporated|Limited|Company|Corp\.|Inc\.|Ltd\.|plc|LLC|L\.L\.C\.|S\.A\.|N\.V\.|AG|SE|GmbH))\b',
            r'\b((?:[A-Z][A-Za-z0-9&\'\-\.]+\s+){1,6}(?:Group|Holdings|Holding|Bancorp|Financial|Energy|Capital|Resources|Technologies|Industries|International|Enterprises|Partners))\b',
        ]
        for doc in docs:
            for pattern in ORG_PATTERNS:
                for match in re.finditer(pattern, doc):
                    name = match.group(1).strip()
                    if len(name) >= 8 and name.lower() not in ('the company', 'this company'):
                        candidates[name] = candidates.get(name, 0) + 1

        if not candidates:
            print("  ⚠ Could not detect company name — using 'the Company'")
            self.company_aliases = set()
            return "the Company"

        names = list(candidates.keys())
        filtered = {n for n in names if not any(n != o and n in o for o in names)}
        candidates = {n: v for n, v in candidates.items() if n in filtered}
        canonical = max(candidates, key=lambda k: (candidates[k], len(k)))

        canonical_tokens = set(self._significant_tokens(canonical))
        self.company_aliases = {
            n for n in candidates if canonical_tokens & set(self._significant_tokens(n))
        }
        print(f"  ✓ Detected main company: {canonical}")
        return canonical

    @staticmethod
    def _significant_tokens(name: str) -> List[str]:
        """Lowercase tokens >3 chars, excluding common stop words."""
        STOP = {'the', 'and', 'for', 'of', 'in', 'a', 'an', 'company',
                'corporation', 'incorporated', 'limited', 'group', 'holdings'}
        return [t for t in re.sub(r'[^a-z0-9 ]', '', name.lower()).split()
                if len(t) > 3 and t not in STOP]

    def normalize_company_name(self, name: str) -> str:
        """Map any alias of the main company to the canonical name."""
        if not name or name.lower() in ('the company', 'this company', ''):
            return self.main_company
        if name in getattr(self, 'company_aliases', set()):
            return self.main_company
        name_tokens = set(self._significant_tokens(name))
        canonical_tokens = set(self._significant_tokens(self.main_company))
        if name_tokens and canonical_tokens:
            overlap = len(name_tokens & canonical_tokens) / min(len(name_tokens), len(canonical_tokens))
            if overlap >= 0.5:
                return self.main_company
        return name

    
    # ========================================================================
    # HIERARCHICAL RETRIEVAL
    # ========================================================================
    def hierarchical_retrieval(self, relation_config, n_sections=TOP_N_SECTIONS, 
                               n_chunks_per_section=TOP_N_CHUNKS_PER_SECTION, 
                               section_threshold=SECTION_SIMILARITY_THRESHOLD):
        print(f"\n{'='*60}")
        print(f"Hierarchical Retrieval: {relation_config.name}")
        print(f"{'='*60}")
        
        # Step 1: Sections
        section_embedding = self.embedding_model.encode([relation_config.section_keywords])[0]
        section_results = self.collection.query(
            query_embeddings=[section_embedding.tolist()],
            n_results=n_sections,
            where={"type": "section"}
        )
        
        use_sections = False
        if section_results['documents'][0]:
            best_sim = 1 - section_results['distances'][0][0]
            print(f"  Best section similarity: {best_sim:.3f}")
            if best_sim >= section_threshold:
                use_sections = True
                print(f"  ✓ Using {len(section_results['documents'][0])} relevant sections")
        
        all_chunks = []
        chunk_embedding = self.embedding_model.encode([relation_config.chunk_keywords])[0]
        
        if use_sections:
            for i, (doc, meta, dist) in enumerate(zip(
                section_results['documents'][0],
                section_results['metadatas'][0],
                section_results['distances'][0]
            ), 1):
                section_id = meta['section_id']
                section_title = meta['title']
                chunk_results = self.collection.query(
                    query_embeddings=[chunk_embedding.tolist()],
                    n_results=n_chunks_per_section,
                    where={"$and": [{"type": "chunk"}, {"section_id": section_id}]}
                )
                for doc, cmeta, cdist in zip(chunk_results['documents'][0], 
                                           chunk_results['metadatas'][0], 
                                           chunk_results['distances'][0]):
                    chunk_data = {
                        "section_title": section_title,
                        "section_id": section_id,
                        "chunk_index": cmeta.get('chunk_index'),
                        "similarity": 1 - cdist,
                        "text": doc
                    }
                    if 'source_page' in cmeta:
                        chunk_data['source_page'] = cmeta['source_page']
                    all_chunks.append(chunk_data)
        else:
            # Direct chunk fallback
            n_chunks = n_sections * n_chunks_per_section
            chunk_results = self.collection.query(
                query_embeddings=[chunk_embedding.tolist()],
                n_results=n_chunks,
                where={"type": "chunk"}
            )
            for doc, meta, dist in zip(chunk_results['documents'][0], 
                                     chunk_results['metadatas'][0], 
                                     chunk_results['distances'][0]):
                chunk_data = {
                    "section_title": meta.get('section_title', 'Unknown'),
                    "section_id": meta.get('section_id'),
                    "chunk_index": meta.get('chunk_index'),
                    "similarity": 1 - dist,
                    "text": doc
                }
                if 'source_page' in meta:
                    chunk_data['source_page'] = meta['source_page']
                all_chunks.append(chunk_data)
        
        print(f"  ✓ Retrieved {len(all_chunks)} chunks total")
        return all_chunks
    
    # ========================================================================
    # LLM EXTRACTION
    # ========================================================================
    def _call_llm(self, prompt: str) -> str:
        """Call either Gemini (if GOOGLE_API_KEY set) or Ollama. Returns raw text."""
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=google_key)
            model = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")
            for attempt in range(5):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    return response.text.strip()
                except Exception as e:
                    err = str(e)
                    is_503 = '503' in err
                    is_429 = '429' in err
                    if (is_429 or is_503) and attempt < 4:
                        wait = 60 if is_503 else 30 * (attempt + 1)
                        print(f"    ⚠ Gemini {'overloaded' if is_503 else 'rate limit'}... retrying in {wait}s ({attempt+1}/4)")
                        time.sleep(wait)
                    else:
                        raise
        else:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False, "temperature": 0.1},
                timeout=OLLAMA_TIMEOUT
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.status_code}")
            return response.json().get('response', '').strip()

    def _extract_entities_batch(self, chunks: List[Dict], relation_config) -> List[tuple]:
        """
        Single Gemini call for all chunks combined.
        Returns list of (entity, chunk) tuples so validation still has the source text.
        """
        # Build one prompt with all chunks separated
        chunks_text = "\n\n---CHUNK SEPARATOR---\n\n".join(
            f"[Chunk {i+1}]\n{c['text']}" for i, c in enumerate(chunks)
        )
        prompt = relation_config.extraction_prompt_template.format(
            text=chunks_text,
            main_company=self.main_company
        )
        try:
            llm_output = self._call_llm(prompt)
            json_start = llm_output.find('[')
            json_end = llm_output.rfind(']') + 1
            if json_start == -1:
                return []
            entities_data = json.loads(llm_output[json_start:json_end])
            results = []
            for e in entities_data:
                # Find the best matching chunk for validation (most text overlap)
                best_chunk = max(chunks, key=lambda c: sum(
                    1 for w in str(e).lower().split() if w in c['text'].lower()
                ))
                if not self._validate_entity_in_text(e, best_chunk['text'], relation_config.name):
                    continue
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    results.append((p, best_chunk))
            return results
        except Exception as e:
            print(f"    ✗ Batch extraction error: {e}")
            return []

    def extract_entities_with_llm(self, text: str, relation_config):
        """Single-chunk extraction (used for Ollama path)."""
        prompt = relation_config.extraction_prompt_template.format(
            text=text,
            main_company=self.main_company
        )
        try:
            llm_output = self._call_llm(prompt)
            json_start = llm_output.find('[')
            json_end = llm_output.rfind(']') + 1
            if json_start == -1:
                return []
            entities_data = json.loads(llm_output[json_start:json_end])
            parsed = []
            for e in entities_data:
                if not self._validate_entity_in_text(e, text, relation_config.name):
                    continue
                p = relation_config.entity_parser(e, **relation_config.entity_parser_kwargs)
                if p:
                    parsed.append(p)
            return parsed
        except Exception as e:
            print(f"    ✗ Extraction error: {e}")
            return []
    
    def _validate_entity_in_text(self, entity: Dict, text: str, relation_type: str) -> bool:
        """Validate that extracted entity is actually grounded in the source text"""
        text_lower = text.lower()
        
        if relation_type == 'CEO':
            person = str(entity.get('person', '')).strip()
            if not person or len(person) <= 2:
                return False
            # Check if person name appears in text
            name_parts = person.split()
            if len(name_parts) < 2:
                return False
            # Must have at least last name in text
            if name_parts[-1].lower() not in text_lower:
                return False
            # Verify CEO context
            role = str(entity.get('role', '')).lower()
            if 'ceo' in role or 'chief executive' in role:
                # Must have CEO-related keywords near the name
                if not any(kw in text_lower for kw in ['ceo', 'chief executive', 'president']):
                    return False
            return True
        
        elif relation_type == 'HAS_METRIC':
            metric = str(entity.get('metric', '')).strip()
            value = str(entity.get('value', '')).strip()
            if not metric or not value:
                return False
            
            # Clean value for matching (remove commas, decimals)
            value_clean = value.replace(',', '').replace('.', '')
            
            # Check if the numeric value appears in text
            # Handle both regular numbers and currency symbols (₹, #, $)
            import re
            
            # More flexible patterns - look for the core digits
            # Extract just the significant digits (first 4-5 digits)
            core_digits = re.sub(r'[^\d]', '', value)[:5]  # First 5 digits
            
            if len(core_digits) >= 3:
                # Look for these digits anywhere in the text
                if core_digits in re.sub(r'[^\d]', '', text):
                    return True
            
            # Fallback: look for the full value with various formats
            patterns = [
                rf'\b{re.escape(value)}\b',  # Exact match
                rf'[₹#$€£¥]\s*{re.escape(value)}',  # With currency prefix
                rf'{re.escape(value)}\s*[₹#$€£¥]',  # With currency suffix
                rf'\({re.escape(value)}\)',  # In parentheses
            ]
            
            found_number = any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
            if not found_number:
                print(f"      ⚠ Metric validation failed: {metric}={value} not found in text")
                return False
            
            return True
        
        elif relation_type == 'FACES_RISK':
            risk_type = str(entity.get('risk_type', '')).strip()
            if not risk_type:
                return False
            
            # Risk type must be explicitly mentioned or strongly implied
            risk_keywords = risk_type.lower().split()
            # At least 2 keywords from risk name must appear in text (or 1 if risk name is short)
            significant_keywords = [k for k in risk_keywords if len(k) > 3]
            if not significant_keywords:
                return True  # If no significant keywords, accept it
            
            matches = sum(1 for kw in significant_keywords if kw in text_lower)
            required_matches = min(2, len(significant_keywords))
            
            if matches < required_matches:
                print(f"      ⚠ Risk validation failed: {risk_type} not grounded in text")
                return False
            
            return True
        
        elif relation_type == 'OPERATES_IN':
            industry = str(entity.get('industry', '')).strip()
            if not industry:
                return False
            # Industry keywords must appear
            industry_keywords = industry.lower().split()
            if not any(kw in text_lower for kw in industry_keywords if len(kw) > 2):
                return False
            return True
        
        return True
    
    # ========================================================================
    # NEO4J OPERATIONS (unchanged)
    # ========================================================================
    def create_node(self, session, node_type: str, name: str, properties: Dict = None):
        props = properties or {}
        props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
        query = f"""
        MERGE (n:{node_type} {{name: $name}})
        ON CREATE SET n.created_at = datetime() {', ' + props_str if props_str else ''}
        ON MATCH SET {props_str if props_str else 'n.updated_at = datetime()'}
        RETURN n
        """
        session.run(query, {"name": name, **props})
    
    def create_relationship(self, session, source_type, source_name, target_type, 
                          target_name, rel_type, properties=None, source_chunk=None, similarity=None):
        props = properties or {}
        if source_chunk:
            props['source_chunk'] = source_chunk[:200]
        if similarity is not None:
            props['confidence'] = similarity
        
        query = f"""
        MATCH (s:{source_type} {{name: $source_name}})
        MATCH (t:{target_type} {{name: $target_name}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r.created_at = datetime()
        ON MATCH SET r.updated_at = datetime()
        RETURN r
        """
        session.run(query, {"source_name": source_name, "target_name": target_name, **props})
    
    # ========================================================================
    # MAIN EXTRACTION
    # ========================================================================
    def extract_relation(self, relation_name: str):
        relation_config = get_relation_config(relation_name)
        if not relation_config:
            print(f"✗ Unknown relation: {relation_name}")
            return {"error": "Unknown relation"}
        
        self._log(f"\n{'='*80}")
        self._log(f"EXTRACTING RELATION: {relation_config.name}")
        self._log(f"{'='*80}")
        self._log(f"Source: {relation_config.source_entity_type}")
        self._log(f"Target: {relation_config.target_entity_type}")
        self._log(f"Relationship: {relation_config.relationship_type}")
        self._log("")
        
        print(f"\n{'='*80}\nEXTRACTING RELATION: {relation_config.name}\n{'='*80}")
        print(f"Source: {relation_config.source_entity_type} → {relation_config.relationship_type} → {relation_config.target_entity_type}")
        
        chunks = self.hierarchical_retrieval(relation_config)
        
        filtered_chunks = [c for c in chunks if c['similarity'] >= CHUNK_SIMILARITY_THRESHOLD]
        print(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {CHUNK_SIMILARITY_THRESHOLD})")
        self._log(f"\n✓ Using {len(filtered_chunks)}/{len(chunks)} chunks (threshold: {CHUNK_SIMILARITY_THRESHOLD})")
        self._log("")
        self._log("="*80)
        self._log("EXTRACTING ENTITIES WITH LLM")
        self._log("="*80)
        
        total_entities = 0
        total_relationships = 0
        created = set()
        ceo_candidates = []

        use_gemini = bool(os.getenv("GOOGLE_API_KEY"))

        with self.driver.session() as session:
            # Gemini: one API call for all chunks
            if use_gemini:
                print(f"\n  Gemini batch: sending {len(filtered_chunks)} chunks in one call...")
                batch_results = self._extract_entities_batch(filtered_chunks, relation_config)
                # Reconstruct per-chunk iteration for the rest of the existing logic
                chunk_entity_pairs = [(entity, chunk) for entity, chunk in batch_results]
            else:
                chunk_entity_pairs = []

            for i, chunk in enumerate(filtered_chunks, 1):
                self._log(f"\nChunk {i}/{len(filtered_chunks)} (similarity: {chunk['similarity']:.3f})")
                self._log(f"  Section: {chunk['section_title']}")
                self._log(f"  Section ID: {chunk['section_id']}")
                self._log(f"  Chunk Index: {chunk.get('chunk_index', 'N/A')}")
                if 'source_page' in chunk:
                    self._log(f"  Page: {chunk['source_page']}")
                self._log(f"  Text preview: {chunk['text'][:200]}...")
                self._log(f"  Full text:")
                self._log("-" * 80)
                self._log(chunk['text'])
                self._log("-" * 80)

                print(f"\nChunk {i}/{len(filtered_chunks)} | Similarity: {chunk['similarity']:.3f} | Section: {chunk['section_title']}")

                if use_gemini:
                    # Pull entities that were matched to this chunk
                    entities = [e for e, c in chunk_entity_pairs if c is chunk]
                else:
                    entities = self.extract_entities_with_llm(chunk['text'], relation_config)

                if not entities:
                    self._log("    ✗ No entities extracted")
                
                for entity in entities:
                    src = entity['source']
                    tgt = entity['target']
                    rel = entity['relationship']
                    key = (src['name'], rel, tgt['name'])
                    
                    if rel == 'CEO_OF':
                        ceo_candidates.append({
                            'entity': entity,
                            'similarity': chunk['similarity'],
                            'text': chunk['text']
                        })
                        self._log(f"    ~ CEO candidate: {src['name']} (confidence: {chunk['similarity']:.3f})")
                        print(f"    ~ CEO candidate: {src['name']} (sim: {chunk['similarity']:.3f})")
                        continue
                    
                    if key in created:
                        self._log(f"    ⊘ Skipping duplicate: ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")
                        print(f"    ⊘ Skipping duplicate")
                        continue
                    
                    log_msg = f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})"
                    if tgt.get('properties'):
                        log_msg += f"\n      Properties: {json.dumps(tgt['properties'], indent=8)}"
                    self._log(log_msg)
                    print(f"    - ({src['type']}: {src['name']}) --[{rel}]--> ({tgt['type']}: {tgt['name']})")
                    
                    self.create_node(session, src['type'], src['name'], src.get('properties', {}))
                    self.create_node(session, tgt['type'], tgt['name'], tgt.get('properties', {}))
                    self.create_relationship(session, src['type'], src['name'], tgt['type'], tgt['name'], 
                                           rel, entity.get('properties', {}), chunk['text'], chunk['similarity'])

                    # For OPERATES_IN: look up SIC from the industry name extracted by LLM
                    if rel == 'OPERATES_IN':
                        industry_name = tgt['name']  # e.g. "Oil & Gas"
                        sector = tgt.get('properties', {}).get('sector', '')
                        sic_code = self._lookup_sic(industry_name)
                        if sic_code:
                            self.create_node(session, 'SICCode', sic_code,
                                             {'code': sic_code, 'industry': industry_name, 'sector': sector})
                            self.create_relationship(session, tgt['type'], tgt['name'],
                                                     'SICCode', sic_code, 'HAS_SIC_CODE')
                            self._log(f"      SIC: ({tgt['type']}: {industry_name}) --[HAS_SIC_CODE]--> (SICCode: {sic_code})")
                            print(f"      SIC: {industry_name} --[HAS_SIC_CODE]--> {sic_code}")
                    
                    created.add(key)
                    total_entities += 1
                    total_relationships += 1
            
            # CEO special handling - semantic validation, not just similarity
            if ceo_candidates:
                self._log(f"\n  Processing {len(ceo_candidates)} CEO candidates...")
                print(f"\n  Processing {len(ceo_candidates)} CEO candidates...")
                
                # Score candidates based on semantic signals, not just embedding similarity
                def score_ceo_candidate(candidate):
                    text = candidate['text'].lower()
                    name = candidate['entity']['source']['name'].lower()
                    
                    score = 0
                    
                    # Strong positive signals
                    if 'president and ceo' in text or 'president & ceo' in text:
                        score += 100
                    if 'chief executive officer' in text:
                        score += 80
                    if f"{name.split()[0].lower()}" in text and 'ceo' in text:
                        score += 50
                    
                    # Context matters - look for current role indicators
                    if any(phrase in text for phrase in ['was appointed', 'serves as', 'is the']):
                        score += 30
                    
                    # Negative signals - wrong context
                    if any(phrase in text for phrase in ['board member', 'director', 'chairman', 'cfo', 'formerly', 'previous']):
                        score -= 50
                    if 'executive vice president' in text and 'ceo' not in text:
                        score -= 40
                    
                    # Penalize if name appears but without CEO context nearby
                    name_parts = name.split()
                    if len(name_parts) >= 2:
                        last_name = name_parts[-1]
                        # Find name position and check for CEO keywords within 100 chars
                        if last_name in text:
                            idx = text.find(last_name)
                            context = text[max(0, idx-100):min(len(text), idx+100)]
                            if 'ceo' not in context and 'chief executive' not in context:
                                score -= 30
                    
                    return score
                
                # Score and sort
                for candidate in ceo_candidates:
                    candidate['semantic_score'] = score_ceo_candidate(candidate)
                    log_msg = f"    ~ {candidate['entity']['source']['name']}: semantic_score={candidate['semantic_score']}, similarity={candidate['similarity']:.3f}"
                    self._log(log_msg)
                    print(log_msg)
                
                ceo_candidates.sort(key=lambda x: x['semantic_score'], reverse=True)
                
                best = ceo_candidates[0]
                if best['semantic_score'] > 0:
                    entity = best['entity']
                    src = entity['source']
                    tgt = entity['target']
                    rel = entity['relationship']
                    
                    log_msg = f"  ✓ Selected CEO: {src['name']} (semantic_score: {best['semantic_score']}, similarity: {best['similarity']:.3f})"
                    self._log(log_msg)
                    print(log_msg)
                    
                    self.create_node(session, src['type'], src['name'])
                    self.create_node(session, tgt['type'], tgt['name'])
                    self.create_relationship(session, src['type'], src['name'], tgt['type'], tgt['name'], 
                                           rel, {}, best['text'], best['similarity'])
                    
                    total_entities += 1
                    total_relationships += 1
                else:
                    log_msg = f"  ✗ No valid CEO candidate found (best score: {best['semantic_score']})"
                    self._log(log_msg)
                    print(log_msg)
        
        summary = f"\n{'='*80}\nEXTRACTION COMPLETE: {relation_name}\n{'='*80}\nEntities: {total_entities}\nRelationships: {total_relationships}\n"
        self._log(summary)
        print(summary)
        return {"relation": relation_name, "entities": total_entities, "relationships": total_relationships}
    def extract_multiple_relations(self, relation_names: List[str]):
        print(f"\n{'='*60}\nMULTI-RELATION EXTRACTION\n{'='*60}")
        results = {}
        for rel in relation_names:
            results[rel] = self.extract_relation(rel)
        return results

    def show_graph_stats(self):
        """Display graph statistics"""
        with self.driver.session() as session:
            print(f"\n{'='*60}")
            print("KNOWLEDGE GRAPH STATISTICS")
            print(f"{'='*60}")
            
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
            print("\nNodes:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            print("\nRelationships:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Relation Knowledge Graph Builder")
    parser.add_argument("relations", nargs="*", help="Relations to extract")
    parser.add_argument("--all", action="store_true", help="Extract all relations")
    parser.add_argument("--list", action="store_true", help="List available relations")
    parser.add_argument("--clear", action="store_true", help="Clear database first")
    parser.add_argument("--collection", default="financial_docs", help="ChromaDB collection name")
    parser.add_argument("--db-path", default="./chroma_db", help="ChromaDB persist directory")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--ollama-url", default=None, help="Ollama API URL")
    parser.add_argument("--ollama-model", default=None, help="Ollama model name")
    parser.add_argument("--main-company", default=None, help="Main company name for OPERATES_IN extraction")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available relations:", list_available_relations())
        return
    
    relations = list_available_relations() if args.all else [r.upper() for r in args.relations]
    
    builder = MultiRelationKGBuilder(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "Lexical12345"),
        collection_name=args.collection,
        persist_directory=args.db_path,
        embedding_model=args.embedding_model,
        ollama_url=args.ollama_url or os.getenv("OLLAMA_URL", OLLAMA_URL),
        ollama_model=args.ollama_model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL),
        main_company=args.main_company or os.getenv("MAIN_COMPANY", "the Company")
    )
    
    try:
        if args.clear:
            builder.clear_database()
        builder.extract_multiple_relations(relations)
        builder.show_graph_stats()
    finally:
        builder.close()


if __name__ == "__main__":
    main()