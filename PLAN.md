# Law Buddy - Bangladesh Legal RAG System
## Comprehensive Development Plan

---

## 1. Project Overview

### What You Have
- **1,484 legal acts** with 35,633 sections and 14,523 footnotes
- Rich metadata: government context, legal system context, token counts
- **5 synthetic case studies** covering Cyber Crime, Contract Law, Family Law, Labor Law, and Property Law
- Bilingual content (English/Bengali)
- Date range: 1799-2025
- Total tokens: 2,500,000+

### What You're Building
A legal AI assistant for:
1. **Normal People**: Get simplified legal suggestions based on their situations
2. **Lawyers**: Find relevant laws, analyze cases, discover precedents, reason about new findings

---

## 2. Recommended RAG Architecture: **Agentic Hybrid RAG**

For your legal domain with two user types and two data sources, I recommend an **Agentic Hybrid RAG** combining:

### 2.1 Multi-Index Strategy
```
┌─────────────────────────────────────────────────────────────┐
│                     VECTOR STORES (ChromaDB)                 │
├─────────────────────────────────────────────────────────────┤
│  Collection 1: acts_sections                                 │
│  - Chunked sections with hierarchical metadata               │
│  - Parent-child relationships preserved                      │
│                                                              │
│  Collection 2: case_studies                                  │
│  - Full case documents with structured metadata              │
│  - Linked to relevant acts                                   │
│                                                              │
│  Collection 3: act_summaries                                 │
│  - High-level act descriptions for broad queries             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Why Agentic Hybrid RAG?

| RAG Type | Use Case in Your App |
|----------|---------------------|
| **Dense Retrieval** | Semantic search for conceptual queries ("what are my rights if landlord evicts me") |
| **Sparse/BM25** | Exact legal term matching ("Section 420 IPC", "Labour Act 2006") |
| **Parent Document Retrieval** | Retrieve full context when section is found |
| **Self-Query RAG** | Filter by metadata (year, court level, area of law) |
| **Agentic RAG** | Multi-step reasoning for complex legal questions |

### 2.3 Architecture Diagram

```
                                    ┌─────────────────────┐
                                    │   User Query (EN/BN)│
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  Query Classifier    │
                                    │  (User Type + Intent)│
                                    └──────────┬──────────┘
                                               │
                      ┌────────────────────────┼────────────────────────┐
                      │                        │                        │
           ┌──────────▼──────────┐  ┌─────────▼─────────┐  ┌──────────▼──────────┐
           │  Simple Q&A Agent   │  │ Case Analysis Agent│  │  Legal Research     │
           │  (Normal People)    │  │ (Lawyers)          │  │  Agent (Lawyers)    │
           └──────────┬──────────┘  └─────────┬─────────┘  └──────────┬──────────┘
                      │                        │                        │
                      └────────────────────────┼────────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Retrieval Router   │
                                    └──────────┬──────────┘
                                               │
              ┌────────────────────────────────┼────────────────────────────────┐
              │                                │                                │
   ┌──────────▼──────────┐        ┌───────────▼───────────┐        ┌──────────▼──────────┐
   │  Hybrid Search      │        │  Self-Query Filter    │        │  Re-ranker          │
   │  (Dense + BM25)     │        │  (Metadata filtering) │        │  (Cross-encoder)    │
   └──────────┬──────────┘        └───────────┬───────────┘        └──────────┬──────────┘
              │                                │                                │
              └────────────────────────────────┼────────────────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Context Assembly   │
                                    │   + Citation Builder │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Response Generator │
                                    │   (User-type aware)  │
                                    └──────────┬──────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │   Response (EN/BN)   │
                                    │   + Citations        │
                                    └─────────────────────┘
```

---

## 3. Technology Stack

### 3.1 Core Stack
| Component | Technology | Reason |
|-----------|------------|--------|
| **Framework** | LangChain + LangGraph | Agent orchestration, chains, tool use |
| **Vector Store** | ChromaDB | Local, persistent, good for MVP |
| **Embeddings** | HuggingFace API: `intfloat/multilingual-e5-large` | Excellent Bengali + English support |
| **LLM** | Mistral AI API | Good reasoning, cost-effective |
| **BM25** | `rank_bm25` | Sparse retrieval for exact matches |
| **Reranker** | HuggingFace: `BAAI/bge-reranker-v2-m3` | Cross-lingual reranking |
| **UI (MVP)** | Streamlit | Quick prototyping |
| **Auth (Later)** | Better Auth | For SaaS features |

### 3.2 Additional Tools
| Tool | Purpose |
|------|---------|
| `langchain-chroma` | ChromaDB integration |
| `langchain-mistralai` | Mistral LLM integration |
| `langchain-huggingface` | HuggingFace embeddings |
| `huggingface-hub` | HuggingFace API access |
| `pydantic` | Data validation |
| `python-dotenv` | Environment management |
| `tiktoken` | Token counting |
| `rank_bm25` | BM25 retrieval |

### 3.3 Environment Variables
```env
MISTRAL_API_KEY=your_mistral_key
HUGGINGFACE_API_KEY=your_hf_key
```

---

## 4. Data Processing Pipeline

### 4.1 Document Chunking Strategy

```python
# Hierarchical chunking for acts
Act (Parent Document)
├── Metadata (act_title, act_no, year, government_context, legal_system_context)
├── Summary (AI-generated)
└── Sections (Child Documents)
    ├── Section 1: chunk with overlap
    ├── Section 2: chunk with overlap
    └── ...

# Case study structure
Case Study (Single Document)
├── Metadata (case_id, court_level, area_of_law, laws_cited, verdict)
└── Full content (facts, issues, arguments, reasoning)
```

### 4.2 Chunking Parameters
- **Section chunks**: 1000 tokens with 200 token overlap
- **Case studies**: Keep as single documents (they're already concise)
- **Preserve**: Section numbers, titles, cross-references

### 4.3 Metadata Schema

```python
# For Acts/Sections
{
    "source_type": "act",
    "act_title": str,
    "act_no": str,
    "act_year": int,
    "section_title": str,
    "section_number": str,
    "language": str,  # "english", "bengali", "mixed"
    "is_repealed": bool,
    "legal_framework": str,
    "government_period": str,
    "parent_act_id": str,  # For parent-child retrieval
}

# For Case Studies
{
    "source_type": "case_study",
    "case_id": str,
    "case_title": str,
    "area_of_law": str,
    "court_level": str,
    "verdict": str,
    "laws_cited": List[str],
    "precedent_value": str,
    "decision_date": str,
}
```

---

## 5. Agent System Design

### 5.1 Agent Types

#### Agent 1: Public Assistant (For Normal People)
```
Purpose: Simplify legal concepts, provide general guidance
Capabilities:
  - Explain rights in simple terms
  - Suggest relevant laws without jargon
  - Provide step-by-step guidance
  - Warn about seeking professional help
```

#### Agent 2: Legal Research Agent (For Lawyers)
```
Purpose: Deep legal research and analysis
Capabilities:
  - Find specific sections and provisions
  - Cross-reference multiple acts
  - Identify relevant case precedents
  - Analyze legal arguments
  - Compare historical and current provisions
```

#### Agent 3: Case Analysis Agent (For Lawyers)
```
Purpose: Analyze cases and find precedents
Capabilities:
  - Match case facts to relevant laws
  - Find similar case studies
  - Analyze verdict patterns
  - Suggest legal strategies
```

### 5.2 Tool Definitions

```python
tools = [
    # Retrieval Tools
    "search_acts",           # Semantic search in acts
    "search_cases",          # Search case studies
    "exact_section_lookup",  # BM25 for exact matches
    "filter_by_metadata",    # Self-query filtering
    
    # Analysis Tools
    "summarize_act",         # Summarize lengthy acts
    "compare_provisions",    # Compare different sections
    "extract_penalties",     # Extract punishment info
    "find_related_acts",     # Find connected legislation
    
    # Utility Tools
    "translate_query",       # EN <-> BN translation
    "simplify_legal_text",   # For normal users
]
```

---

## 6. Project Structure

```
law_buddy/
├── data/
│   ├── acts/                    # 1484 JSON files (existing)
│   ├── case_studies/            # Markdown files (existing)
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Settings and constants
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── act_processor.py     # Parse and chunk acts
│   │   ├── case_processor.py    # Parse case studies
│   │   └── metadata_extractor.py # Extract metadata
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # HuggingFace embeddings
│   │   ├── chroma_store.py      # ChromaDB operations
│   │   ├── bm25_store.py        # BM25 index
│   │   └── hybrid_retriever.py  # Combined retrieval
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py        # Base agent class
│   │   ├── public_agent.py      # For normal people
│   │   ├── research_agent.py    # For lawyers (research)
│   │   └── case_agent.py        # For lawyers (case analysis)
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── query_classifier.py  # Classify user intent
│   │   ├── retrieval_chain.py   # RAG chain
│   │   └── response_chain.py    # Generate responses
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_tools.py      # Search implementations
│   │   ├── analysis_tools.py    # Analysis implementations
│   │   └── utility_tools.py     # Helper tools
│   │
│   └── prompts/
│       ├── __init__.py
│       ├── system_prompts.py    # System prompts
│       └── templates.py         # Prompt templates
│
├── app/
│   └── streamlit_app.py         # Streamlit MVP
│
├── scripts/
│   ├── ingest_data.py           # Data ingestion script
│   ├── test_vectorstore.py      # Test vector operations
│   └── evaluate_rag.py          # Evaluation script
│
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_agents.py
│
├── .env                         # API keys
├── .gitignore
├── pyproject.toml
├── PLAN.md                      # This file
└── README.md
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Week 1-2) ✓ CURRENT
- [x] Set up project structure
- [x] Implement embeddings module (HuggingFace API)
- [x] Set up ChromaDB with collections
- [x] Implement BM25 store
- [x] Test similarity search functionality
- [ ] Implement data processors for acts and case studies
- [ ] Create chunking and metadata extraction
- [ ] Implement basic ingestion pipeline

### Phase 2: Core RAG (Week 3-4)
- [ ] Implement self-query retriever with metadata filtering
- [ ] Add parent document retrieval
- [ ] Create reranking pipeline
- [ ] Build hybrid retriever (Dense + BM25)
- [ ] Test retrieval quality with evaluation metrics

### Phase 3: Agent System (Week 5-6)
- [ ] Implement query classifier
- [ ] Build public assistant agent
- [ ] Build legal research agent
- [ ] Build case analysis agent
- [ ] Create tool implementations
- [ ] Implement LangGraph orchestration

### Phase 4: MVP UI (Week 7)
- [ ] Build Streamlit interface
- [ ] Implement user type selection
- [ ] Add conversation history
- [ ] Display citations and sources
- [ ] Add Bengali/English toggle

### Phase 5: Evaluation & Refinement (Week 8)
- [ ] Create evaluation dataset
- [ ] Test retrieval accuracy
- [ ] Test response quality
- [ ] Optimize prompts
- [ ] Performance tuning

---

## 8. Key Technical Decisions

### 8.1 Embedding Model Choice
**`intfloat/multilingual-e5-large`** via HuggingFace Inference API
- Supports 100+ languages including Bengali
- 1024 dimensions, good balance of quality/speed
- Works well for legal text
- API-based (not local) for easier deployment

### 8.2 Retrieval Strategy
```python
# Hybrid search formula
final_score = α * dense_score + (1-α) * bm25_score
# where α = 0.7 (favor semantic for concepts, but respect exact matches)
```

### 8.3 Context Window Management
- Mistral Large: 32K context, use up to 8K for retrieval
- Prioritize: 5-7 most relevant chunks
- Include: Act metadata + relevant sections + case precedents

### 8.4 Citation Format
```
[Act Name, Year - Section X]
Example: [The Penal Code, 1860 - Section 420]

[Case Reference]
Example: [State vs Rahman, 2022 - BD-CR-001]
```

---

## 9. Sample User Flows

### Flow 1: Normal Person Query
```
User: "My landlord is trying to evict me without notice. What are my rights?"
User Type: Normal Person

1. Query classified as: Housing/Property rights
2. Retrieve: Transfer of Property Act 1882, relevant tenancy sections
3. Find: Similar case studies (if any)
4. Generate: Simple language response with:
   - Your rights explained simply
   - What the law says (cited)
   - Recommended next steps
   - Advice to consult a lawyer for specific situation
```

### Flow 2: Lawyer Query
```
User: "Find all provisions related to breach of contract penalties 
       and any precedents where partial damages were awarded"
User Type: Lawyer

1. Query classified as: Legal Research + Case Analysis
2. Multi-step agent execution:
   a. Search acts for "breach of contract", "damages", "compensation"
   b. Filter by: Contract Act 1872, related commercial laws
   c. Search case studies for: verdict="partial", area="Contract Law"
3. Generate: Comprehensive response with:
   - Relevant sections with full text
   - Case precedents analysis
   - Comparison of provisions
   - Full citations
```

---

## 10. Future SaaS Features (Post-MVP)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Document Upload** | Lawyers upload case documents for analysis | High |
| **Case Tracking** | Save and track ongoing cases | High |
| **Legal Templates** | Generate legal document templates | Medium |
| **Subscription Tiers** | Free (limited), Pro (full access), Enterprise | High |
| **API Access** | REST API for integrations | Medium |
| **Analytics Dashboard** | Usage analytics for lawyers | Low |
| **Notification System** | Law updates and amendments | Medium |
| **Collaboration** | Team workspace for law firms | Low |
| **Mobile App** | iOS/Android apps | Low |

---

## 11. Deployment Architecture (Future)

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│  Next.js + React (Web) / React Native (Mobile)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    API Gateway                               │
│  FastAPI + Better Auth                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
┌────────▼────────┐ ┌─▼──────────┐ ┌▼─────────────┐
│  RAG Engine     │ │ User Service│ │ Subscription │
│  (Current MVP)  │ │             │ │ Service      │
└────────┬────────┘ └────────────┘ └──────────────┘
         │
┌────────▼─────────────────────────────────────────┐
│              Data Layer                          │
│  ChromaDB + PostgreSQL + Redis                   │
└──────────────────────────────────────────────────┘
```

---

## 12. Evaluation Metrics

### Retrieval Metrics
- **Precision@K**: Relevant results in top K
- **Recall@K**: Coverage of relevant documents
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality

### Generation Metrics
- **Faithfulness**: Response grounded in retrieved context
- **Answer Relevance**: Response addresses the query
- **Citation Accuracy**: Correct attribution of sources
- **Legal Accuracy**: Manual review by legal experts

---

## 13. Cost Estimates (Monthly)

### MVP Phase
| Service | Usage | Cost |
|---------|-------|------|
| HuggingFace API | ~1M tokens/month | Free tier |
| Mistral AI | ~100K tokens/day | ~$30-50 |
| ChromaDB | Local/VPS | Included in VPS |
| VPS (4GB RAM) | 1 instance | ~$12-24 |
| **Total** | | **~$42-74** |

### Production Phase (1000 users)
| Service | Usage | Cost |
|---------|-------|------|
| HuggingFace API | ~10M tokens/month | ~$50 |
| Mistral AI | ~1M tokens/day | ~$300-500 |
| VPS/Cloud | Multiple instances | ~$100-200 |
| Database | PostgreSQL + Redis | ~$50 |
| **Total** | | **~$500-800** |

---

## 14. Legal Disclaimers

### Important Notices to Include
1. **For Normal Users**: 
   - "This is AI-generated legal information, not legal advice"
   - "Always consult with a qualified lawyer for your specific situation"
   - "We are not responsible for decisions made based on this information"

2. **For Lawyers**:
   - "AI-assisted research tool, verify all citations and references"
   - "Not a substitute for professional judgment"
   - "Always cross-check with official legal databases"

3. **Data Sources**:
   - "Based on Bangladesh Laws portal data (up to 2025)"
   - "May not reflect latest amendments"
   - "Verify with official sources before use in legal proceedings"

---

## 15. Success Criteria

### MVP Success (8 weeks)
- [ ] System can handle 100+ queries/day
- [ ] Retrieval accuracy > 80% on test set
- [ ] Response time < 10 seconds
- [ ] User satisfaction > 4/5 stars
- [ ] 50+ beta users actively testing

### Production Success (6 months)
- [ ] 1000+ registered users
- [ ] 500+ active monthly users
- [ ] Retrieval accuracy > 90%
- [ ] Response time < 5 seconds
- [ ] User satisfaction > 4.5/5 stars
- [ ] Revenue: $500+/month

---

## 16. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Hallucination** | High | Strict grounding to sources, citation requirements |
| **Incorrect Legal Info** | Critical | Add disclaimers, manual review, expert validation |
| **API Costs** | Medium | Rate limiting, caching, model selection |
| **Data Privacy** | High | Encrypt sensitive data, GDPR compliance |
| **Performance** | Medium | Caching, optimization, load balancing |

---

## Next Steps (Immediate)

1. ✅ Export this plan to PLAN.md
2. ✅ Create folder structure
3. ✅ Set up dependencies
4. ✅ Implement embeddings module
5. ✅ Implement ChromaDB store
6. ✅ Test similarity search
7. ⏳ Implement data processors
8. ⏳ Build ingestion pipeline
9. ⏳ Create evaluation framework

---

**Last Updated**: February 1, 2026
**Status**: Phase 1 - Foundation (In Progress)
