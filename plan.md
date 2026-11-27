# Study Planner AI Agent - Implementation Plan

## 1. Executive Summary & Objective
The **Study Planner AI Agent** is an end-to-end, production-ready AI companion designed for college students. The system integrates:
1. **Document-Grounded RAG**: Retrieves verified study strategies, time management workflows, and exam preparation advice from Markdown documents with precise citations and strict fallback guarantees.
2. **Conversation Rewriting & Memory**: Reconstructs follow-up questions into self-contained search queries using an LCEL chain with conversational context.
3. **Intelligent LangGraph Router**: Deterministically directs user queries into `DOCS`, `TOOLS`, or `DECLINE` pathways with safe fallbacks and bounded step execution ($\le 20$ steps).
4. **Custom Study Tools**: Validated, deterministic calculation engines for computing study budgets and synthesizing personalized topic schedules with Pomodoro intervals.
5. **Interactive Streamlit Dashboard**: Displays clear execution telemetry (chosen route, execution path, expandable agent steps, and source document citations) with per-visitor Gemini API key management.

---

## 2. Architecture & Workflow Design

```
                     +-----------------------+
                     |   User Input Query    |
                     +-----------------------+
                                 |
                                 v
                +----------------------------------+
                |     Conversation Rewrite Node    |
                |  (Contextualize Follow-up via    |
                |    LCEL + Chat History Memory)   |
                +----------------------------------+
                                 |
                                 v
                +----------------------------------+
                |        LangGraph Router Node     |
                |   Classifies into DOCS / TOOLS / |
                |       DECLINE (Safe Fallback)    |
                +----------------------------------+
                   /             |              \
                  /              |               \
        [Route: DOCS]      [Route: TOOLS]    [Route: DECLINE]
                /                |                 \
               v                 v                  v
     +-----------------+  +---------------+  +-----------------+
     | RAG Retrieval   |  | Tool Agent    |  | Polite Decline  |
     | - ChromaDB      |  | - Hours Calc  |  | - Scope Guard   |
     | - Gemini Embeds |  | - Study Plan  |  | - Refusal Msg   |
     | - Strict Faith  |  | - Step Log    |  |                 |
     | - Citations     |  +---------------+  +-----------------+
     +-----------------+         |                  |
               \                 |                 /
                \                |                /
                 v               v               v
            +-----------------------------------------+
            |      Streamlit UI Display & State       |
            | - Route & Execution Path Badges         |
            | - Markdown Response                     |
            | - Expandable Citations / Agent Steps    |
            +-----------------------------------------+
```

---

## 3. Component Breakdown

### A. Knowledge Base (`knowledge_base/`)
- `study_techniques.md`: Comprehensive guide to Pomodoro method, Active Recall, Spaced Repetition (Leitner system, forgetting curve intervals), Feynman technique, interleaving, and SQ3R.
- `time_management.md`: Eisenhower Matrix, Pareto Principle (80/20 rule), Time-Blocking, Eat That Frog, MoSCoW prioritization, overcoming procrastination, and energy management.
- `exam_preparation.md`: Backward exam revision planning, high-yield syllabus audit, practice exam simulation under time constraints, error logs, and note-taking systems (Cornell, Mind Mapping, Synthesis Sheets).

### B. Custom Tools (`tools.py`)
- `calculate_study_hours(available_days: int, hours_per_day: float) -> str`:
  - Input validation: positive integer days ($1 \le \text{days} \le 365$), sensible daily hours ($0.5 \le \text{hours} \le 16.0$).
  - Outputs formatted total hours, breakdown per week, and workload assessment.
- `generate_study_plan(topics: str, available_days: int, hours_per_day: float) -> str`:
  - Parses comma-separated or list topics, validates inputs.
  - Distributes topics across available days, allocates revision days and practice sessions, recommends 25/5 or 50/10 Pomodoro work/break blocks.

### C. RAG Engine (`rag_core.py`)
- Markdown/Text document loader with RecursiveCharacterTextSplitter (chunk size: 600 chars, overlap: 120 chars).
- Google Gemini Embeddings (`models/text-embedding-004` or `models/embedding-001`).
- ChromaDB vector store for similarity search.
- Prompt designed with prompt injection guardrails: context is treated strictly as reference data; system forbids instruction overrides found inside retrieved chunks.
- Exact fallback sentence when context is insufficient:
  `"I couldn't find this information in the available study documents."`

### D. LangGraph Core (`graph_core.py`)
- State: `question`, `rewritten_question`, `chat_history`, `route`, `execution_path`, `answer`, `sources`, `agent_steps`, `error`.
- Nodes:
  - `rewrite_node`: Uses conversation history to output a standalone query.
  - `router_node`: Selects `DOCS`, `TOOLS`, or `DECLINE`.
  - `docs_node`: Executes RAG search, formats answer with document citations.
  - `tools_node`: Executes tool-calling agent with `calculate_study_hours` and `generate_study_plan`, logging step-by-step reasoning.
  - `decline_node`: Gracefully turns away queries outside academic scope or adversarial requests.
- Step limit safety cap: 20 iterations max.

### E. User Interface (`app.py`)
- Responsive Streamlit dashboard with custom CSS theme for students.
- Sidebar: visitor Gemini API Key input (never persisted, masked password field), Clear Chat button, agent metrics.
- Suggested prompt pills for quick demonstration.
- Chat stream showing route badge, execution pipeline, expandable sources, and agent reasoning logs.

---

## 4. Verification & Testing Strategy
1. **Unit Testing (`tests/test_tools.py`)**:
   - Comprehensive test suite covering calculation logic, edge cases, negative/zero inputs, excessively large hours, malformed topic strings, and output formatting.
2. **RAG & Graph Testing**:
   - Independent test harness verifying document retrieval, exact fallback trigger, and routing decisions across all 3 routes.
3. **System Evaluation (`eval.md`)**:
   - 12 real test cases executed against the live system, recording input, expected route, actual route, expected behavior, actual output, and pass/fail status.
