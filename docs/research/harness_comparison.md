# Harness Comparison: SWE-agent, AutoCodeRover, and Devin-Style Agents

**Purpose:** Inform the Kimi-K2.6 Agentic Harness design with proven patterns from existing systems.

---

## Overview

| System | Institution | Open Source | SWE-bench Lite | SWE-bench Full | Core Paradigm |
|--------|-------------|-------------|----------------|----------------|---------------|
| SWE-agent | Princeton NLP / Stanford | Yes | ~18.0% | ~12.47% | ReAct + ACI (Agent-Computer Interface) |
| AutoCodeRover | NUS APR Group | Yes | ~19% pass@1 | ~12.4% pass@1 | Two-stage: Context Retrieval → Patch Generation |
| Devin (Cognition) | Cognition AI | No | Not publicly disclosed | Not publicly disclosed | Planner + Critic + Sandbox |

---

## 1. SWE-agent (NeurIPS 2024)

### 1.1 Agent Loop Structure
SWE-agent uses a **ReAct-style interactive loop**:

```
System Prompt + Issue Description
         ↓
  ┌─────────────────────────────────────┐
  │  LM generates THOUGHT + COMMAND   │
  │         (single action per turn)   │
  ↓                                    │
  Docker sandbox executes command      │
  ↓                                    │
  Environment returns observation      │
  ↓                                    │
  History updated, loop back ──────────┘
         ↓
    submit / budget exceeded / format errors
```

- **Turn-based**: One thought + one command per LLM call.
- **History management**: Past observations (except most recent 5) are collapsed into single-line summaries to prevent context-window bloat.
- **Termination**: `submit` command, cost budget exceeded, or 3 consecutive format errors.

### 1.2 Action Parser
- Enforces strict output format: **DISCUSSION** block + **COMMAND** in markdown code fences.
- Configurable via YAML (`parse_function`): supports alternative formats (JSON, XML delimiters).
- **Format error recovery**: If malformed, system injects a `format_error_template` and asks the model to retry. After 3 failures, the episode terminates ("Early Exit").

### 1.3 Tool Set (The ACI)
SWE-agent's core innovation is the **Agent-Computer Interface** — a custom toolkit that outperforms raw shell access.

| Tool | Purpose | Guardrails |
|------|---------|------------|
| `open <path> [<line>]` | Stateful file viewer (100-line window) | Shows line numbers, total length, truncation indicators |
| `scroll_down / scroll_up` | Move window by 100 lines | Prevents flooding |
| `goto <line>` | Jump to specific line | Bounds checking |
| `find_file <name>` | Find files by name | — |
| `search_file <term>` | Search within open file | Max 50 matches; warns if exceeded |
| `search_dir <term>` | Search across directory | Max 50 matches |
| `edit <start>:<end> <text> end_of_edit` | Replace line range in open file | **Post-edit linting** via `flake8`; syntax errors revert the edit |
| `create <filename>` | Create new file | — |
| `submit` | Generate patch and end episode | — |

**Design principles**:
1. Actions are simple and easy to understand.
2. Actions are compact and efficient.
3. Environment feedback is informative but concise.
4. Guardrails mitigate error propagation (e.g., linting on edits).

### 1.4 Memory / Skill Representation
- **In-context memory only**: The entire conversation history (with collapsed old observations) is fed back each turn.
- **No external skill store**: No retrieval-augmented skill memory; relies on demonstrations (optional successful trajectories prepended to history) and system prompt.
- **Environment state**: Stateful variables (`CURRENT_FILE`, `CURRENT_LINE`, `WINDOW_SIZE`) persist across turns via the shell environment.

### 1.5 Failure Recovery Strategy
- **Format error retry**: 3 attempts before early exit.
- **Lint guardrail**: Edits introducing syntax errors are auto-reverted with detailed feedback.
- **No automatic replanning**: The agent must notice failures in observations and adjust its next action manually.

### 1.6 Configuration
- Single YAML file defines: prompt templates, command files, parse functions, history processor, environment variables.
- Highly modular and hackable.

---

## 2. AutoCodeRover (ISSTA 2024 / arXiv)

### 2.1 Agent Loop Structure
AutoCodeRover uses a **two-stage pipeline** rather than a single continuous loop:

```
Stage 1: Context Retrieval
  Issue Description
       ↓
  LLM extracts hints (class/method/file names)
       ↓
  Iterative API calls to structure-aware search
       ↓
  Build optimal context (AST navigation + SBFL hints)
       ↓
  Identify buggy locations

Stage 2: Patch Generation
  Problem + Locations + Retrieval History
       ↓
  LLM generates patch (up to 3 retries with syntax checking)
       ↓
  Patch validation via test execution (if available)
```

- **Retrieval is separate from patching**: The context is fully assembled before any code generation begins.
- **Iterative retrieval**: The LLM decides which API to call next based on prior results, building context dynamically.

### 2.2 Action Parser
- The LLM calls **structured APIs** (not free-form shell commands).
- API calls are function-style with named arguments (e.g., `search_method_in_class(method="foo", class="Bar")`).
- No generic action parser like SWE-agent; the LLM emits API calls within a constrained schema.

### 2.3 Tool Set
AutoCodeRover exposes **structure-aware retrieval APIs** that operate on the AST:

| API | Purpose |
|-----|---------|
| `search_class(name)` | Find class definitions |
| `search_method_in_class(method, class)` | Find method within a class |
| `search_code_in_file(code, file)` | Search for code snippet in file |
| (implied) SBFL integration | Spectrum-based fault localization feeds top suspicious methods to LLM |

**Key difference from SWE-agent**: Tools navigate the **program structure** (AST) rather than the file system. This is described as "software-engineering-oriented" rather than "AI-agent-oriented."

### 2.4 Memory / Skill Representation
- **Growing retrieval context**: Problem statement + retrieved code + previous API results + LLM analysis accumulates across retrieval iterations.
- **No persistent skill store**: Skills are not extracted or reused across sessions.
- **AST as structured memory**: The codebase itself is parsed into an AST, which acts as a queryable structural memory.

### 2.5 Failure Recovery Strategy
- **Syntax checking on patch generation**: Up to 3 retries if generated patch has syntax errors.
- **Test-driven validation**: If tests are available, patches are validated before submission.
- **SBFL augmentation**: When tests exist, fault localization provides additional hints to guide retrieval.

### 2.6 Efficiency
- Average cost: **~$0.43 USD per issue** on SWE-bench Lite.
- Average resolution time: **~3–4 minutes**.
- **65–66%** of plausible patches are semantically correct.

---

## 3. Devin-Style Agent Architecture (Cognition AI)

### 3.1 Agent Loop Structure
Devin uses a **dual-model, goal-directed iterative loop**:

```
User Request / Issue
       ↓
Planner Model → Generate multi-step plan
       ↓
Execute in sandboxed container
       ↓
Critic Model → Review changes for logic/security issues
       ↓
Run tests
       ↓
[Failure] → Re-plan and iterate
       ↓
[Success] → Produce PR with rationale
```

- **Planner + Critic**: Two distinct models create an internal feedback loop for quality control before any action is finalized.
- **Self-healing test loop**: Tests → failures → automatic re-planning → iterative fixes.
- **Human-in-the-loop by design**: Produces detailed PRs; human review remains mandatory.

### 3.2 Action Parser
- Not publicly documented, but follows the industry-standard pattern: LLM emits structured JSON/tool-calls, external orchestration executes them.

### 3.3 Tool Set
Devin has a comprehensive tool registry:

| Category | Tools |
|----------|-------|
| File System | Read, write, multi-file patch, large-scale refactor |
| Terminal | Full shell access, builds, linters, migration scripts |
| Browser | Web browsing for documentation/research |
| Search | Semantic codebase search, repository-wide pattern recognition |
| Testing | Test suite execution, self-healing iteration |
| PR Management | GitHub/GitLab integration, review comment responses |
| Delegation | Managed "sub-Devins" — parallel child sessions in isolated VMs |

### 3.4 Memory Architecture (4-Tier)

| Tier | Type | Persistence | Use Case |
|------|------|-------------|----------|
| **L1** | In-Context | Ephemeral (128K–10M+ tokens) | Active conversation, file contents, tool results |
| **L2** | External Short-Term Cache | Session-scoped (Redis/local) | Tool outputs, preferences, intermediate state |
| **L3** | Long-Term Vector Store | Persistent (Chroma/Pinecone/LanceDB) | Codebase embeddings, past session summaries |
| **L4** | Parametric Memory | Static (model weights) | Pre-trained knowledge |

- **Enterprise ingestion**: Supports 10M+ token contexts for whole-repo reasoning.
- **Knowledge Base**: Deduplicates and consolidates codebase patterns into reusable knowledge entries.
- **Playbooks**: Reusable session templates created from successful past sessions.
- **Session Analysis**: Analyzes past sessions to extract patterns, identify failures, and improve prompts.

### 3.5 Failure Recovery Strategy
- **Self-healing test loop**: Automatic re-planning on test failures.
- **Critic review**: Catches logic and security issues before execution.
- **Max iteration guards, token budgets, timeouts**: Safety boundaries prevent runaway loops.
- **Subagent delegation**: Complex tasks are broken into parallel child sessions.

### 3.6 Open Source Status
- **Devin is proprietary and closed-source.**
- No full open-source clone exists.
- Architectural patterns (ReAct loop, 4-tier memory, sandboxed execution) are implemented in SWE-agent, Aider, OpenDevin/E2B, and LangGraph.

---

## 4. Comparison Matrix

| Dimension | SWE-agent | AutoCodeRover | Devin (Cognition) | Kimi-K2.6 Harness (Current) |
|-----------|-----------|---------------|-------------------|------------------------------|
| **Loop Structure** | ReAct (turn-based) | Two-stage (retrieve → patch) | Planner + Critic + Self-heal | TVC (plan → execute → verify → correct) |
| **Actions per Turn** | 1 (thought + command) | 1 API call (retrieval stage) | Multi-step plans | 1 (plan or execute or correct) |
| **Action Format** | Markdown code blocks (configurable) | Structured API calls (AST-based) | JSON/tool-calls (assumed) | Free-form LLM text → shell.exec |
| **Tool Set** | File viewer, search, edit, submit | AST search APIs (class/method/code) | File, shell, browser, search, PR, subagents | Shell.exec only |
| **Tool Abstraction** | ACI (custom LM-centric tools) | Program-structure-aware APIs | Full general-purpose toolkit | Raw shell (no abstraction) |
| **Memory / Context** | In-context history (collapsed old obs) | Growing retrieval context | 4-tier (in-context → cache → vector → parametric) | LangGraph checkpointing + SQLite skill store |
| **Skill Representation** | None (demonstrations only) | None | Playbooks, knowledge base, embeddings | zstd-compressed traces, tag-based retrieval |
| **Failure Recovery** | Format retry, lint guardrail | Syntax retry, test validation | Self-healing loop, critic review, replan | Correct node (LLM analyzes failure → new plan) |
| **Sandboxing** | Docker | Not emphasized (test validation) | Docker/Firecracker per-session | None (subprocess on host) |
| **Benchmark Performance** | ~18% Lite, ~12.5% Full | ~19% Lite, ~12.4% Full | Not disclosed | Mock mode only |
| **Cost Efficiency** | GPT-4 class (~$1–2/issue) | ~$0.43/issue (very low) | Enterprise pricing | Ollama local (near-zero marginal cost) |
| **Open Source** | Yes | Yes | No | Yes |

---

## 5. Design Implications for Kimi-K2.6 Harness

### 5.1 What to Adopt

1. **SWE-agent's ACI principles**
   - Replace raw `shell.exec` with purpose-built tools (file viewer, search, edit).
   - Add guardrails (linting on edit, truncation on search output).
   - Enforce a strict action format with parser + retry logic.

2. **AutoCodeRover's structure-aware retrieval**
   - For code-related tasks, AST-based search beats text grep.
   - Separate context retrieval from execution (our plan node already does this, but we can augment with AST tools).

3. **Devin's memory tiers**
   - Our L1–L4 memory architecture is well-aligned; we should implement the missing tiers (L3 vector search, L4 parametric is implicit via model weights).
   - Playbooks / skill extraction from successful traces is a Phase 4 goal; this validates the priority.

4. **Planner/Critic separation (Devin)**
   - Our current TVC loop has plan, execute, and correct nodes. Adding an explicit **critic** node between execute and verify could improve quality.

### 5.2 What to Avoid

1. **Over-reliance on free-form shell**
   - SWE-agent shows that raw shell underperforms a custom ACI by ~6 percentage points on SWE-bench Lite.
   - Our execute node currently passes raw LLM output to `subprocess.run(..., shell=True)`. This is the #1 security and performance risk flagged in our design decisions.

2. **No structured action parser**
   - AutoCodeRover's constrained API schema prevents hallucinated commands.
   - We need a parse layer that maps LLM output to validated tool calls, not raw strings.

3. **Missing sandboxing**
   - SWE-agent and Devin both use Docker. We currently run on the host. Before real benchmark runs, we need container isolation.

### 5.3 Prioritized Recommendations

| Priority | Recommendation | Phase |
|----------|----------------|-------|
| **P0** | Implement structured action parser (DISCUSSION + COMMAND format) with retry | Phase 2 |
| **P0** | Add sandboxed execution (Docker/Firecracker) for execute node | Phase 2 |
| **P1** | Replace raw shell with ACI-style tools (open, search, edit, submit) | Phase 2 |
| **P1** | Add AST-based code search tools (leverage tree-sitter) | Phase 3 |
| **P2** | Add explicit critic node in TVC loop | Phase 3 |
| **P2** | Implement L3 vector memory for skill retrieval | Phase 4 |
| **P3** | Subagent delegation for parallel task execution | Phase 5 |

---

## Sources

- SWE-agent GitHub: [github.com/princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent/)
- SWE-agent Paper: [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
- SWE-agent Docs: [swe-agent.com](https://swe-agent.com/0.7/background/)
- AutoCodeRover GitHub: [github.com/nus-apr/auto-code-rover](https://github.com/nus-apr/auto-code-rover)
- AutoCodeRover Paper: [arXiv:2404.05427](https://arxiv.org/html/2404.05427v2)
- AutoCodeRover Website: [autocoderover.dev](https://autocoderover.dev/)
- Devin Docs: [docs.devin.ai](https://docs.devin.ai/work-with-devin/advanced-capabilities)
- AI Agent Architecture: [abstractalgorithms.dev](https://abstractalgorithms.dev/how-ai-coding-agents-work)

---

*Document generated: 2026-04-25*
