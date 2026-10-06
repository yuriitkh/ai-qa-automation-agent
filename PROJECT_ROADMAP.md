# AI QA Automation Agent — Project Roadmap

This roadmap documents the current state of the AI QA Automation Agent and its planned evolution toward a multi-project QA platform. It is architectural guidance, not an implementation specification.

## 1. Current State

The project is a local QA automation framework that turns a natural-language QA task into executed browser tests.

Currently existing:

- **Natural-language task input** — the user describes what to test in plain language.
- **`TestCaseDecomposer`** — LLM-backed decomposition of a task into a `TestCase`.
- **`TestCase` / `TestStep`** — domain model of testing intent (what to test, step by step).
- **Deterministic Browser Discovery** — browser-based discovery of interactive elements, navigation paths, and page structure.
- **AI Discovery fallback** — optional LLM-assisted discovery suggestions, merged into deterministic results when deterministic discovery is not fully successful.
- **`LLMTestPlanGenerator`** — generates a structured, validated `QATestPlan` from a `TestStep` and a discovery result.
- **Multi-provider LLM Router** — routes `TEST_PLAN` and `DISCOVERY` requests across configured providers (default order is environment-driven, e.g. openai, gemini, openrouter, groq).
- **Provider fallback** — retryable provider failures move to the next provider; intentional hard stops (missing configuration or authentication) halt immediately.
- **Structured `QATestPlan`** — ordered `QATestStep` records with actions and parameters.
- **`TestPlan` / `TestPlanVersion`** — a `TestPlan` describes how to execute one `TestStep`; each generation creates an immutable `TestPlanVersion`.
- **Playwright `BrowserRunner`** — executes a `QATestPlan` in a real browser and produces structured results.
- **Deterministic locator recovery** — conservative matching of stale planned selectors against freshly discovered elements.
- **Stale-plan regeneration** — when deterministic recovery cannot match, the plan is regenerated with the LLM as a new `TestPlanVersion`.
- **`Execution` / `ExecutionRepository`** — concrete execution records with status, timestamps, errors, and evidence.
- **Reporting** — structured test report generation from runs.
- **Structured Execution Trace** — provider-independent observability artifact for one pipeline run.
- **Plan caching/versioning** — `PlanStore` (in-memory and SQLite) caches the latest `TestPlanVersion` per `TestStep`.

**Architectural principle: Deterministic first, AI second.**

Deterministic browser discovery and deterministic execution are preferred. AI is used only where deterministic mechanisms cannot resolve the task: decomposing the task, generating a plan when there is no cache, discovery fallback, and plan regeneration after a stale-UI failure.

## 2. Current Architecture

Conceptual pipeline:

```
Natural Language Task
→ TestCaseDecomposer
→ TestCase
→ TestStep(s)
→ Browser Discovery
→ LLMTestPlanGenerator
→ LLM Router
→ QATestPlan
→ BrowserRunner
→ Execution
→ Evidence / History / Reporting / ExecutionTrace
```

Current domain relationships:

```
TestCase
  ↓
TestStep
  ↓
TestPlan
  ↓
TestPlanVersion
  ↓
Execution
```

Responsibilities of the major objects:

- **TestCase** — the complete testing intention produced from the natural-language task.
- **TestStep** — one atomic testing intention within a `TestCase`.
- **TestPlan** — describes HOW a `TestStep` is executed (the `QATestPlan` with ordered `QATestStep` actions).
- **TestPlanVersion** — an immutable, executable version of a `TestPlan`. Historical executions always refer to a specific version.
- **Execution** — one concrete run of a `TestPlanVersion`: status, timestamps, planned step index, actual result, error, and evidence.
- **Evidence** — execution artifacts (currently screenshots), especially for failures.
- **TestRun** — aggregate result of all executions of a `TestCase` in one pipeline run.
- **ExecutionTrace** — structured diagnostic information about the pipeline run itself: decomposition, discovery, provider attempts, plan generation, execution attempts, locator recovery, regeneration, errors, and timings.

## 3. Near-Term Roadmap

### 3.1 Structured Execution Trace
**Status: COMPLETED**

The Structured Execution Trace is implemented and committed (`qa_agent/execution_trace.py`, `qa_agent/redaction.py`, integration in `QATestPipeline` and `LLMRouter`). It records decomposition, per-step discovery (including AI fallback), plan cache HIT/MISS, provider attempts with fallback outcomes, plan generation, execution attempts, locator recovery, regeneration, errors, and phase timings. Recording is best-effort and presentation-independent; secrets are redacted at record time.

### 3.2 Clean CLI
**Status: NEXT**

The CLI should become a clean user-facing entry point for running a natural-language QA task and displaying the execution result and trace.

Implementation details are intentionally not designed yet.

### 3.3 Stability and Failure-Path Testing
**Status: PLANNED**

Focus on:

- provider failures
- fallback behavior
- invalid LLM output
- discovery failures
- stale locators
- deterministic recovery
- regeneration
- assertion failures
- infrastructure/browser failures
- trace correctness
- cache/version behavior

### 3.4 Small Web UI
**Status: PLANNED**

The future Web UI should initially be a thin presentation layer over the existing pipeline.

Conceptually:

```
POST /run
    ↓
QATestPipeline
    ↓
ExecutionTrace
    ↓
HTML/Web UI
```

The UI should eventually allow:

- entering the natural-language task
- running the test
- viewing TestCase/TestSteps
- viewing Discovery
- viewing provider attempts/fallback
- viewing the generated TestPlan
- viewing execution results
- viewing evidence
- viewing the final report

Do not implement this now.

### 3.5 README / Public Demo
**Status: PLANNED**

The project should later be presented as a working AI QA Automation prototype with one or more polished end-to-end demonstrations and public documentation.

## 4. Future QA Platform

The longer-term evolution is from a local automation framework into a multi-project QA platform.

Conceptual hierarchy:

```
Organization
  └── Users
  └── Projects
       ├── Environments
       ├── Product Versions
       ├── Test Suites
       │    └── Test Cases
       │         └── Test Case Versions
       │              └── Test Steps
       │                   └── Test Plan Versions
       └── Test Runs
            └── Executions
                 └── Evidence
```

`ExecutionTrace` is a cross-cutting execution artifact that remains useful at every level for diagnostics and reporting.

## 5. Future Domain Model

The concepts below are documented for future design; none of them are implemented now.

### Organization
Top-level tenant/container for users and projects.

### User
A person working with one or more projects.

Authentication/login should be treated as a separate authentication subsystem rather than introducing a separate "Login" domain entity.

### Roles / Access
Initially conceptual:

- Admin
- User

Detailed RBAC is not designed yet.

### Project
A QA automation/testing project. Multiple users may work on the same project.

### Environment
Examples:

- DEV
- TEST
- STAGING
- PROD

### Product Version
The version/build of the system under test.

### Test Suite
Logical grouping of Test Cases.

Possible categories include:

- Functional
- Regression
- Smoke
- Integration
- Performance
- Security

**Test Case lifecycle status and Test Suite/type are different concepts and must not be merged.**

### Test Case
Represents the complete testing intention.

Potential attributes:

- ID
- description
- preconditions
- status
- type/category/tags
- steps

Lifecycle status is conceptually separate from type/category.

Possible lifecycle statuses:

- ACTIVE
- INACTIVE
- ARCHIVED

### Test Case Version
Immutable historical version of a Test Case.

If a Test Case changes from v1 to v2, previous executions must still refer to v1.

### Test Step
Represents one atomic testing intention.

The current architecture already contains `TestStep` and `TestPlan`.

Do NOT introduce an independent `TestStepVersion` unless a future requirement demonstrates that it is necessary.

### Test Plan
Describes HOW a Test Step is executed.

### Test Plan Version
Immutable executable version of a Test Plan.

A Test Case Version may therefore contain a Test Step whose executable implementation references a specific `TestPlanVersion`.

Example:

```
Test Case v2
  → Test Step "Open Boliglån"
      → Test Plan v5
```

### Operating Modes: AI Autonomy vs Human Approval

**Status: FUTURE — planned concept only; not implemented.**

This section is architectural guidance. It intentionally specifies no database schema, UI implementation, API design, or code changes. None of this functionality exists today.

The project should support two primary operating modes.

#### Autonomous Mode

The AI may:

- decompose the user's task into TestSteps;
- define Expected Results;
- generate Test Plans;
- execute the TestCase.

Human approval is not required before execution.

#### Controlled / Approved Mode

The AI proposes the TestCase and TestPlan, but a human can review and approve them before execution.

The human should eventually be able to:

- edit TestStep text;
- edit the Expected Result;
- edit the executable Test Plan;
- approve a Step / Expected Result / Plan;
- see which content was AI-generated;
- see which content was manually edited;
- see the current approved version;
- view change history;
- see what changed (before → after);
- see who changed it;
- see when it was changed;
- optionally add a comment explaining the change.

#### Approval versioning

Execution should reference a specific approved/versioned definition:

```
AI generated
    ↓
Draft
    ↓
Human edited
    ↓
Approved Version
    ↓
Execution
```

Example progression of one plan:

```
TestPlan v1 — AI generated
TestPlan v2 — AI regenerated
TestPlan v3 — Human edited
TestPlan v4 — Human approved
```

This builds on the existing `TestPlanVersion` chain rather than introducing a parallel versioning concept: AI generations and regenerations already produce new immutable versions of a Test Plan today, and in the future the same versions would additionally carry draft/approval state and edit provenance.

Content distinctions to preserve in this future model:

- AI-generated content;
- human-edited content;
- approved content;
- execution based on a specific version.

### Test Run
Represents one complete run of a Test Case or a collection of Test Cases.

It should conceptually preserve:

- Test Case Version
- user who started the run
- date/time
- environment
- product version/build
- browser and browser version
- overall result
- comments
- report
- evidence

Keep Test Run separate from Execution.

Example:

```
Test Run #152
  → TC #101 PASS
  → TC #102 FAIL
  → TC #103 PASS
  → TC #104 SKIPPED
```

### Execution
Represents the concrete execution of a Test Step / Test Plan Version.

It contains the actual result, timestamps, errors, and evidence.

Suggested future result categories:

- PASS
- FAIL
- BLOCKED
- SKIPPED
- ERROR

Keep FAIL and ERROR conceptually distinct.

### Evidence
Screenshots and other execution artifacts, especially for failures.

### Execution Trace
Structured diagnostic/execution information across the pipeline.

It should remain useful for:

- CLI
- JSON
- HTML reports
- Web UI
- future persistence

## 6. Architectural Principles

1. Deterministic first, AI second.
2. LLM providers are interchangeable.
3. Provider failures should not unnecessarily fail the entire pipeline when fallback is possible.
4. Test Plans are structured and validated before execution.
5. Historical Test Plan versions must remain reproducible.
6. Test Case Versions must preserve historical intent.
7. Execution is separate from planning.
8. Test Run is separate from individual Execution.
9. Execution Trace is a presentation-independent artifact.
10. UI should remain a thin layer over the existing pipeline.
11. Domain models should remain independent from infrastructure.
12. Avoid premature infrastructure and platform complexity.

## 7. Explicitly Deferred

These features are intentionally NOT part of the immediate roadmap:

- full database/platform implementation
- authentication implementation
- advanced RBAC
- cloud deployment
- billing
- teams/invitations
- SSO
- audit logs
- notifications
- Jira/Slack integrations
- API tokens
- mobile testing
- desktop testing
- distributed execution
- queues
- vector DB/RAG
- multi-agent architecture
- autonomous crawling
- Docker/Kubernetes

These may become future features, but they should not drive current architecture prematurely.

## 8. Roadmap Summary

**Phase 1 — Core Engine**
COMPLETED

**Phase 2 — Observability**
COMPLETED
- Structured Execution Trace

**Phase 3 — Developer/User Experience**
NEXT
- Clean CLI
- stability/failure-path testing

**Phase 4 — Presentation**
PLANNED
- Small Web UI
- HTML report
- polished E2E demo
- README/public documentation

**Phase 5 — QA Platform Foundation**
FUTURE
- Organizations
- Users
- Projects
- Environments
- Test Suites
- Test Cases
- Test Case Versions
- Test Runs
- persistence

**Phase 6 — Platform Expansion**
FUTURE
- authentication/RBAC
- integrations
- additional test types
- distributed/cloud execution
- etc.
