# AI QA Agent — Project Roadmap

## 1. Project Vision

AI QA Agent is an open-source AI-powered web testing framework.

The user provides a natural-language testing task, for example:

> "Check the navigation menu, open the Loans section, verify the page title and heading, then test the search form."

The agent should:

1. Understand the task.
2. Discover the website structure.
3. Identify relevant navigation and interactive elements.
4. Split complex tasks into atomic test steps.
5. Generate a structured Test Plan.
6. Execute the Test Plan using Playwright.
7. Capture execution evidence.
8. Return a clear PASS/FAIL result.
9. Reuse successful Test Plans on future executions.
10. Re-discover and regenerate the Test Plan when the cached plan no longer works.

The architecture must remain modular so that LLMs, Discovery engines, browsers and other technologies can be replaced or extended without rewriting the core application.

---

# 2. Current Architecture

```text
Natural Language Task
        |
        v
Browser Discovery
        |
        v
LLM Router
   /          \
Gemini       Groq
   \          /
    \        /
     Test Plan
        |
        v
Playwright Runner
        |
        v
Execution Result
```

## Current main components

### Models

`qa_agent/models.py`

Current structured models include:

- `QATestPlan`
- `QATestStep`

Current supported actions include:

- `navigate`
- `assert_page_loaded`
- `assert_title`
- `assert_visible`
- `click`
- `fill`
- `assert_hidden`
- `assert_url`

### Browser Discovery

`qa_agent/browser_discovery.py`

Current discovery includes:

- URL
- title
- headings
- links
- buttons
- visible text elements
- navigation menu
- navigation paths
- direct navigation paths under development

Current real-site investigation includes:

- DNB custom navigation
- FINN direct navigation
- NAV hamburger/drawer navigation

### LLM layer

`qa_agent/llm/`

Current providers:

- Gemini
- Groq

Current architecture:

```text
LLMProvider
    |
    +-- GeminiProvider
    |
    +-- GroqProvider
```

`Router` supports provider fallback.

### Browser execution

`qa_agent/browser_runner.py`

Current Playwright execution supports:

- navigation
- clicks
- filling fields
- page-load assertions
- title assertions
- visibility assertions
- hidden assertions
- URL assertions

The runner stops on the first failed step.

### Current testing status

The project currently has a substantial automated test suite and recent runner changes resulted in:

- 47 tests passed
- 14 subtests passed

The discovery work has already uncovered several real-world navigation patterns rather than relying only on theoretical examples.

---

# 3. Architectural Principles

## Principle 1 — Deterministic first, AI second

Use Playwright/DOM/accessibility information whenever possible.

AI should be used when deterministic discovery cannot understand the page.

```text
Deterministic Discovery
        |
        | sufficient?
        |
      YES ----------------> Test Planning
        |
       NO
        |
        v
   AI Discovery
        |
        v
   Test Planning
```

## Principle 2 — Provider independence

LLMs must be replaceable.

```text
LLMProvider
 ├── Gemini
 ├── Groq
 ├── OpenAI
 └── future providers
```

Discovery should use the same principle:

```text
DiscoveryProvider
 ├── PlaywrightDiscovery
 ├── StagehandDiscovery
 └── future providers
```

Do not hard-code the project around one vendor.

## Principle 3 — Deterministic execution

The LLM generates a structured Test Plan.

Playwright performs the actual browser actions.

The LLM should not directly control every browser action unless an AI fallback is explicitly required.

## Principle 4 — Evidence over assumptions

Discovery and execution should return measurable results:

- success
- partial
- failure
- coverage
- warnings
- execution evidence

---

# 4. Planned Test Case Architecture

Future model:

```text
TestCase
 ├── ID
 ├── name
 ├── description
 ├── steps
 ├── test plan versions
 └── execution history
```

Example:

```text
TC-0001
Login with valid credentials

1. Open login page
2. Enter email
3. Enter password
4. Click Login
5. Verify dashboard
```

---

# 5. Large Test Case / Step Decomposition

A natural-language task may be too large for one Test Plan.

The agent should first create atomic steps.

Example:

```text
Original:

"Register a user and verify the welcome email."
```

May become:

```text
1. Open registration page
2. Fill registration form
3. Submit registration
4A. Open email verification page
4B. Enter verification code
4C. Submit verification code
4D. Verify successful verification
```

When a step is split, preserve its original number:

```text
10
```

becomes:

```text
10A
10B
10C
```

The goal is to keep individual steps small enough to be deterministic and independently diagnosable.

---

# 6. Test Plan Caching

Successful Test Plans should be persisted.

First execution:

```text
Discovery
   |
   v
LLM
   |
   v
Test Plan
   |
   v
Playwright
   |
   v
PASS
   |
   v
SAVE PLAN
```

Future execution:

```text
Load saved Test Plan
        |
        v
Playwright
        |
        v
PASS
```

No LLM call is required if the saved Test Plan still works.

If execution fails:

```text
Saved Test Plan
       |
       v
     FAIL
       |
       v
Discovery
       |
       v
LLM
       |
       v
New Test Plan
       |
       v
Playwright
       |
       v
PASS/FAIL
```

Test Plans should be versioned.

Example:

```text
TC-0001

Plan v1
Plan v2
Plan v3
```

An execution should record which Test Plan version was used.

---

# 7. Execution History

Every execution should eventually record:

- Test Case ID
- Test Plan version
- date/time
- browser
- browser version
- operating system
- duration
- PASS/FAIL
- failed step
- error
- screenshots
- relevant DOM/snapshot information
- console errors when available

Example:

```text
TC-0001

Run #1  PASS
Run #2  PASS
Run #3  FAIL
Run #4  PASS
```

This will later enable statistics such as:

- pass rate
- failure rate
- average execution time
- most unstable steps
- most common failures

---

# 8. Discovery Result

The current Discovery architecture should evolve from simply returning discovered data into a measurable result.

Possible statuses:

```text
SUCCESS
PARTIAL
FAILED
```

Example:

```json
{
  "status": "partial",
  "coverage": 0.82,
  "strategies_used": [
    "dnb_menu",
    "direct_navigation"
  ],
  "warnings": [
    "Some navigation items could not be replayed"
  ]
}
```

The result should be task-aware.

For example, a task about a login form does not require navigation paths if Discovery successfully finds the relevant form and inputs.

---

# 9. AI Discovery Fallback

When deterministic Discovery cannot provide enough information:

```text
Playwright Discovery
        |
        v
Is information sufficient?
     /        \
   YES         NO
    |           |
    |           v
    |      AI Discovery
    |           |
    +-----+-----+
          |
          v
      Test Plan
```

The AI Discovery layer should be replaceable.

Possible future implementations:

- Stagehand
- browser-use
- other open-source or commercial browser-agent technologies

The first implementation should not replace the existing Discovery engine.

It should be a fallback.

---

# 10. Discovery Statistics

The project should eventually collect non-sensitive technical statistics such as:

```text
Sites tested: 100

Primary Discovery:
  SUCCESS: 78
  PARTIAL: 15
  FAILED: 7

AI fallback required: 7
AI fallback successful: 6
AI fallback failed: 1
```

This data should guide future improvements to `browser_discovery.py`.

The goal is evidence-based development rather than guessing which navigation patterns are common.

---

# 11. Future Web UI

A future web interface may provide:

```text
Dashboard

Test Cases
Executions
Test Plans
History
Statistics
Settings
Providers
```

Test Case editor:

```text
TC-0001
Login with valid credentials

Steps
--------------------------------
1. Open login page
2. Enter email
3. Enter password
4. Click Login
5. Verify dashboard

[Run Test]
[Edit]
[History]
```

This should be implemented only after the underlying models, storage and execution architecture are stable.

---

# 12. Functional Target for AI QA Agent v1.0

The first serious GitHub release should support most common web QA operations.

## Navigation

- open URL
- links
- menus
- submenus
- hamburger menus
- tabs
- drawers
- breadcrumbs
- pagination
- back/forward

## Interactive elements

- buttons
- links
- inputs
- textareas
- checkboxes
- radio buttons
- selects
- date pickers
- sliders
- file inputs

## Forms

- fill fields
- clear fields
- submit
- required-field validation
- validation messages
- success messages
- dependent fields

## Assertions

- text
- visibility
- hidden state
- enabled/disabled
- URL
- title
- attributes
- input values
- checked state
- element count

## Dynamic UI

- dialogs
- modals
- dropdowns
- accordions
- tooltips
- toast messages
- notifications
- loading indicators
- SPA transitions
- AJAX-driven content

## Evidence

- screenshots
- URL
- action log
- timestamps
- errors
- relevant DOM information
- console errors when available

---

# 13. Recommended Development Order

## Phase 1 — Current work

1. Finish `browser_discovery.py`
2. Support multiple navigation patterns
3. Add `direct_navigation_paths`
4. Verify DNB, FINN and NAV
5. Add Discovery tests
6. Introduce `DiscoveryResult`

## Phase 2 — Core architecture

7. Create `DiscoveryProvider`
8. Create `TestCase`
9. Create `TestStep`
10. Create `TestPlanVersion`
11. Separate Test Case from Test Plan

## Phase 3 — Intelligent planning

12. Add task decomposition
13. Detect oversized/non-atomic steps
14. Split steps using `10A`, `10B`, `10C`
15. Validate Test Plan before execution

## Phase 4 — Reuse and performance

16. Persist successful Test Plans
17. Add Test Plan versioning
18. Execute cached plans
19. Rediscover/regenerate after cached-plan failure

## Phase 5 — Execution history

20. Store executions
21. Store per-step results
22. Store browser/environment information
23. Store execution evidence
24. Add basic statistics

## Phase 6 — AI Discovery

25. Define `DiscoveryProvider`
26. Implement AI Discovery fallback
27. Measure fallback success
28. Compare deterministic Discovery vs AI Discovery

## Phase 7 — Runner expansion

29. Add more browser actions
30. Add more assertions
31. Improve locator strategies
32. Add semantic locators
33. Improve dynamic UI handling

## Phase 8 — Reporting

34. HTML report
35. JSON report
36. Screenshots
37. Failure evidence
38. Execution timeline

## Phase 9 — Web UI

39. Test Case list
40. Test Case editor
41. Run Test
42. Execution history
43. Statistics dashboard
44. Provider/settings management

---

# 14. Rough Timeline

Based on the current development pace:

### 2–3 weeks

A strong technical MVP:

```text
Discovery
+
LLM Router
+
Test Case model
+
Step decomposition
+
Test Plan cache
+
Playwright execution
+
basic history
+
basic reporting
```

### 4–6 weeks

A serious GitHub project:

```text
Everything above
+
AI Discovery fallback
+
versioned Test Plans
+
execution history
+
evidence
+
more browser actions
+
better locator strategies
+
provider abstraction
```

### 6–10 weeks

A substantially more complete product:

```text
Everything above
+
Web UI
+
statistics
+
advanced reporting
+
multiple Discovery providers
+
more robust recovery
+
broader web-testing capabilities
```

These are development estimates for the current individual-project pace, not guaranteed deadlines.

---

# 15. Product Strategy

The project should remain focused initially on:

> **Universal AI-assisted Web QA**

rather than trying to solve web, desktop, mobile, API and everything else simultaneously.

The core architecture should nevertheless make future expansion possible.

The central pipeline is:

```text
Natural Language
       |
       v
Task Analyzer
       |
       v
Website Discovery
       |
       v
Website Model
       |
       v
Test Case / Steps
       |
       v
Test Plan
       |
       v
Playwright Runner
       |
       v
Evidence
       |
       v
Execution History
       |
       v
Report
```

The long-term goal is not to make the LLM do everything.

The goal is to combine:

```text
Deterministic automation
+
Structured models
+
AI reasoning
+
Provider independence
+
Caching
+
Evidence
```

into a reliable QA system.