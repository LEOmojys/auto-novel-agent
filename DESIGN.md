# Auto Novel Prompt Design System

## 0. Console Entries

Two console frontends share the same `NovelAgentService` boundary:

- **Zero-dependency console (default)**: `python webui.py` (or `python api.py`) — serves `novel_agent/webui/static/` via the built-in `http.server` on `127.0.0.1:7860` (override with `--port`/`--host`). Multi-turn console mode: the user only picks a project and types the creative requirement — run identifiers are assigned automatically by `service.start_auto` (a retry on a terminal run reuses its id; a new requirement gets the next `{project}-run-{N}`), and the composer has no run-id field.
- **Gradio console (compatibility)**: `api_v1.py` — preserved for tooling/tests, keeps the familiar form controls including a manual run-id field.

The sections below describe the shared visual language; the zero-dependency console implements it in `novel_agent/webui/static/style.css`.

## 1. Atmosphere & Identity

This is a compact, work-focused control console for authors supervising a governed novel-generation run. It should feel calm, legible, and explicit about state: the defining signal is a durable status snapshot beside the commands that can change it. The UI preserves familiar form controls rather than introducing a visual redesign.

## 2. Color

| Role | Token | Value | Usage |
| --- | --- | --- | --- |
| Surface primary | `--surface-primary` | `#F8FAFC` | Page background |
| Surface secondary | `--surface-secondary` | `#FFFFFF` | Gradio groups and read panels |
| Text primary | `--text-primary` | `#172033` | Labels and run details |
| Text secondary | `--text-secondary` | `#526070` | Supporting metadata |
| Border default | `--border-default` | `#CBD5E1` | Deliberate group separation |
| Accent primary | `--accent-primary` | Native Gradio primary | Primary start action and focus |
| Accent hover | `--accent-hover` | Native Gradio hover | Hover feedback |
| Status success | `--status-success` | `#15803D` | Accepted or completed state |
| Status warning | `--status-warning` | `#A16207` | Review, pause, or queued state |
| Status error | `--status-error` | `#B91C1C` | Failure and cancel state |
| Status info | `--status-info` | `#1D4ED8` | Read-only audit information |

Use color only to identify interactive controls or genuine run state. Status text always includes a textual state name, so color is never the only signal.

## 3. Typography

| Level | Size | Weight | Line Height | Usage |
| --- | --- | --- | --- | --- |
| H1 | 28px | 700 | 1.3 | Console title |
| H2 | 20px | 600 | 1.4 | Gradio group headings |
| Body | 16px | 400 | 1.6 | Form fields and read panels |
| Body small | 14px | 400 | 1.5 | Metadata and help text |
| Caption | 12px | 500 | 1.4 | Hashes and revision metadata |

- Primary: `Microsoft YaHei, PingFang SC, Noto Sans CJK SC, system-ui, sans-serif`
- Mono: `Cascadia Mono, Consolas, monospace`
- Do not rely on line-clamped labels. Chinese labels may wrap at narrow widths without hiding meaning.

## 4. Spacing & Layout

All spacing follows a 4px base: `--space-1` 4px, `--space-2` 8px, `--space-3` 12px, `--space-4` 16px, `--space-5` 20px, `--space-6` 24px, and `--space-8` 32px.

The console uses one document scroll owner. Its primary layout is a two-column Gradio row that becomes a single vertical column at widths below 640px. Controls form a wrapping cluster; long hashes, JSON, and artifact text use a dedicated read panel with wrapping rather than forcing horizontal page scroll.

## 5. Components

### Run Control Form

- Structure: project selector, requirements input, start command. The zero-dependency console omits the run identifier (auto-managed); the Gradio entry keeps a manual run-id field for compatibility.
- States: default, validation error, pending response, queued confirmation.
- Accessibility: visible labels, keyboard order follows author workflow, errors remain adjacent to the submitted command.

### Run Command Cluster

- Structure: pause, resume, cancel, and refresh buttons.
- States: enabled only when the action is legal; command result is rendered as a persisted snapshot, not an optimistic flag.
- Accessibility: each action has a text label and deterministic tab order.

### Status Snapshot

- Structure: one read-only JSON status panel and one concise status message.
- States: empty before a run is selected, normal snapshot, review-required, and error.
- Accessibility: state value and review reason are text, not color-only badges.

### Governed Data Viewer

- Structure: artifact revision fields, artifact text panel, audit and quality report panel, rebase confirmation input.
- States: empty, loading, available, missing artifact, validation error, and rebase confirmation result.
- Accessibility: long content remains selectable and wraps; destructive rebase requires the typed confirmation payload.

## 6. Motion & Interaction

Motion intensity is 1. No decorative animation is used. Native button hover, focus, and active feedback remain available through Gradio. The controls provide clear command results and persistent snapshots instead of transient animation.

## 7. Depth & Surface

Strategy: mixed. The control plane uses modest borders and native Gradio grouping to distinguish input, command, and read-only regions. No nested custom cards, glass effects, decorative gradients, or ornamental visual elements are added.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- Target WCAG 2.2 AA: text contrast at least 4.5:1, visible keyboard focus, and complete keyboard access to every command.
- Preserve labels above inputs. Do not use placeholder text as a label.
- Respect user zoom and CJK line wrapping at 200 percent zoom and at a 375px viewport.
- Every destructive or state-changing action displays its persisted outcome or a specific validation error.

### Accepted Debt

| Item | Location | Why accepted | Owner / Exit |
| --- | --- | --- | --- |
| Native Gradio component CSS is not fully tokenized | `api_v1.py` Blocks tree | The Gradio entry uses server-rendered vendor components; overriding vendor styles would create unrelated visual risk. The zero-dependency console (`novel_agent/webui/static/`) owns its full stylesheet and is the styling reference going forward. | Replace only if the product moves to an owned frontend. |
| No automatic run progress animation | Run status snapshot | The worker-stage aggregator is intentionally not represented as a fake progress meter. | Add only with persisted stage events and user-visible timing semantics. |
