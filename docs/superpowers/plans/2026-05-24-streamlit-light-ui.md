# Streamlit Light UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the Streamlit demo from a pure-black console into a minimal, premium light interface inspired by OpenAI and Apple.

**Architecture:** Keep the current Streamlit layout and component functions intact. Replace the CSS design tokens and component styles in `CONSOLE_CSS`, with minimal markup changes only if needed for visual polish.

**Tech Stack:** Python, Streamlit, inline CSS, pytest.

---

### Task 1: Replace Dark Theme Tokens And Global Surfaces

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Update CSS variables**

In `CONSOLE_CSS`, replace the current `:root` color token block with:

```css
    :root {
        --bg:           #f7f7f5;
        --bg-elevated:  #ffffff;
        --bg-panel:     #ffffff;
        --bg-hover:     #f5f5f2;
        --bg-soft:      #f1f1ef;
        --border:       #deded8;
        --border-soft:  #ecece7;
        --border-panel: #e4e4df;
        --text:         #1f1f1f;
        --text-muted:   #6e6e73;
        --text-subtle:  #9a9a92;
        --accent:       #10a37f;
        --accent-soft:  #e7f5f0;
        --success:      #15803d;
        --warning:      #b7791f;
        --danger:       #c2410c;
        --shadow-soft:  0 18px 45px rgba(31, 31, 31, 0.06);
        --shadow-card:  0 1px 2px rgba(31, 31, 31, 0.04);
        --radius:       8px;
        --radius-lg:    10px;
    }
```

- [ ] **Step 2: Update global app and sidebar backgrounds**

Set `.stApp` to use a light radial background:

```css
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(16, 163, 127, 0.08), transparent 32rem),
            radial-gradient(circle at top left, rgba(31, 31, 31, 0.04), transparent 26rem),
            var(--bg);
        color: var(--text);
    }
```

Set `section[data-testid="stSidebar"]` to white with a subtle border:

```css
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.86);
        border-right: 1px solid var(--border-soft);
    }
```

### Task 2: Restyle Controls, Cards, And Panels

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Restyle buttons**

Use near-black primary buttons and light hover states:

```css
    .stButton > button {
        width: 100%;
        background: #1f1f1f;
        color: #ffffff;
        font-weight: 500;
        font-size: 0.9rem;
        border: 0;
        border-radius: var(--radius);
        height: 3rem;
        min-height: 3rem;
        box-sizing: border-box;
        box-shadow: var(--shadow-card);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: background 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }

    button[data-testid="stBaseButton-primary"] {
        background: #1f1f1f;
        color: #ffffff;
        border: 0;
        box-shadow: var(--shadow-card);
    }

    .stButton > button:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #333333;
        color: #ffffff;
        border-color: transparent;
        box-shadow: 0 8px 22px rgba(31, 31, 31, 0.12);
    }
```

- [ ] **Step 2: Restyle panels and cards**

Ensure `.panel`, `.answer-panel`, `.status-card`, `.empty-state`, expanders, and metrics use white surfaces, soft borders, and low shadows:

```css
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        box-shadow: var(--shadow-card);
```

- [ ] **Step 3: Restyle form fields**

Use light gray form backgrounds and green focus borders:

```css
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div {
        background: var(--bg-soft) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: var(--radius) !important;
        font-size: 0.9rem !important;
        transition: border-color 140ms ease, background 140ms ease;
    }

    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.12) !important;
        outline: none !important;
        background: #ffffff !important;
    }
```

### Task 3: Verify UI And Regression Tests

**Files:**
- Test: existing tests

- [ ] **Step 1: Run focused UI-related tests**

Run:

```bash
python -m pytest tests/test_streamlit_client.py tests/test_readme_doc.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Verify running app**

If the app is already running, reload `http://127.0.0.1:8501`. If not, start the services:

```bash
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Expected: the page uses a light background, white cards, dark primary button, and readable text.
