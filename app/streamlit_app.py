from __future__ import annotations

from html import escape
import os
from pathlib import Path
import sys
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api_client import (
    ApiClientError,
    ask_api,
    get_knowledge_status_api,
    list_knowledge_documents_api,
    reindex_knowledge_api,
    run_eval_api,
    upload_knowledge_document_api,
)


TASK_OPTIONS = {
    "自动判断": None,
    "文档问答": "document_qa",
    "时序查询": "timeseries_query",
    "异常诊断": "anomaly_diagnosis",
    "策略建议": "policy_recommendation",
}

WORKFLOW_OPTIONS = {
    "LangGraph workflow": "langgraph",
    "Deterministic baseline": "deterministic",
}

DEMO_WALKTHROUGHS = [
    {
        "label": "BEAR 数据边界",
        "task_type": "document_qa",
        "question": "BEAR 轨迹在本项目中应如何表述，为什么不能说成真实生产数据？",
        "why": "展示 RAG 引用、数据边界和防幻觉约束。",
    },
    {
        "label": "温度时序查询",
        "task_type": "timeseries_query",
        "question": "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？",
        "why": "展示 router 如何选择时序工具并返回结构化 metric summary。",
    },
    {
        "label": "策略建议边界",
        "task_type": "policy_recommendation",
        "question": "如果当前温度超过舒适上限，是否应该调整控制策略？",
        "why": "展示控制建议来自 policy 工具，LLM 只负责解释。",
    },
]

METRIC_GROUPS = {
    "Retrieval": ["citation_hit_rate", "context_recall"],
    "Answer": ["expected_keyword_coverage", "lexical_answer_coverage"],
    "Tool": [
        "tool_selection_accuracy",
        "tool_execution_success_rate",
        "evidence_coverage",
    ],
    "Planner": [
        "planned_step_accuracy",
        "planned_step_order_accuracy",
        "required_step_recall",
        "policy_final_step_rate",
    ],
    "Quality Proxy": ["answer_correctness_proxy", "faithfulness_proxy"],
}


def get_default_api_base_url() -> str:
    return os.environ.get("HVAC_COPILOT_API_BASE_URL", "http://localhost:8000")

DASHBOARD_COPY = {
    "title": "DataCenter-HVAC Copilot",
    "subtitle": "RAG + Tool Agent for BEAR HVAC simulation evidence.",
    "boundary": "BEAR 仅作为 HVAC 仿真 / 可控代理场景。LLM 不直接控制环境，策略动作只来自 policy tool。",
}

CONSOLE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600&family=Noto+Sans+SC:wght@400;500;600&family=Noto+Serif+SC:wght@500;600&family=Playfair+Display:wght@500;600&family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

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
        --mouse-x:      68%;
        --mouse-y:      12%;
        --font-sans:    'Inter', 'Noto Sans SC', 'PingFang SC', 'HarmonyOS Sans SC',
                         'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
        --font-display: 'Playfair Display', 'Noto Serif SC', Georgia, serif;
        --font-accent:  'Noto Serif SC', 'Songti SC', 'SimSun', serif;
        --font-numeric: 'Space Grotesk', 'Inter', 'Noto Sans SC', sans-serif;
        --font-mono:    'JetBrains Mono', 'SF Mono', Consolas, monospace;
        --shadow-soft:  0 18px 45px rgba(31, 31, 31, 0.06);
        --shadow-card:  0 1px 2px rgba(31, 31, 31, 0.04);
        --radius:       8px;
        --radius-lg:    10px;
    }

    html, body, [class*="css"], .stApp, .stApp * {
        font-family: var(--font-sans) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: geometricPrecision;
    }

    code, pre, .stCode, [data-testid="stCode"] {
        font-family: var(--font-mono) !important;
    }

    .stApp {
        background:
            radial-gradient(circle at var(--mouse-x) var(--mouse-y), rgba(16, 163, 127, 0.10), transparent 24rem),
            radial-gradient(circle at top left, rgba(31, 31, 31, 0.035), transparent 26rem),
            var(--bg);
        color: var(--text);
        isolation: isolate;
        overflow-x: hidden;
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: -1;
    }

    .stApp::before {
        background:
            radial-gradient(ellipse at calc(var(--mouse-x) + 4%) calc(var(--mouse-y) + 7%), rgba(16, 163, 127, 0.12), transparent 18rem),
            radial-gradient(ellipse at 18% 88%, rgba(31, 31, 31, 0.035), transparent 22rem);
        filter: blur(2px);
        opacity: 0.72;
        animation: ambient-breathe 11s ease-in-out infinite;
    }

    .stApp::after {
        background:
            linear-gradient(118deg, transparent 0 18%, rgba(16, 163, 127, 0.11) 18.4%, transparent 18.9% 100%),
            linear-gradient(142deg, transparent 0 61%, rgba(31, 31, 31, 0.055) 61.35%, transparent 61.8% 100%),
            linear-gradient(32deg, transparent 0 70%, rgba(16, 163, 127, 0.07) 70.2%, transparent 70.55% 100%);
        opacity: 0.42;
        transform: translate3d(0, 0, 0);
        animation: ambient-breathe 14s ease-in-out infinite reverse;
    }

    @keyframes ambient-breathe {
        0%, 100% {
            opacity: 0.38;
            transform: translate3d(0, 0, 0) scale(1);
        }
        50% {
            opacity: 0.68;
            transform: translate3d(0.45rem, -0.35rem, 0) scale(1.015);
        }
    }

    @media (prefers-reduced-motion: reduce) {
        .stApp::before,
        .stApp::after {
            animation: none;
        }
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.86);
        border-right: 1px solid var(--border-soft);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-muted);
    }

    .sidebar-shell {
        padding: 0.35rem 0 1rem;
    }

    .sidebar-kicker {
        color: var(--text-subtle);
        font-family: var(--font-mono) !important;
        font-size: 0.68rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }

    .sidebar-title {
        color: var(--text);
        font-family: var(--font-accent) !important;
        font-size: 1.22rem;
        font-weight: 560;
        letter-spacing: -0.025em;
        margin-bottom: 0.35rem;
    }

    .sidebar-subtitle {
        color: var(--text-muted);
        font-size: 0.82rem;
        line-height: 1.58;
        margin-bottom: 1rem;
    }

    .sidebar-config-group {
        background: rgba(255, 255, 255, 0.76);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius);
        box-shadow: var(--shadow-card);
        margin: 0.72rem 0;
        padding: 0.82rem 0.86rem;
    }

    .sidebar-config-title {
        color: var(--text);
        font-family: var(--font-accent) !important;
        font-size: 0.88rem;
        font-weight: 560;
        letter-spacing: -0.018em;
        margin-bottom: 0.64rem;
    }

    .config-row {
        display: grid;
        grid-template-columns: minmax(5.3rem, 0.8fr) minmax(0, 1.2fr);
        align-items: baseline;
        gap: 0.55rem;
        padding: 0.28rem 0;
    }

    .config-label {
        color: var(--text-subtle);
        font-family: var(--font-mono) !important;
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 0.035em;
    }

    .config-value {
        color: var(--text);
        font-family: var(--font-numeric) !important;
        font-size: 0.78rem;
        font-weight: 500;
        line-height: 1.35;
        overflow-wrap: anywhere;
        text-align: right;
    }

    .config-dot {
        display: inline-block;
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 999px;
        margin-right: 0.35rem;
        transform: translateY(-0.04rem);
        background: var(--text-subtle);
    }

    .config-dot.success {
        background: var(--accent);
    }

    .config-dot.warning {
        background: var(--warning);
    }

    .config-dot.neutral {
        background: var(--text-subtle);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 0.35rem;
        padding-bottom: 4rem;
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, h4, h5 {
        font-family: var(--font-accent) !important;
        color: var(--text);
        font-weight: 560;
        letter-spacing: -0.018em;
    }

    p, span, label, div {
        letter-spacing: 0;
    }

    p, .stMarkdown, .stText, label {
        line-height: 1.68;
    }

    /* Tabs — minimal underline only */
    div[data-testid="stTabs"] {
        border-bottom: 1px solid var(--border-soft);
        margin-bottom: 1.75rem;
    }

    div[data-testid="stTabs"] button {
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.9rem;
        line-height: 1.45;
        background: transparent;
        border-bottom: 1px solid transparent;
        transition: color 120ms ease, border-color 120ms ease;
    }

    div[data-testid="stTabs"] button:hover {
        color: var(--text);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text);
        border-bottom-color: var(--accent);
    }

    .stButton {
        margin-bottom: 1.1rem;
    }

    /* Buttons — flat, no gradient */
    .stButton > button {
        width: 100%;
        background: rgba(255, 255, 255, 0.72);
        color: var(--text);
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: -0.003em;
        border: 0;
        border-radius: var(--radius);
        height: 3rem;
        min-height: 3rem;
        box-sizing: border-box;
        box-shadow: inset 0 0 0 1px var(--border-panel), var(--shadow-card);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: background 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }

    button[data-testid="stBaseButton-primary"] {
        background: rgba(255, 255, 255, 0.72);
        color: var(--text);
        border: 0;
        box-shadow: inset 0 0 0 1px var(--border-panel), var(--shadow-card);
    }

    .stButton > button:hover {
        background: var(--accent-soft);
        color: #0f6f58;
        border-color: transparent;
        box-shadow: inset 0 0 0 1px rgba(16, 163, 127, 0.32), 0 8px 22px rgba(16, 163, 127, 0.10);
    }

    button[data-testid="stBaseButton-primary"]:hover {
        background: var(--accent-soft);
        color: #0f6f58;
        border-color: transparent;
        box-shadow: inset 0 0 0 1px rgba(16, 163, 127, 0.32), 0 8px 22px rgba(16, 163, 127, 0.10);
    }

    .stButton > button:active {
        background: #dff2eb;
        color: #0f6f58;
    }

    /* Form inputs */
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

    /* Hero */
    .console-hero {
        margin-bottom: 1.15rem;
        padding: 0;
    }

    .console-kicker {
        color: var(--text-subtle);
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        margin-bottom: 0.48rem;
        font-family: var(--font-mono) !important;
    }

    .console-title {
        font-family: var(--font-display) !important;
        color: var(--text);
        font-size: clamp(1.95rem, 3.1vw, 2.75rem);
        line-height: 0.98;
        font-weight: 560;
        letter-spacing: -0.035em;
        margin: 0 0 0.52rem 0;
    }

    .console-subtitle {
        color: var(--text-muted);
        font-size: 0.96rem;
        line-height: 1.5;
        margin: 0;
        max-width: 720px;
        font-weight: 400;
    }

    /* Boundary notice — quiet, not loud */
    .boundary-strip {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        color: var(--text-muted);
        padding: 0.56rem 0.82rem;
        border-radius: var(--radius);
        margin-bottom: 1.35rem;
        font-size: 0.85rem;
        line-height: 1.5;
        box-shadow: var(--shadow-card);
    }

    .boundary-strip::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--warning);
        flex-shrink: 0;
    }

    /* Panels — borderless minimalism */
    .panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-card);
    }

    .panel-title {
        font-family: var(--font-accent) !important;
        color: var(--text);
        font-weight: 560;
        font-size: 1rem;
        margin: -0.1rem 0 1.25rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid var(--border-soft);
        letter-spacing: -0.02em;
    }

    /* Answer panel — quiet, content-first */
    .answer-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius-lg);
        padding: 1.75rem 2rem;
        margin-bottom: 1.75rem;
        box-shadow: var(--shadow-card);
    }

    .answer-panel h3 {
        color: var(--text-muted);
        margin: 0 0 1.25rem 0;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        font-family: var(--font-mono) !important;
    }

    .answer-panel > div {
        color: var(--text);
        font-size: 0.95rem;
        line-height: 1.72;
    }

    /* Status grid — three info chips */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
        margin: 0 0 1.75rem 0;
    }

    .status-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius);
        padding: 1rem 1.15rem;
        min-height: 92px;
        transition: border-color 120ms ease;
        box-shadow: var(--shadow-card);
    }

    .status-card:hover {
        border-color: var(--accent);
    }

    .status-card .label {
        color: var(--text-subtle);
        font-size: 0.72rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-family: var(--font-mono) !important;
    }

    .status-card .value {
        font-family: var(--font-numeric) !important;
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 560;
        line-height: 1.35;
        overflow-wrap: anywhere;
        margin-bottom: 0.4rem;
    }

    .status-card .hint {
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.45;
        overflow-wrap: anywhere;
    }

    /* Section label */
    .section-label {
        color: var(--text-subtle);
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        margin: 1.75rem 0 0.85rem;
        font-family: var(--font-mono) !important;
    }

    /* Expander — quieter */
    div[data-testid="stExpander"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius-lg);
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stExpander"] summary {
        color: var(--text);
        font-weight: 560;
        font-size: 0.9rem;
    }

    div[data-testid="stExpander"] summary:hover {
        color: var(--text);
    }

    /* DataFrame — flat, no gridlines */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-panel);
        border-radius: var(--radius);
        overflow: hidden;
    }

    div[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
        background: var(--bg-panel);
    }

    /* Inline code */
    code {
        color: var(--text);
        background: var(--accent-soft);
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-size: 0.85em;
    }

    /* Streamlit metric — clean */
    div[data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius);
        padding: 0.85rem 1rem;
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-subtle);
        font-size: 0.72rem;
        font-weight: 500;
        font-family: var(--font-mono) !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: var(--font-numeric) !important;
        color: var(--text);
        font-weight: 500;
    }

    /* Caption */
    .stCaption, [data-testid="stCaption"] {
        color: var(--text-muted) !important;
        font-size: 0.82rem !important;
    }

    /* Alert / info / warning */
    div[data-baseweb="notification"] {
        border-radius: var(--radius);
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        box-shadow: var(--shadow-card);
    }

    div[data-testid="stStatusWidget"],
    div[data-testid="stToolbar"],
    div[data-testid="stMainMenu"],
    div[data-testid="stActionButton"] {
        display: none !important;
    }

    /* Sidebar text input */
    section[data-testid="stSidebar"] input {
        background: var(--bg-soft) !important;
        border: 1px solid var(--border-soft) !important;
    }

    /* Empty state — calmer */
    .empty-state {
        background: var(--bg-panel);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius-lg);
        padding: 2rem 2rem;
        margin-bottom: 1.25rem;
        text-align: left;
        box-shadow: var(--shadow-card);
    }

    .empty-state .empty-kicker {
        color: var(--text-subtle);
        font-size: 0.72rem;
        font-weight: 500;
        margin-bottom: 0.65rem;
        letter-spacing: 0.04em;
        font-family: var(--font-mono) !important;
    }

    .empty-state .empty-title {
        font-family: var(--font-accent) !important;
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 560;
        margin-bottom: 0.75rem;
    }

    .empty-state .empty-body {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    /* Model running indicator */
    .thinking-panel {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--border-panel);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-card);
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.25rem;
        padding: 1.2rem 1.35rem;
    }

    .thinking-orbit {
        position: relative;
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        border: 1.5px solid rgba(16, 163, 127, 0.14);
        border-top-color: var(--accent);
        border-right-color: rgba(16, 163, 127, 0.42);
        box-shadow:
            0 0 0 0.34rem rgba(16, 163, 127, 0.055),
            0 0 1.2rem rgba(16, 163, 127, 0.12),
            inset 0 0 0 0.42rem rgba(16, 163, 127, 0.045);
        flex: 0 0 auto;
        animation:
            thinking-spin 1.08s cubic-bezier(0.52, 0.16, 0.28, 0.92) infinite,
            thinking-pulse 2.4s ease-in-out infinite;
    }

    .thinking-orbit::after {
        content: "";
        position: absolute;
        inset: 0.48rem;
        border-radius: 999px;
        background:
            radial-gradient(circle at 34% 30%, rgba(255, 255, 255, 0.95), rgba(16, 163, 127, 0.16) 54%, transparent 62%);
        box-shadow: 0 0 0.8rem rgba(16, 163, 127, 0.16);
        animation: thinking-pulse 2.4s ease-in-out infinite reverse;
    }

    .thinking-copy {
        min-width: 0;
    }

    .thinking-title {
        color: var(--text);
        font-family: var(--font-accent) !important;
        font-size: 1rem;
        font-weight: 560;
        letter-spacing: -0.018em;
        margin-bottom: 0.18rem;
    }

    .thinking-body {
        color: var(--text-muted);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    @keyframes thinking-spin {
        to {
            transform: rotate(360deg);
        }
    }

    @keyframes thinking-pulse {
        0%, 100% {
            opacity: 0.72;
            filter: saturate(0.92);
        }
        50% {
            opacity: 1;
            filter: saturate(1.12);
        }
    }

    /* Mobile */
    @media (max-width: 900px) {
        .status-grid {
            grid-template-columns: 1fr;
        }
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.5rem;
        }
        .answer-panel,
        .panel {
            padding: 1.25rem;
        }
    }
</style>
"""

MOUSE_GLOW_SCRIPT = """
<script>
(() => {
    try {
        const targetDocument = window.parent?.document || document;
        let mouseX = 68;
        let mouseY = 12;
        let frameRequested = false;

        const applyGlowPosition = () => {
            const app = targetDocument.querySelector(".stApp");
            if (app) {
                app.style.setProperty("--mouse-x", `${mouseX}%`);
                app.style.setProperty("--mouse-y", `${mouseY}%`);
            }
            frameRequested = false;
        };

        const handleMouseMove = (event) => {
            const width = targetDocument.documentElement.clientWidth || window.innerWidth || 1;
            const height = targetDocument.documentElement.clientHeight || window.innerHeight || 1;
            mouseX = Math.max(0, Math.min(100, (event.clientX / width) * 100));
            mouseY = Math.max(0, Math.min(100, (event.clientY / height) * 100));
            if (!frameRequested) {
                window.requestAnimationFrame(applyGlowPosition);
                frameRequested = true;
            }
        };

        targetDocument.addEventListener("mousemove", handleMouseMove, { passive: true });
        applyGlowPosition();
    } catch (error) {
        // Static CSS defaults remain active when iframe access is restricted.
    }
})();
</script>
"""


def get_dashboard_copy() -> dict[str, str]:
    return DASHBOARD_COPY.copy()


def build_status_cards(result: dict) -> list[dict[str, str]]:
    tools = result.get("tools", [])
    citations = result.get("citations", [])
    tool_results = result.get("tool_results", [])
    audit = result.get("answer_audit") or {}
    data_source = result.get("data_source") or {}
    return [
        {
            "label": "Route",
            "value": str(result.get("route", "unknown")),
            "hint": str(result.get("route_reason") or "deterministic router"),
        },
        {
            "label": "Policy / Tools",
            "value": ", ".join(tools) if tools else "none",
            "hint": "工具输出动作，LLM 只解释证据",
        },
        {
            "label": "Generator",
            "value": str(result.get("answer_generator", "unknown")),
            "hint": "evidence-grounded final answer",
        },
        {
            "label": "Evidence",
            "value": f"{len(citations)} citations / {len(tool_results)} tool results",
            "hint": "引用与结构化工具结果",
        },
        {
            "label": "Audit",
            "value": "passed" if audit.get("passed") else "review",
            "hint": ", ".join(audit.get("violations", [])) or "boundary checks clear",
        },
        {
            "label": "Data Source",
            "value": str(data_source.get("kind", "unknown")),
            "hint": str(data_source.get("path", "")),
        },
    ]


def build_sidebar_config_groups(
    *,
    api_base_url: str,
    workflow_label: str,
    last_result: dict | None = None,
) -> list[dict[str, object]]:
    result = last_result or {}
    workflow_trace = result.get("workflow_trace") or []
    planner = _extract_planner_name(workflow_trace)
    generator = str(result.get("answer_generator") or "deterministic / optional LLM")
    data_source = result.get("data_source") or {}
    data_kind = str(data_source.get("kind") or "processed_csv / fallback")
    tools = result.get("tools") or []
    policy_value = _summarize_policy_backend(tools)

    return [
        {
            "title": "Connection",
            "items": [
                {
                    "label": "Endpoint",
                    "value": _compact_endpoint(api_base_url),
                    "tone": "neutral",
                },
                {"label": "Service", "value": "hvac-copilot", "tone": "success"},
            ],
        },
        {
            "title": "Model Runtime",
            "items": [
                {"label": "Workflow", "value": workflow_label, "tone": "success"},
                {"label": "Planner", "value": planner, "tone": _runtime_tone(planner)},
                {"label": "Generator", "value": generator, "tone": _runtime_tone(generator)},
            ],
        },
        {
            "title": "Data & Policy",
            "items": [
                {"label": "Data", "value": data_kind, "tone": "success"},
                {"label": "Policy", "value": policy_value, "tone": _runtime_tone(policy_value)},
                {"label": "Boundary", "value": "LLM explains only", "tone": "warning"},
            ],
        },
        {
            "title": "Evaluation",
            "items": [
                {"label": "Eval Set", "value": "hvac_eval.jsonl", "tone": "neutral"},
                {"label": "Metrics", "value": "deterministic proxy", "tone": "neutral"},
                {"label": "Judge", "value": "optional", "tone": "warning"},
            ],
        },
    ]


def render_sidebar_config_group_html(group: dict[str, object]) -> str:
    title = escape(str(group["title"]))
    rows = []
    for item in group["items"]:  # type: ignore[index]
        item_dict = item if isinstance(item, dict) else {}
        label = escape(str(item_dict.get("label", "")))
        value = escape(str(item_dict.get("value", "")))
        tone = escape(str(item_dict.get("tone", "neutral")))
        rows.append(
            '<div class="config-row">'
            f'<div class="config-label">{label}</div>'
            f'<div class="config-value"><span class="config-dot {tone}"></span>{value}</div>'
            "</div>"
        )
    return (
        '<div class="sidebar-config-group">'
        f'<div class="sidebar-config-title">{title}</div>'
        + "".join(rows)
        + "</div>"
    )


def _compact_endpoint(api_base_url: str) -> str:
    parsed = urlparse(api_base_url)
    if parsed.netloc:
        return parsed.netloc
    return api_base_url.replace("http://", "").replace("https://", "").rstrip("/")


def _extract_planner_name(workflow_trace: list[dict]) -> str:
    for item in workflow_trace:
        planner = item.get("planner")
        if planner:
            return str(planner)
    return "deterministic / optional LLM"


def _summarize_policy_backend(tools: list[str]) -> str:
    if any("dropt" in str(tool).lower() or "diff" in str(tool).lower() for tool in tools):
        return "DROPT checkpoint"
    if any("policy" in str(tool).lower() for tool in tools):
        return "policy tool"
    return "rule-based fallback"


def _runtime_tone(value: str) -> str:
    lowered = value.lower()
    if "fallback" in lowered or "optional" in lowered or "deterministic" in lowered:
        return "warning"
    return "success"


def group_eval_metrics(metrics: dict) -> dict[str, list[tuple[str, float]]]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for group_name, metric_names in METRIC_GROUPS.items():
        values = []
        for metric_name in metric_names:
            if metric_name in metrics:
                values.append((metric_name, float(metrics[metric_name])))
        if values:
            grouped[group_name] = values
    return grouped


def build_prediction_preview(predictions: list[dict]) -> list[dict]:
    preview = []
    for prediction in predictions:
        citations = prediction.get("citations", [])
        tool_results = prediction.get("tool_results", [])
        answer_audit = prediction.get("answer_audit") or {}
        answer = str(prediction.get("answer") or "")
        preview.append(
            {
                "id": prediction.get("id"),
                "task_type": prediction.get("task_type"),
                "route": prediction.get("route"),
                "tools": ", ".join(prediction.get("tools", [])),
                "citation_count": len(citations),
                "tool_result_count": len(tool_results),
                "has_citation": bool(citations),
                "has_tool_result": bool(tool_results),
                "answer_length": len(answer),
                "audit_passed": answer_audit.get("passed"),
                "audit_violations": ", ".join(answer_audit.get("violations", [])),
            }
        )
    return preview


def build_execution_timeline(result: dict) -> list[dict]:
    citations = result.get("citations", [])
    contexts = result.get("retrieved_contexts", [])
    tools = result.get("tools", [])
    tool_results = result.get("tool_results", [])
    data_source = result.get("data_source", {})
    return [
        {
            "stage": "Route",
            "status": result.get("route", "unknown"),
            "detail": result.get("route_reason") or "No route reason returned.",
        },
        {
            "stage": "Retrieval",
            "status": f"{len(citations)} citations / {len(contexts)} contexts",
            "detail": _summarize_citations(citations),
        },
        {
            "stage": "Tool Call",
            "status": f"{len(tool_results)} results",
            "detail": ", ".join(tools) if tools else "No tool call required.",
        },
        {
            "stage": "Answer Generator",
            "status": result.get("answer_generator", "unknown"),
            "detail": f"Final answer generated by {result.get('answer_generator', 'unknown')}.",
        },
        {
            "stage": "Data Boundary",
            "status": data_source.get("kind", "unknown"),
            "detail": (
                f"{data_source.get('path', '')} | 当前轨迹仅用于 HVAC 仿真 / 可控代理场景，"
                "不能表述为真实数据中心生产遥测。"
            ),
        },
    ]


def build_workflow_trace_rows(result: dict) -> list[dict]:
    rows = []
    for index, item in enumerate(result.get("workflow_trace", []), start=1):
        tools = item.get("tools") or _planned_tools(item)
        citation_count = int(item.get("citation_count", 0) or 0)
        tool_result_count = int(item.get("tool_result_count", 0) or 0)
        audit_value = item.get("audit_passed", item.get("passed"))
        rows.append(
            {
                "step": index,
                "node": item.get("node", "unknown"),
                "route": item.get("route", result.get("route", "unknown")),
                "classifier": str(item.get("classifier", item.get("planner", "n/a"))),
                "fallback": _format_fallback_status(item.get("fallback_used")),
                "tools": ", ".join(str(tool) for tool in tools) if tools else "none",
                "evidence": f"{citation_count} citations / {tool_result_count} tool results",
                "audit": _format_audit_status(audit_value),
            }
        )
    return rows


def _planned_tools(item: dict) -> list[str]:
    specs = item.get("planned_step_specs") or []
    tools = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        tool = spec.get("tool")
        if tool:
            tools.append(str(tool))
    return tools


def _format_audit_status(value: object) -> str:
    if value is True:
        return "passed"
    if value is False:
        return "review"
    return "n/a"


def _format_fallback_status(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def build_safety_audit_rows(result: dict) -> list[dict]:
    audit = result.get("answer_audit") or {}
    violations = set(audit.get("violations", []))
    checks = audit.get("checks", [])
    return [
        {
            "check": check,
            "status": "violation" if check in violations else "passed",
        }
        for check in checks
    ]


def _summarize_citations(citations: list[dict]) -> str:
    if not citations:
        return "No document citation returned."
    source_ids = [str(citation.get("source_id", "unknown")) for citation in citations[:3]]
    return ", ".join(source_ids)


def _render_console_shell() -> None:
    copy = get_dashboard_copy()
    st.markdown(CONSOLE_CSS, unsafe_allow_html=True)
    components.html(MOUSE_GLOW_SCRIPT, height=0, width=0)
    st.markdown(
        f"""
        <div class="console-hero">
            <div class="console-kicker">v0.1 · BEAR HVAC Simulation</div>
            <h1 class="console-title">{escape(copy["title"])}</h1>
            <p class="console-subtitle">{escape(copy["subtitle"])}</p>
        </div>
        <div class="boundary-strip">{escape(copy["boundary"])}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(last_result: dict | None = None) -> tuple[str, str]:
    st.sidebar.markdown(
        """
        <div class="sidebar-shell">
            <div class="sidebar-kicker">SYSTEM SETUP</div>
            <div class="sidebar-title">Runtime Console</div>
            <div class="sidebar-subtitle">Configure the demo endpoint and inspect the active model, data, policy, and evaluation profile.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    api_base_url = st.sidebar.text_input(
        "API Endpoint",
        value=get_default_api_base_url(),
        help="FastAPI service used by the Streamlit demo.",
    )
    workflow_label = st.sidebar.selectbox(
        "Workflow Engine",
        list(WORKFLOW_OPTIONS.keys()),
        index=0,
        help="LangGraph is the default orchestrated path; deterministic remains available for comparison.",
    )
    groups = build_sidebar_config_groups(
        api_base_url=api_base_url,
        workflow_label=workflow_label,
        last_result=last_result,
    )
    for group in groups:
        st.sidebar.markdown(render_sidebar_config_group_html(group), unsafe_allow_html=True)
    return api_base_url, workflow_label


def _render_panel_start(title: str) -> None:
    st.markdown(
        f'<div class="panel-title">{escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _render_status_grid(cards: list[dict[str, str]]) -> None:
    st.markdown(render_status_grid_html(cards), unsafe_allow_html=True)


def render_status_grid_html(cards: list[dict[str, str]]) -> str:
    card_html = []
    for card in cards:
        card_html.append(
            '<div class="status-card">'
            f'<div class="label">{escape(card["label"])}</div>'
            f'<div class="value">{escape(card["value"])}</div>'
            f'<div class="hint">{escape(card["hint"])}</div>'
            "</div>"
        )
    return '<div class="status-grid">' + "".join(card_html) + "</div>"


def render_thinking_indicator_html(title: str = "Analyzing evidence") -> str:
    return (
        '<div class="thinking-panel">'
        '<div class="thinking-orbit" aria-hidden="true"></div>'
        '<div class="thinking-copy">'
        f'<div class="thinking-title">{escape(title)}</div>'
        '<div class="thinking-body">模型正在组织证据、调用工具并生成 grounded answer。</div>'
        "</div>"
        "</div>"
    )


def _render_section_label(label: str) -> None:
    st.markdown(
        f'<div class="section-label">{escape(label)}</div>',
        unsafe_allow_html=True,
    )


def _render_answer_panel(answer: str) -> None:
    st.markdown(
        f"""
        <div class="answer-panel">
            <h3>Grounded Answer</h3>
            <div>{escape(answer).replace(chr(10), "<br>")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="DataCenter-HVAC Copilot", layout="wide")
    _render_console_shell()

    last_result = st.session_state.get("last_result")
    api_base_url, workflow_label = _render_sidebar(last_result)
    tab_ask, tab_knowledge, tab_eval = st.tabs(["Copilot", "Knowledge Base", "评测摘要"])
    with tab_ask:
        _render_ask_tab(api_base_url, workflow_label)
    with tab_knowledge:
        _render_knowledge_base_tab(api_base_url)
    with tab_eval:
        _render_eval_tab(api_base_url)


def _render_ask_tab(api_base_url: str, workflow_label: str) -> None:
    control_col, result_col = st.columns([0.34, 0.66], gap="large")
    with control_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        _render_panel_start("Mission Control")
        walkthrough_labels = ["自定义问题"] + [case["label"] for case in DEMO_WALKTHROUGHS]
        walkthrough_label = st.selectbox("典型案例", walkthrough_labels)
        selected_case = next(
            (case for case in DEMO_WALKTHROUGHS if case["label"] == walkthrough_label),
            None,
        )
        if selected_case:
            st.caption(selected_case["why"])

        default_task_type = selected_case["task_type"] if selected_case else None
        default_task_label = next(
            label for label, value in TASK_OPTIONS.items() if value == default_task_type
        )
        task_label = st.selectbox(
            "任务类型",
            list(TASK_OPTIONS.keys()),
            index=list(TASK_OPTIONS.keys()).index(default_task_label),
        )
        question = st.text_area(
            "问题",
            value=(
                selected_case["question"]
                if selected_case
                else "episode_001 中 zone_a 在最近 3 小时的温度最大值是多少？"
            ),
            height=210,
        )
        run_clicked = st.button("运行分析", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        if not run_clicked:
            _render_empty_state()
            return
        if not question.strip():
            st.warning("请输入问题。")
            return
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            render_thinking_indicator_html("Analyzing evidence"),
            unsafe_allow_html=True,
        )
        try:
            result = ask_api(
                api_base_url=api_base_url,
                question=question.strip(),
                task_type=TASK_OPTIONS[task_label],
                workflow_engine=WORKFLOW_OPTIONS[workflow_label],
                session_id=st.session_state.get("session_id"),
                memory_enabled=True,
            )
        except ApiClientError as exc:
            thinking_placeholder.empty()
            st.error(str(exc))
            return

        st.session_state["last_result"] = result
        if result.get("session_id"):
            st.session_state["session_id"] = result["session_id"]
        thinking_placeholder.empty()
        _render_result(result)


def _render_eval_tab(api_base_url: str) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _render_panel_start("Evaluation Bench")
    eval_path = st.text_input("评测集路径", value="data/eval/hvac_eval.jsonl")
    if st.button("运行评测", type="primary"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown(
            render_thinking_indicator_html("Running evaluation"),
            unsafe_allow_html=True,
        )
        try:
            result = run_eval_api(api_base_url=api_base_url, eval_path=eval_path)
        except ApiClientError as exc:
            thinking_placeholder.empty()
            st.error(str(exc))
            return
        thinking_placeholder.empty()
        _render_eval_result(result)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_knowledge_base_tab(api_base_url: str) -> None:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _render_panel_start("Knowledge Base")
    uploaded = st.file_uploader(
        "Upload SOP, manual, or operation guide",
        type=["md", "txt", "pdf", "docx"],
    )
    if uploaded is not None and st.button("Index document", type="primary"):
        try:
            result = upload_knowledge_document_api(
                api_base_url,
                uploaded.name,
                uploaded.getvalue(),
            )
            st.success(f"Indexed {result['document']['filename']}")
            st.json(result.get("index_status", {}))
            st.session_state["knowledge_status"] = get_knowledge_status_api(api_base_url)
        except ApiClientError as exc:
            st.error(str(exc))

    columns = st.columns(2)
    with columns[0]:
        if st.button("Refresh knowledge status"):
            try:
                st.session_state["knowledge_status"] = get_knowledge_status_api(api_base_url)
            except ApiClientError as exc:
                st.warning(str(exc))
    with columns[1]:
        if st.button("Rebuild FAISS index"):
            try:
                st.session_state["knowledge_status"] = reindex_knowledge_api(api_base_url)
            except ApiClientError as exc:
                st.warning(str(exc))

    try:
        status = st.session_state.get("knowledge_status") or get_knowledge_status_api(api_base_url)
        documents = list_knowledge_documents_api(api_base_url)["documents"]
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.metric("Documents", status.get("document_count", 0))
        with metric_columns[1]:
            st.metric("Chunks", status.get("chunk_count", 0))
        st.json(status.get("index", {}))
        if documents:
            st.dataframe(pd.DataFrame(documents), use_container_width=True)
        else:
            st.info("No uploaded knowledge documents yet.")
    except ApiClientError as exc:
        st.warning(str(exc))
    st.markdown("</div>", unsafe_allow_html=True)


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-kicker">Ready</div>
            <div class="empty-title">Select a walkthrough or enter a custom question</div>
            <div class="empty-body">
                推荐先看 <code>BEAR 数据边界</code>、<code>温度时序查询</code> 和
                <code>策略建议边界</code> 三个案例 —— 它们覆盖了 RAG 检索、时序工具和策略边界三种典型路由。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_status_grid(
        [
            {"label": "Route", "value": "Standby", "hint": "默认进入 LangGraph workflow"},
            {"label": "Tools", "value": "Standby", "hint": "按问题类型调用 RAG / timeseries / policy"},
            {"label": "Audit", "value": "Armed", "hint": "回答会检查生产遥测和 LLM 控制误述"},
        ]
    )


def _render_result(result: dict) -> None:
    _render_answer_panel(result.get("answer", ""))
    _render_status_grid(build_status_cards(result))

    _render_section_label("Structured Evidence")
    _render_tool_summary(result.get("tool_results", []))

    with st.expander("Execution Timeline", expanded=True):
        st.dataframe(pd.DataFrame(build_execution_timeline(result)), use_container_width=True)

    trace_rows = build_workflow_trace_rows(result)
    if trace_rows:
        with st.expander("LangGraph Workflow Trace", expanded=True):
            st.caption(
                "StateGraph nodes executed by the selected workflow, including route, tool, evidence, and audit state."
            )
            st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)

    audit = result.get("answer_audit")
    if audit:
        with st.expander("Safety Audit", expanded=True):
            st.metric("Audit", "passed" if audit.get("passed") else "review")
            st.dataframe(
                pd.DataFrame(build_safety_audit_rows(result)),
                use_container_width=True,
            )

    with st.expander("Citations", expanded=True):
        citations = result.get("citations", [])
        if citations:
            st.json(citations)
        else:
            st.caption("No citations returned.")

    with st.expander("Tool Results", expanded=True):
        tool_results = result.get("tool_results", [])
        if tool_results:
            st.json(tool_results)
        else:
            st.caption("No tool results returned.")

    with st.expander("Retrieved Contexts"):
        contexts = result.get("retrieved_contexts", [])
        if contexts:
            st.json(contexts)
        else:
            st.caption("No retrieved contexts returned.")


def _render_tool_summary(tool_results: list[dict]) -> None:
    for result in tool_results:
        summary = result.get("summary")
        if isinstance(summary, dict):
            _render_section_label("Metric Summary")
            summary_frame = pd.DataFrame([summary])
            st.dataframe(summary_frame, use_container_width=True)

        records = result.get("records") or result.get("series")
        if isinstance(records, list) and records:
            records_frame = pd.DataFrame(records)
            _render_section_label("Trend")
            st.dataframe(records_frame, use_container_width=True)
            if {"timestamp", "value"}.issubset(records_frame.columns):
                chart_frame = records_frame.copy()
                chart_frame["timestamp"] = pd.to_datetime(chart_frame["timestamp"])
                st.line_chart(chart_frame, x="timestamp", y="value")

        anomalies = result.get("anomalies")
        if isinstance(anomalies, list) and anomalies:
            _render_section_label("Anomalies")
            st.dataframe(pd.DataFrame(anomalies), use_container_width=True)


def _render_eval_result(result: dict) -> None:
    metrics = result.get("metrics", {})
    if metrics:
        _render_section_label("Metrics")
        grouped_metrics = group_eval_metrics(metrics)
        for group_name, metric_items in grouped_metrics.items():
            st.markdown(f"**{group_name}**")
            columns = st.columns(min(3, len(metric_items)))
            for index, (name, value) in enumerate(metric_items):
                with columns[index % len(columns)]:
                    st.metric(name, f"{value:.3f}")
        st.caption(
            "Quality Proxy 指标来自本地 must_include / must_not_include 弱标注，"
            "不等价于人工评审或 LLM judge。"
        )
        st.dataframe(
            pd.DataFrame(
                [{"metric": name, "value": value} for name, value in metrics.items()]
            ),
            use_container_width=True,
        )

    predictions = result.get("predictions", [])
    if predictions:
        st.subheader("Predictions")
        preview = build_prediction_preview(predictions)
        st.dataframe(pd.DataFrame(preview), use_container_width=True)
        with st.expander("Raw predictions"):
            st.json(predictions)


if __name__ == "__main__":
    main()
