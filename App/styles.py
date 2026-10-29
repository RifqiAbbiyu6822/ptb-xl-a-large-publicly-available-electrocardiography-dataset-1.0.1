"""CSS kustom aplikasi.

Design tokens
  Surfaces   #111214 page / #17181B surface / #1E2024 raised
  Lines      bone @ 8% dan 18%
  Text       #E9E7E2 bone / #9A9892 muted / #68675F faint
  Type       Instrument Sans
"""
import streamlit as st

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: dark;
  --bg: #111214; --surface: #17181B; --raised: #1E2024;
  --line: rgba(233,231,226,0.08); --line-strong: rgba(233,231,226,0.18);
  --text: #E9E7E2; --muted: #9A9892; --faint: #68675F;
  --good: #7FCB9A; --bad: #EE7B73; --warn: #E3B25A;
  --r-sm: 8px; --r-md: 12px;
  --gutter: clamp(20px, 5vw, 72px);
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

html, body, [class*="css"], .stApp {
  font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
}

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp { background: var(--bg); color: var(--text); }
.stApp p, .stApp label, .stApp li, .stApp span { color: inherit; }

.block-container {
  max-width: min(1120px, 94vw) !important;
  padding: 3.2rem var(--gutter) 5rem var(--gutter) !important;
}
div.element-container { margin-bottom: 0 !important; }

::selection { background: rgba(233,231,226,0.22); color: var(--text); }
:focus-visible { outline: 2px solid var(--text) !important; outline-offset: 2px; }
hr { border-color: var(--line) !important; }

/* --- Type -------------------------------------------------------------- */
h1, h2, h3, h4, h5 { color: var(--text) !important; font-family: inherit !important; }
h1 { font-weight: 500 !important; letter-spacing: -0.035em !important; line-height: 1.08 !important;
  font-size: clamp(1.9rem, 3.8vw, 2.9rem) !important; margin: 0 !important; padding: 0 !important; max-width: 20em; }
h3, h5 { font-weight: 500 !important; letter-spacing: -0.015em !important; font-size: 1.05rem !important; }
p { line-height: 1.6; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; font-size: 13px !important; }

/* --- Hero ---------------------------------------------------------------- */
.hero { padding: 0 0 8px 0; margin-bottom: 44px; }
.hero-sub { color: var(--muted); font-size: 16px; margin: 18px 0 0 0; max-width: 34em; line-height: 1.55; }

.ecg-trace-wrap { width: 100%; margin: 36px 0 0 0;
  -webkit-mask-image: linear-gradient(90deg, #000 70%, transparent);
  mask-image: linear-gradient(90deg, #000 70%, transparent); }
.ecg-trace-svg { width: 100%; height: auto; display: block; overflow: visible; }
.ecg-trace-path { stroke: var(--text); stroke-opacity: 0.75; stroke-width: 1.4; stroke-linejoin: round;
  stroke-dasharray: 1; stroke-dashoffset: 1; animation: ecgDraw 2.8s var(--ease) 0.15s forwards; }
@keyframes ecgDraw { to { stroke-dashoffset: 0; } }

/* --- Section headings ------------------------------------------------------ */
.section-title { font-size: 1.05rem; font-weight: 500; letter-spacing: -0.015em; color: var(--text); margin: 0 0 6px 0; }
.section-hint { font-size: 13.5px; color: var(--muted); margin: 0 0 22px 0; max-width: 46em; }

/* --- Controls ------------------------------------------------------------- */
.stButton > button, button[data-testid="stBaseButton-secondary"] {
  border-radius: 10px; font-weight: 500; border: 1px solid var(--line-strong); background: transparent; color: var(--text);
  transition: background .2s var(--ease), border-color .2s var(--ease), color .2s var(--ease); }
.stButton > button:hover { background: var(--raised); border-color: var(--line-strong); color: var(--text); }

.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: var(--text); color: #111214; border: 1px solid var(--text); font-weight: 600;
  padding: 0.65rem 1.2rem; letter-spacing: -0.005em; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  background: #FFFFFF; border-color: #FFFFFF; color: #111214; }
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #111214; }
.stButton > button:active { transform: scale(0.995); }

.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
  border-radius: var(--r-sm) !important; background: var(--surface) !important; border: 1px solid var(--line) !important; }

div[data-testid="stFileUploader"] label p { color: var(--muted); font-size: 13px; }
div[data-testid="stFileUploaderDropzone"] { background: var(--surface) !important; border: 1px dashed var(--line-strong) !important;
  border-radius: var(--r-md) !important; transition: border-color .2s var(--ease), background .2s var(--ease); }
div[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(233,231,226,0.45) !important; background: var(--raised) !important; }
div[data-testid="stFileUploaderDropzone"] small, div[data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; }
div[data-testid="stFileUploaderFile"] { color: var(--text); }

div[role="radiogroup"] label { font-size: 14px; color: var(--muted); }
div[role="radiogroup"] label:has(input:checked) { color: var(--text); }
div[data-testid="stToggle"] label p { font-size: 14px; color: var(--muted); }
div[data-testid="stSlider"] label p { font-size: 13px; color: var(--muted); }

div[data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--line-strong); background: var(--surface) !important; }
div[data-testid="stAlert"] > div { background: transparent !important; }
div[data-testid="stAlert"] p { color: var(--text); font-size: 14px; }
div[data-testid="stToast"] { background: var(--raised); border: 1px solid var(--line-strong); }

/* --- Tabs ----------------------------------------------------------------- */
div[data-testid="stTabs"] { margin-top: 8px; }
div[data-testid="stTabs"] button[role="tab"] { font-weight: 500; color: var(--muted); font-size: 14.5px;
  padding: 10px 2px; margin-right: 22px; transition: color .2s var(--ease); }
div[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text); background: transparent; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--text); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--text) !important; height: 1.5px !important; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--line) !important; }
div[data-testid="stTabs"] [role="tablist"] { gap: 0; }

/* --- Progress -------------------------------------------------------------- */
.scan-banner { margin: 20px 0 8px 0; }
.scan-track { height: 2px; background: var(--raised); overflow: hidden; border-radius: 2px; }
.scan-fill { height: 100%; background: var(--text); transition: width .45s var(--ease); }
.scan-label { margin-top: 10px; font-size: 13px; color: var(--muted); }

/* --- Recording summary ------------------------------------------------------ */
.meta { display: flex; flex-wrap: wrap; gap: 8px 40px; margin: 0 0 32px 0; padding: 18px 0;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.meta div { display: flex; flex-direction: column; gap: 2px; }
.meta dt { font-size: 12.5px; color: var(--faint); }
.meta dd { margin: 0; font-size: 15px; color: var(--text); font-weight: 500; }

/* --- Verdict + diagnosis rows ---------------------------------------------- */
.verdict { margin: 28px 0 36px 0; }
.verdict-label { font-size: 13px; color: var(--muted); margin: 0 0 8px 0; }
.verdict-value { display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0; font-size: clamp(1.7rem, 3vw, 2.3rem);
  font-weight: 500; letter-spacing: -0.03em; line-height: 1.15; color: var(--text); }
.verdict-value.none { color: var(--muted); }
.verdict-item { display: inline-flex; align-items: center; gap: 12px; }
.verdict-item i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.dx-list { border-bottom: 1px solid var(--line); }
.dx-row { display: grid; grid-template-columns: minmax(150px, 230px) 1fr 58px minmax(96px, 128px);
  align-items: center; gap: 24px; padding: 20px 0; border-top: 1px solid var(--line); }
.dx-name { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.dx-code { display: inline-flex; align-items: center; gap: 10px; font-size: 16px; font-weight: 600; color: var(--muted); }
.dx-code i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); opacity: 0.4; display: inline-block; }
.dx-long { font-size: 13px; color: var(--faint); padding-left: 18px; }
.dx-row.active .dx-code { color: var(--text); }
.dx-row.active .dx-code i { opacity: 1; }
.dx-row.active .dx-long { color: var(--muted); }

.dx-bar { position: relative; height: 4px; background: var(--raised); border-radius: 4px; }
.dx-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--c); border-radius: 4px; opacity: 0.35;
  animation: barGrow 0.9s var(--ease) both; }
.dx-row.active .dx-fill { opacity: 1; }
.dx-tick { position: absolute; top: -5px; width: 1px; height: 14px; background: var(--text); opacity: 0.55; }
@keyframes barGrow { from { width: 0; } }

.dx-pct { text-align: right; font-size: 18px; font-weight: 500; color: var(--muted); letter-spacing: -0.01em; }
.dx-row.active .dx-pct { color: var(--text); }
.dx-state { text-align: right; display: flex; flex-direction: column; gap: 2px; font-size: 13px; color: var(--faint); }
.dx-state b { font-weight: 500; color: var(--faint); }
.dx-row.active .dx-state b { color: var(--c); }
.dx-state span { font-size: 12px; color: var(--faint); }
.dx-note { font-size: 12.5px; color: var(--faint); margin: 18px 0 0 0; max-width: 46em; }

/* --- Signal panel --------------------------------------------------------------- */
.wide-plot-hint { color: var(--faint); font-size: 12.5px; margin: 14px 0 12px 0; }

/* --- Empty state -------------------------------------------------------------------- */
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 32px; margin-top: 48px; }
.step { border-top: 1px solid var(--line-strong); padding-top: 16px; }
.step-num { font-size: 13px; color: var(--faint); font-weight: 500; }
.step-title { font-size: 15px; font-weight: 500; color: var(--text); margin: 8px 0 6px 0; }
.step-desc { font-size: 13.5px; color: var(--muted); line-height: 1.6; margin: 0; }

.legend { margin-top: 56px; }
.legend-title { font-size: 1.05rem; font-weight: 500; letter-spacing: -0.015em; color: var(--text); margin: 0 0 14px 0; }
.legend-grid { display: flex; flex-wrap: wrap; gap: 10px 36px; }
.legend-item { display: inline-flex; align-items: center; gap: 10px; font-size: 14px; color: var(--muted); }
.legend-item i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.legend-item b { color: var(--text); font-weight: 600; }

@media (max-width: 900px) {
  .steps { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 720px) {
  :root { --gutter: 16px; }
  .block-container { padding-top: 1.6rem !important; }
  .hero { margin-bottom: 32px; }
  .dx-row { grid-template-columns: 1fr auto; gap: 12px 16px; padding: 16px 0; }
  .dx-bar { grid-column: 1 / -1; grid-row: 2; }
  .dx-pct { grid-row: 1; grid-column: 2; }
  .dx-state { grid-column: 1 / -1; grid-row: 3; text-align: left; flex-direction: row; gap: 10px; }
  .meta { gap: 14px 28px; }
}
@media (max-width: 520px) {
  .steps { grid-template-columns: 1fr; gap: 24px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    animation-delay: 0ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.001ms !important;
    scroll-behavior: auto !important;
  }
  .ecg-trace-path { stroke-dashoffset: 0; }
}
"""


def inject_css():
    st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)
