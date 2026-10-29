"""CSS kustom aplikasi.

Design tokens
  Surfaces   #111214 page / #17181B surface / #1E2024 raised
  Lines      bone @ 8% dan 18%
  Text       #E9E7E2 bone / #9A9892 muted / #86857D faint
  Type       Instrument Sans, 3 bobot (400 / 500 / 600)
             meta 12px / secondary 14px / body 16px
  Viewport   narrow 375 / medium 768 / wide 1280 (satu breakpoint: < 768)
  Motion     150ms / 200ms, hanya opacity dan transform
"""
import streamlit as st

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');

:root {
  color-scheme: dark;
  --bg: #111214; --surface: #17181B; --raised: #1E2024;
  --line: rgba(233,231,226,0.08); --line-strong: rgba(233,231,226,0.18);
  --text: #E9E7E2; --muted: #9A9892; --faint: #86857D;
  --good: #7FCB9A; --bad: #EE7B73; --warn: #E3B25A;
  --r-sm: 8px; --r-md: 12px;

  /* type */
  --font-regular: 400; --font-medium: 500; --font-semibold: 600;
  --text-meta: 0.75rem; --text-secondary: 0.875rem; --text-body: 1rem;
  --text-h3: 1.2rem; --text-h1: clamp(1.728rem, 3.8vw, 2.488rem);
  --leading-tight: 1.2; --leading-body: 1.55; --leading-meta: 1.4;

  /* viewport */
  --vp-narrow: 375px; --vp-medium: 768px; --vp-wide: 1280px;
  --hit: 44px;
  --gutter: clamp(16px, 5vw, 72px);

  /* motion */
  --motion-150: 150ms; --motion-200: 200ms;
  --ease: cubic-bezier(0.2, 0.7, 0.2, 1);
}

html, body, [class*="css"], .stApp {
  font-family: 'Instrument Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-variant-numeric: tabular-nums;
  font-size: var(--text-body);
  font-weight: var(--font-regular);
}

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
section[data-testid="stSidebar"] { display: none !important; }

.stApp { background: var(--bg); color: var(--text); }
.stApp p, .stApp label, .stApp li, .stApp span { color: inherit; }

.block-container {
  width: 100%;
  max-width: min(var(--vp-wide), 100%) !important;
  padding: 3.2rem var(--gutter) 5rem var(--gutter) !important;
}
div.element-container { margin-bottom: 0 !important; }

::selection { background: rgba(233,231,226,0.22); color: var(--text); }
:focus-visible { outline: 2px solid var(--text) !important; outline-offset: 2px; }
hr { border-color: var(--line) !important; }

/* --- Type -------------------------------------------------------------- */
h1, h2, h3, h4, h5 { color: var(--text) !important; font-family: inherit !important; line-height: var(--leading-tight) !important; }
h1 { font-weight: var(--font-semibold) !important; letter-spacing: -0.02em !important;
  font-size: var(--text-h1) !important; margin: 0 !important; padding: 0 !important; max-width: 20em; }
h3, h5 { font-weight: var(--font-medium) !important; font-size: var(--text-h3) !important; }
p { line-height: var(--leading-body); }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; font-size: var(--text-secondary) !important; }

/* --- Hero ---------------------------------------------------------------- */
.hero { padding: 0; margin-bottom: 48px; }
.hero-sub { color: var(--muted); font-size: var(--text-body); margin: 16px 0 0 0; max-width: 36em; line-height: var(--leading-body); }

.ecg-trace-wrap { width: 100%; margin: 32px 0 0 0;
  -webkit-mask-image: linear-gradient(90deg, #000 70%, transparent);
  mask-image: linear-gradient(90deg, #000 70%, transparent); }
.ecg-trace-svg { width: 100%; height: auto; display: block; overflow: visible; }
.ecg-trace-path { stroke: var(--text); stroke-opacity: 0.75; stroke-width: 1.4; stroke-linejoin: round;
  stroke-dasharray: 1; stroke-dashoffset: 1; animation: ecgDraw 1.2s var(--ease) 0.1s forwards; }

/* --- Section headings ------------------------------------------------------ */
.section-title { font-size: var(--text-h3); font-weight: var(--font-medium); line-height: var(--leading-tight);
  color: var(--text); margin: 0 0 8px 0; }
.section-hint { font-size: var(--text-secondary); color: var(--muted); margin: 0 0 24px 0; max-width: 46em; }

/* --- Controls (target tap >= 44px) ----------------------------------------- */
.stButton > button, button[data-testid="stBaseButton-secondary"] {
  min-height: var(--hit); border-radius: 10px; font-size: var(--text-body); font-weight: var(--font-medium);
  border: 1px solid var(--line-strong); background: transparent; color: var(--text);
  transition: background var(--motion-150) var(--ease), border-color var(--motion-150) var(--ease),
              color var(--motion-150) var(--ease), transform var(--motion-150) var(--ease); }

.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  min-height: var(--hit); background: var(--text); color: #111214; border: 1px solid var(--text);
  font-weight: var(--font-semibold); padding: 0.65rem 1.2rem; }
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p { color: #111214; }
.stButton > button:active { transform: scale(0.99); }

@media (hover: hover) {
  .stButton > button:hover { background: var(--raised); border-color: var(--line-strong); color: var(--text); }
  .stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
    background: #FFFFFF; border-color: #FFFFFF; color: #111214; }
  div[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(233,231,226,0.45) !important; background: var(--raised) !important; }
  div[data-testid="stTabs"] button[role="tab"]:hover { color: var(--text); background: transparent; }
}

.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] {
  min-height: var(--hit); border-radius: var(--r-sm) !important; background: var(--surface) !important;
  border: 1px solid var(--line) !important; }

div[data-testid="stFileUploader"] label p { color: var(--muted); font-size: var(--text-secondary); }
div[data-testid="stFileUploaderDropzone"] { min-height: var(--hit); background: var(--surface) !important;
  border: 1px dashed var(--line-strong) !important; border-radius: var(--r-md) !important;
  transition: border-color var(--motion-150) var(--ease), background var(--motion-150) var(--ease); }
div[data-testid="stFileUploaderDropzone"] small, div[data-testid="stFileUploaderDropzone"] span { color: var(--muted) !important; font-size: var(--text-secondary); }
div[data-testid="stFileUploaderFile"] { color: var(--text); }

div[role="radiogroup"] label { min-height: var(--hit); display: inline-flex; align-items: center;
  font-size: var(--text-body); color: var(--muted); }
div[role="radiogroup"] label:has(input:checked) { color: var(--text); }
div[data-testid="stToggle"] label { min-height: var(--hit); display: inline-flex; align-items: center; }
div[data-testid="stToggle"] label p { font-size: var(--text-secondary); color: var(--muted); }
div[data-testid="stSlider"] label p { font-size: var(--text-secondary); color: var(--muted); }

div[data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--line-strong); background: var(--surface) !important;
  animation: fadeIn var(--motion-200) var(--ease) both; }
div[data-testid="stAlert"] > div { background: transparent !important; }
div[data-testid="stAlert"] p { color: var(--text); font-size: var(--text-secondary); }
div[data-testid="stToast"] { background: var(--raised); border: 1px solid var(--line-strong); }

/* --- Tabs ----------------------------------------------------------------- */
div[data-testid="stTabs"] { margin-top: 8px; }
div[data-testid="stTabs"] button[role="tab"] { min-height: var(--hit); font-weight: var(--font-medium); color: var(--muted);
  font-size: var(--text-body); padding: 0 4px; margin-right: 24px; transition: color var(--motion-150) var(--ease); }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--text); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--text) !important; height: 2px !important; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--line) !important; }
div[data-testid="stTabs"] [role="tablist"] { gap: 0; }
div[data-testid="stTabs"] [role="tabpanel"] { animation: fadeIn var(--motion-150) var(--ease) both; }

/* --- Progress (tinggi tetap; hanya transform yang beranimasi) --------------- */
.scan-banner { margin: 24px 0 8px 0; min-height: 40px; }
.scan-track { height: 2px; background: var(--raised); overflow: hidden; border-radius: 2px; }
.scan-fill { height: 100%; width: 100%; background: var(--text); transform-origin: left center;
  transition: transform var(--motion-200) var(--ease); }
.scan-label { margin-top: 12px; font-size: var(--text-secondary); color: var(--muted); }

/* --- Recording summary ------------------------------------------------------ */
.meta { display: flex; flex-wrap: wrap; gap: 12px 40px; margin: 0 0 32px 0; padding: 16px 0;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  animation: fadeIn var(--motion-200) var(--ease) both; }
.meta div { display: flex; flex-direction: column; gap: 4px; }
.meta dt { font-size: var(--text-meta); line-height: var(--leading-meta); color: var(--faint); }
.meta dd { margin: 0; font-size: var(--text-body); color: var(--text); font-weight: var(--font-medium); }

/* --- Verdict + diagnosis rows ---------------------------------------------- */
.verdict { margin: 32px 0 40px 0; animation: fadeIn var(--motion-200) var(--ease) both; }
.verdict-label { font-size: var(--text-secondary); color: var(--muted); margin: 0 0 8px 0; }
.verdict-value { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 0; font-size: clamp(1.728rem, 3vw, 2.074rem);
  font-weight: var(--font-semibold); letter-spacing: -0.01em; line-height: var(--leading-tight); color: var(--text); }
.verdict-value.none { color: var(--muted); }
.verdict-item { display: inline-flex; align-items: center; gap: 12px; }
.verdict-item i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.dx-list { border-bottom: 1px solid var(--line); }
.dx-row { display: grid; grid-template-columns: minmax(150px, 240px) 1fr 64px minmax(110px, 140px);
  align-items: center; gap: 24px; padding: 20px 0; border-top: 1px solid var(--line); }
.dx-name { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.dx-code { display: inline-flex; align-items: center; gap: 10px; font-size: var(--text-body); font-weight: var(--font-semibold); color: var(--muted); }
.dx-code i { width: 8px; height: 8px; border-radius: 50%; background: var(--c); opacity: 0.5; display: inline-block; }
.dx-long { font-size: var(--text-secondary); color: var(--faint); padding-left: 18px; }
.dx-row.active .dx-code { color: var(--text); }
.dx-row.active .dx-code i { opacity: 1; }
.dx-row.active .dx-long { color: var(--muted); }

.dx-bar { position: relative; height: 4px; background: var(--raised); border-radius: 4px; }
.dx-fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--c); border-radius: 4px; opacity: 0.4;
  transform-origin: left center;
  animation: barGrow var(--motion-200) var(--ease) both; }   /* jeda per baris 25ms, diatur dari Python */
.dx-row.active .dx-fill { opacity: 1; }
.dx-tick { position: absolute; top: -5px; width: 1px; height: 14px; background: var(--text); opacity: 0.6; }

.dx-pct { text-align: right; font-size: var(--text-h3); font-weight: var(--font-medium); color: var(--muted); }
.dx-row.active .dx-pct { color: var(--text); }
.dx-state { text-align: right; display: flex; flex-direction: column; gap: 2px; font-size: var(--text-secondary); color: var(--faint); }
.dx-state b { font-weight: var(--font-medium); color: var(--faint); }
.dx-row.active .dx-state b { color: var(--c); }
.dx-state span { font-size: var(--text-meta); line-height: var(--leading-meta); color: var(--faint); }
.dx-note { font-size: var(--text-secondary); color: var(--faint); margin: 20px 0 0 0; max-width: 46em; }

/* --- Signal panel --------------------------------------------------------------- */
.wide-plot-hint { color: var(--faint); font-size: var(--text-secondary); margin: 16px 0 12px 0; }

/* --- Empty state -------------------------------------------------------------------- */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 32px; margin-top: 48px; }
.step { border-top: 1px solid var(--line-strong); padding-top: 16px; }
.step-num { font-size: var(--text-secondary); color: var(--faint); font-weight: var(--font-medium); }
.step-title { font-size: var(--text-body); font-weight: var(--font-medium); color: var(--text); margin: 8px 0; }
.step-desc { font-size: var(--text-secondary); color: var(--muted); line-height: var(--leading-body); margin: 0; }

.legend { margin-top: 56px; }
.legend-title { font-size: var(--text-h3); font-weight: var(--font-medium); line-height: var(--leading-tight); color: var(--text); margin: 0 0 16px 0; }
.legend-grid { display: flex; flex-wrap: wrap; gap: 12px 36px; }
.legend-item { display: inline-flex; align-items: center; gap: 10px; font-size: var(--text-secondary); color: var(--muted); }
.legend-item i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.legend-item b { color: var(--text); font-weight: var(--font-semibold); }

/* --- Motion (hanya opacity, transform, dan dash offset garis dekoratif) --------------- */
@keyframes fadeIn { from { opacity: 0; } }
@keyframes barGrow { from { transform: scaleX(0); } }
@keyframes ecgDraw { to { stroke-dashoffset: 0; } }

/* --- Narrow (< 768px) ----------------------------------------------------------------- */
@media (max-width: 767.98px) {
  :root { --gutter: 16px; }
  .block-container { padding-top: 1.6rem !important; }
  .hero { margin-bottom: 32px; }
  .dx-row { grid-template-columns: 1fr auto; gap: 12px 16px; padding: 16px 0; }
  .dx-bar { grid-column: 1 / -1; grid-row: 2; }
  .dx-pct { grid-row: 1; grid-column: 2; }
  .dx-state { grid-column: 1 / -1; grid-row: 3; text-align: left; flex-direction: row; gap: 12px; align-items: baseline; }
  .meta { gap: 16px 28px; }
  .steps { gap: 24px; }
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