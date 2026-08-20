"""
CV Interaktif — "Data Ledger" Edition
Jalankan dengan: streamlit run cv_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Andi Pratama — CV",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DATA DUMMY
# ============================================================
PROFILE = {
    "name": "Andi Pratama",
    "title": "Data Analyst & Backend Developer",
    "location": "Kediri, Jawa Timur, Indonesia",
    "email": "andi.pratama@email.com",
    "phone": "+62 812-3456-7890",
    "linkedin": "linkedin.com/in/andipratama",
    "github": "github.com/andipratama",
    "initials": "AP",
    "roles": ["Data Analyst", "Python Developer", "Dashboard Builder"],
    "summary": (
        "Data Analyst dengan pengalaman 4+ tahun mengolah data bisnis menjadi "
        "insight yang actionable. Terbiasa bekerja dengan Python, SQL, dan tools "
        "visualisasi data. Memiliki ketertarikan kuat pada pengembangan backend "
        "dan otomasi proses menggunakan Python."
    ),
}

STATS = [
    ("4+", "Tahun Pengalaman"),
    ("3", "Perusahaan"),
    ("12", "Proyek Selesai"),
    ("3", "Sertifikasi"),
]

EXPERIENCE = [
    {
        "role": "Senior Data Analyst",
        "company": "PT Teknologi Nusantara",
        "period": "2023 — Sekarang",
        "location": "Surabaya, Indonesia",
        "points": [
            "Membangun dashboard monitoring penjualan real-time untuk 15+ cabang",
            "Mengotomasi pipeline ETL harian, memangkas waktu proses dari 3 jam ke 20 menit",
            "Memimpin tim 3 analyst junior dalam proyek segmentasi pelanggan",
        ],
    },
    {
        "role": "Data Analyst",
        "company": "CV Digital Solusi",
        "period": "2021 — 2022",
        "location": "Malang, Indonesia",
        "points": [
            "Menganalisis data transaksi untuk mendukung keputusan strategi pemasaran",
            "Membuat laporan mingguan menggunakan Python (pandas) dan Power BI",
            "Berkolaborasi dengan tim produk mendefinisikan metrik utama (KPI)",
        ],
    },
    {
        "role": "Junior Backend Developer",
        "company": "Startup Kreatif ID",
        "period": "2020 — 2021",
        "location": "Remote",
        "points": [
            "Mengembangkan REST API menggunakan Flask untuk aplikasi mobile",
            "Mengelola database PostgreSQL dan optimasi query",
        ],
    },
]

EDUCATION = [
    {
        "degree": "S1 Teknik Informatika",
        "school": "Universitas Brawijaya",
        "period": "2016 — 2020",
        "detail": "IPK 3.65/4.00 — Fokus pada Data Science & Rekayasa Perangkat Lunak",
    },
    {
        "degree": "SMA Negeri 1 Kediri",
        "school": "Jurusan IPA",
        "period": "2013 — 2016",
        "detail": "Lulus dengan predikat cumlaude",
    },
]

CERTIFICATIONS = [
    {"name": "Google Data Analytics Professional Certificate", "year": "2023"},
    {"name": "SQL for Data Science — Coursera", "year": "2022"},
    {"name": "Python for Everybody — University of Michigan", "year": "2021"},
]

SKILLS = {
    "Bahasa & Query": [("Python", 90), ("SQL", 85), ("JavaScript", 60)],
    "Tools & Framework": [("Pandas / NumPy", 90), ("Streamlit", 80), ("Flask", 75), ("Power BI", 70)],
    "Infrastruktur": [("Git / GitHub", 80), ("PostgreSQL", 75), ("Docker", 55)],
}

LANGUAGES = [("Bahasa Indonesia", "Native"), ("Bahasa Inggris", "Profesional — TOEFL 570")]

PROJECTS = [
    {
        "name": "Dashboard Penjualan Nasional",
        "desc": "Dashboard interaktif berbasis Streamlit untuk memonitor performa penjualan 15+ cabang secara real-time.",
        "tags": ["Python", "Streamlit", "PostgreSQL"],
        "link": "github.com/andipratama/sales-dashboard",
    },
    {
        "name": "Sistem Rekomendasi Produk",
        "desc": "Model machine learning sederhana untuk merekomendasikan produk berdasarkan riwayat pembelian.",
        "tags": ["Python", "Scikit-learn", "Pandas"],
        "link": "github.com/andipratama/product-recsys",
    },
    {
        "name": "API Manajemen Inventaris",
        "desc": "REST API untuk pengelolaan stok barang gudang dengan autentikasi JWT.",
        "tags": ["Flask", "PostgreSQL", "Docker"],
        "link": "github.com/andipratama/inventory-api",
    },
]

# ============================================================
# CSS — "DATA LEDGER" DESIGN SYSTEM
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;0,9..144,900;1,9..144,500&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root{
        --paper:#F7F7F3;
        --panel:#FFFFFF;
        --ink:#15181B;
        --muted:#767D85;
        --line:#E4E3DD;
        --accent:#0F766E;
        --accent-soft:#DCF2EE;
        --accent-ink:#0A4B45;
    }

    html, body, [class*="css"]{ font-family:'Inter', sans-serif; color:var(--ink); }
    .stApp{ background-color:var(--paper); }
    #MainMenu, footer, header{ visibility:hidden; }
    .block-container{ padding-top:2rem; max-width:1100px; }

    @media (prefers-reduced-motion: reduce){
        *{ animation:none !important; transition:none !important; }
    }
    *:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }

    /* ---------- keyframes ---------- */
    @keyframes fadeInUp{ from{opacity:0; transform:translateY(16px);} to{opacity:1; transform:translateY(0);} }
    @keyframes fillBar{ from{width:0%;} to{width:var(--w);} }
    @keyframes pulseDot{
        0%{ box-shadow:0 0 0 0 rgba(15,118,110,.45); }
        70%{ box-shadow:0 0 0 7px rgba(15,118,110,0); }
        100%{ box-shadow:0 0 0 0 rgba(15,118,110,0); }
    }
    @keyframes roleRotate{
        0%{ opacity:0; transform:translateY(10px); }
        6%{ opacity:1; transform:translateY(0); }
        28%{ opacity:1; transform:translateY(0); }
        34%{ opacity:0; transform:translateY(-10px); }
        100%{ opacity:0; transform:translateY(-10px); }
    }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"]{ background-color:var(--ink); }
    [data-testid="stSidebar"] *{ color:#EDEEEC !important; }
    [data-testid="stSidebar"] .block-container{ padding-top:2.2rem; }

    .avatar{
        width:64px; height:64px; border-radius:50%;
        background:var(--accent); color:#fff !important;
        display:flex; align-items:center; justify-content:center;
        font-family:'Fraunces', serif; font-weight:600; font-size:1.4rem;
        margin-bottom:.9rem; transition:transform .25s ease;
    }
    .avatar:hover{ transform:scale(1.06) rotate(-2deg); }

    .status-pill{
        display:inline-flex; align-items:center; gap:7px;
        font-family:'JetBrains Mono', monospace; font-size:.72rem;
        letter-spacing:.04em; color:#B9F0E7 !important;
        background:rgba(15,118,110,.18); border:1px solid rgba(15,118,110,.4);
        padding:4px 10px; border-radius:20px; margin:.6rem 0 1.1rem 0;
    }
    .status-dot{ width:7px; height:7px; border-radius:50%; background:#2DD4BF; animation:pulseDot 2s infinite; }

    .side-contact{ font-size:.86rem; line-height:2; opacity:.9; }
    .side-divider{ border:none; border-top:1px solid rgba(255,255,255,.14); margin:1.1rem 0; }

    [data-testid="stSidebar"] .stDownloadButton button{
        background:var(--accent) !important; color:#fff !important; border:none !important;
        border-radius:8px !important; font-weight:600 !important; letter-spacing:.02em;
        transition:transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stSidebar"] .stDownloadButton button:hover{
        transform:translateY(-2px); box-shadow:0 8px 18px rgba(0,0,0,.35);
    }

    /* ---------- hero ---------- */
    .eyebrow{
        font-family:'JetBrains Mono', monospace; font-size:.78rem; letter-spacing:.12em;
        color:var(--accent-ink); text-transform:uppercase; margin-bottom:.6rem;
        animation:fadeInUp .5s ease-out both;
    }
    .hero-name{
        font-family:'Fraunces', serif; font-weight:900; font-style:normal;
        font-size:clamp(2.4rem, 5.4vw, 4.1rem); line-height:.98; letter-spacing:-1px;
        margin:0 0 .3rem 0; animation:fadeInUp .6s ease-out .05s both;
    }
    .role-rotator{
        position:relative; height:1.7em; overflow:hidden; margin-bottom:1.1rem;
        animation:fadeInUp .6s ease-out .1s both;
    }
    .role-rotator span{
        position:absolute; left:0; top:0;
        font-family:'Fraunces', serif; font-style:italic; font-weight:500;
        font-size:1.35rem; color:var(--accent); opacity:0;
        animation:roleRotate 9s ease-in-out infinite;
    }
    .role-rotator span:nth-child(2){ animation-delay:3s; }
    .role-rotator span:nth-child(3){ animation-delay:6s; }

    .hero-summary{
        max-width:640px; color:var(--muted); font-size:1.02rem; line-height:1.65;
        animation:fadeInUp .6s ease-out .15s both;
    }

    .stat-strip{
        display:flex; flex-wrap:wrap; gap:0; margin-top:1.8rem;
        border-top:1px solid var(--line); border-bottom:1px solid var(--line);
        animation:fadeInUp .6s ease-out .2s both;
    }
    .stat-item{
        flex:1; min-width:120px; padding:1rem 1.2rem 1rem 0; border-right:1px solid var(--line);
    }
    .stat-item:last-child{ border-right:none; }
    .stat-num{ font-family:'JetBrains Mono', monospace; font-weight:600; font-size:1.7rem; color:var(--ink); }
    .stat-label{ font-size:.78rem; color:var(--muted); letter-spacing:.02em; margin-top:2px; }

    /* ---------- section labels ---------- */
    .sec-label{
        font-family:'JetBrains Mono', monospace; font-size:.76rem; letter-spacing:.14em;
        text-transform:uppercase; color:var(--accent-ink); margin:.2rem 0 1.1rem 0;
        display:flex; align-items:center; gap:10px;
    }
    .sec-label::after{ content:""; flex:1; height:1px; background:var(--line); }

    /* ---------- cards ---------- */
    .card{
        background:var(--panel); border:1px solid var(--line); border-radius:12px;
        padding:1.15rem 1.3rem; margin-bottom:.85rem;
        transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        animation:fadeInUp .55s ease-out both;
    }
    .card:hover{
        transform:translateY(-3px); border-color:var(--accent);
        box-shadow:0 12px 28px -14px rgba(15,23,20,.25);
    }
    .card-top{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap; }
    .card-title{ font-family:'Fraunces', serif; font-weight:600; font-size:1.12rem; }
    .card-period{
        font-family:'JetBrains Mono', monospace; font-size:.76rem; color:var(--accent-ink);
        background:var(--accent-soft); padding:2px 9px; border-radius:20px; white-space:nowrap;
    }
    .card-subtitle{ font-size:.9rem; color:var(--muted); margin:.15rem 0 .6rem 0; }
    .card ul{ margin:.3rem 0 0 1.1rem; padding:0; }
    .card li{ font-size:.92rem; color:#40464C; margin-bottom:.32rem; line-height:1.5; }

    /* stagger children inside a single html block */
    .stack .card:nth-child(1){ animation-delay:.03s; }
    .stack .card:nth-child(2){ animation-delay:.11s; }
    .stack .card:nth-child(3){ animation-delay:.19s; }
    .stack .card:nth-child(4){ animation-delay:.27s; }
    .stack .card:nth-child(5){ animation-delay:.35s; }

    /* ---------- timeline (experience) ---------- */
    .timeline{ position:relative; padding-left:26px; }
    .timeline::before{
        content:""; position:absolute; left:6px; top:6px; bottom:6px; width:1px; background:var(--line);
    }
    .t-item{ position:relative; margin-bottom:1.2rem; animation:fadeInUp .55s ease-out both; }
    .t-item:nth-child(1){ animation-delay:.03s; }
    .t-item:nth-child(2){ animation-delay:.13s; }
    .t-item:nth-child(3){ animation-delay:.23s; }
    .t-dot{
        position:absolute; left:-26px; top:7px; width:9px; height:9px; border-radius:50%;
        background:var(--panel); border:2px solid var(--accent);
    }

    /* ---------- skills ---------- */
    .skill-group-title{ font-family:'Fraunces', serif; font-weight:600; font-size:1rem; margin:1.1rem 0 .7rem 0; }
    .skill-row{ margin-bottom:.85rem; }
    .skill-top{ display:flex; justify-content:space-between; font-size:.88rem; margin-bottom:5px; }
    .skill-name{ color:var(--ink); font-weight:500; }
    .skill-pct{ font-family:'JetBrains Mono', monospace; color:var(--muted); font-size:.8rem; }
    .skill-track{ height:6px; border-radius:4px; background:var(--line); overflow:hidden; }
    .skill-fill{ height:100%; border-radius:4px; background:var(--accent); animation:fillBar 1.1s cubic-bezier(.2,.8,.2,1) both; }

    .lang-card{
        background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:.9rem 1.1rem;
        transition:border-color .2s ease, transform .2s ease;
    }
    .lang-card:hover{ border-color:var(--accent); transform:translateY(-2px); }

    /* ---------- projects ---------- */
    .proj-card{
        background:var(--panel); border:1px solid var(--line); border-radius:12px;
        padding:1.2rem 1.25rem; height:100%; display:flex; flex-direction:column;
        transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease;
        animation:fadeInUp .55s ease-out both;
    }
    .proj-card:hover{
        transform:translateY(-4px); border-color:var(--accent);
        box-shadow:0 14px 30px -14px rgba(15,23,20,.28);
    }
    .proj-title{ font-family:'Fraunces', serif; font-weight:600; font-size:1.06rem; margin-bottom:.4rem; }
    .proj-desc{ font-size:.88rem; color:var(--muted); line-height:1.55; flex-grow:1; }
    .proj-tags{ margin:.8rem 0 .6rem 0; }
    .tag{
        display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.7rem;
        background:var(--accent-soft); color:var(--accent-ink); padding:3px 9px;
        border-radius:20px; margin:0 5px 5px 0;
    }
    .proj-link{
        font-family:'JetBrains Mono', monospace; font-size:.8rem; font-weight:600;
        color:var(--accent) !important; text-decoration:none; display:inline-flex; align-items:center; gap:5px;
    }
    .proj-link:hover{ text-decoration:underline; }

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"]{ gap:1.6rem; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"]{
        font-family:'JetBrains Mono', monospace; font-size:.82rem; letter-spacing:.03em;
        color:var(--muted); padding:.5rem 0; background:transparent;
    }
    .stTabs [aria-selected="true"]{ color:var(--ink) !important; font-weight:600; }
    .stTabs [data-baseweb="tab-highlight"]{ background-color:var(--accent) !important; height:2px; }

    /* ---------- footer ---------- */
    .cv-footer{
        margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid var(--line);
        font-family:'JetBrains Mono', monospace; font-size:.76rem; color:var(--muted);
        display:flex; justify-content:space-between; flex-wrap:wrap; gap:.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR — PROFIL & KONTAK
# ============================================================
with st.sidebar:
    st.markdown(f"<div class='avatar'>{PROFILE['initials']}</div>", unsafe_allow_html=True)
    st.markdown(f"### {PROFILE['name']}")
    st.markdown(f"{PROFILE['title']}")
    st.markdown(
        "<div class='status-pill'><span class='status-dot'></span>TERBUKA UNTUK PELUANG BARU</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='side-divider'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='side-contact'>
        📍 {PROFILE['location']}<br>
        ✉️ {PROFILE['email']}<br>
        📱 {PROFILE['phone']}<br>
        🔗 <a href='https://{PROFILE['linkedin']}' target='_blank'>{PROFILE['linkedin']}</a><br>
        🐙 <a href='https://{PROFILE['github']}' target='_blank'>{PROFILE['github']}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='side-divider'>", unsafe_allow_html=True)

    cv_text = f"""CURRICULUM VITAE
=================
Nama   : {PROFILE['name']}
Posisi : {PROFILE['title']}
Email  : {PROFILE['email']}
Telp   : {PROFILE['phone']}

RINGKASAN
{PROFILE['summary']}

PENGALAMAN KERJA
""" + "\n".join(f"- {e['role']} di {e['company']} ({e['period']})" for e in EXPERIENCE)

    st.download_button(
        "⬇  Unduh CV (.txt)",
        data=cv_text,
        file_name=f"CV_{PROFILE['name'].replace(' ', '_')}.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ============================================================
# HERO
# ============================================================
role_spans = "".join(f"<span>{r}</span>" for r in PROFILE["roles"])
st.markdown(
    f"""
    <div class='eyebrow'>◆ CURRICULUM VITAE — {PROFILE['location'].upper()}</div>
    <div class='hero-name'>{PROFILE['name']}</div>
    <div class='role-rotator'>{role_spans}</div>
    <div class='hero-summary'>{PROFILE['summary']}</div>
    """,
    unsafe_allow_html=True,
)

stat_html = "".join(
    f"<div class='stat-item'><div class='stat-num'>{n}</div><div class='stat-label'>{l}</div></div>"
    for n, l in STATS
)
st.markdown(f"<div class='stat-strip'>{stat_html}</div>", unsafe_allow_html=True)

st.write("")
st.write("")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_exp, tab_edu, tab_skills, tab_proj = st.tabs(
    ["Ringkasan", "Pengalaman", "Pendidikan", "Keahlian", "Proyek"]
)

# ---------- RINGKASAN ----------
with tab_overview:
    st.markdown("<div class='sec-label'>Sertifikasi</div>", unsafe_allow_html=True)
    cert_html = "".join(
        f"""<div class='card'>
                <div class='card-top'>
                    <div class='card-title'>{c['name']}</div>
                    <div class='card-period'>{c['year']}</div>
                </div>
            </div>"""
        for c in CERTIFICATIONS
    )
    st.markdown(f"<div class='stack'>{cert_html}</div>", unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>Bahasa</div>", unsafe_allow_html=True)
    lcols = st.columns(len(LANGUAGES))
    for col, (lang, level) in zip(lcols, LANGUAGES):
        with col:
            st.markdown(
                f"<div class='lang-card'><b>{lang}</b><br><span style='color:var(--muted); font-size:.85rem;'>{level}</span></div>",
                unsafe_allow_html=True,
            )

# ---------- PENGALAMAN ----------
with tab_exp:
    st.markdown("<div class='sec-label'>Riwayat Pekerjaan</div>", unsafe_allow_html=True)
    items = ""
    for e in EXPERIENCE:
        points = "".join(f"<li>{p}</li>" for p in e["points"])
        items += f"""
        <div class='t-item'>
            <div class='t-dot'></div>
            <div class='card'>
                <div class='card-top'>
                    <div class='card-title'>{e['role']}</div>
                    <div class='card-period'>{e['period']}</div>
                </div>
                <div class='card-subtitle'>{e['company']} · {e['location']}</div>
                <ul>{points}</ul>
            </div>
        </div>
        """
    st.markdown(f"<div class='timeline'>{items}</div>", unsafe_allow_html=True)

# ---------- PENDIDIKAN ----------
with tab_edu:
    st.markdown("<div class='sec-label'>Riwayat Pendidikan</div>", unsafe_allow_html=True)
    edu_html = "".join(
        f"""<div class='card'>
                <div class='card-top'>
                    <div class='card-title'>{e['degree']}</div>
                    <div class='card-period'>{e['period']}</div>
                </div>
                <div class='card-subtitle'>{e['school']}</div>
                <div style='font-size:.9rem; color:#40464C;'>{e['detail']}</div>
            </div>"""
        for e in EDUCATION
    )
    st.markdown(f"<div class='stack'>{edu_html}</div>", unsafe_allow_html=True)

# ---------- KEAHLIAN ----------
with tab_skills:
    st.markdown("<div class='sec-label'>Keahlian Teknis</div>", unsafe_allow_html=True)
    cols = st.columns(len(SKILLS))
    for col, (group, items) in zip(cols, SKILLS.items()):
        with col:
            rows = "".join(
                f"""<div class='skill-row'>
                        <div class='skill-top'><span class='skill-name'>{name}</span><span class='skill-pct'>{pct}%</span></div>
                        <div class='skill-track'><div class='skill-fill' style='--w:{pct}%;'></div></div>
                    </div>"""
                for name, pct in items
            )
            st.markdown(
                f"<div class='skill-group-title'>{group}</div>{rows}",
                unsafe_allow_html=True,
            )

# ---------- PROYEK ----------
with tab_proj:
    st.markdown("<div class='sec-label'>Proyek Unggulan</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, p in enumerate(PROJECTS):
        tags = "".join(f"<span class='tag'>{t}</span>" for t in p["tags"])
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class='proj-card'>
                    <div class='proj-title'>{p['name']}</div>
                    <div class='proj-desc'>{p['desc']}</div>
                    <div class='proj-tags'>{tags}</div>
                    <a class='proj-link' href='https://{p['link']}' target='_blank'>↗ Lihat proyek</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    f"""
    <div class='cv-footer'>
        <span>© 2026 {PROFILE['name']}</span>
        <span>Dibangun dengan Python · Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)