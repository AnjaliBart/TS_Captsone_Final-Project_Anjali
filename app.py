"""
Streamlit Dashboard

"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analyzer import get_analyzer, AnalysisResult

st.set_page_config(
    page_title="Return Risk Analyzer",
    page_icon="🔎", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 60%, #0d2137 100%);
    border-radius: 16px; padding: 2.2rem 3rem; margin-bottom: 1.8rem; color: white;
}
.hero h1 { font-size: 2.2rem; margin: 0 0 0.3rem 0; color: #f0f4ff !important; }
.hero p  { color: #a0aec0; font-size: 1rem; margin: 0; }

.kpi-card {
    background: white; border-radius: 12px; padding: 1.1rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); border-top: 3px solid #4F8BF9;
    text-align: center;
}
.kpi-val { font-size: 2rem; font-weight: 700; color: #1a202c; }
.kpi-lbl { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; }

.claim-row {
    display: flex; align-items: flex-start; background: #f7fafc;
    border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.55rem;
    border-left: 4px solid #ccc; gap: 0.8rem;
}
.claim-confirmed    { border-left-color: #48bb78; }
.claim-contradicted { border-left-color: #fc8181; background: #fff5f5; }
.claim-unverified   { border-left-color: #f6ad55; }

.fix-card {
    background: #fffbeb; border: 1px solid #f6e05e;
    border-radius: 10px; padding: 0.75rem 1.1rem;
    margin-bottom: 0.45rem; font-size: 0.91rem;
}
.risk-Low      { color:#276749; background:#f0fff4; border:1px solid #9ae6b4; }
.risk-Medium   { color:#7b4c00; background:#fffbeb; border:1px solid #f6d860; }
.risk-High     { color:#9b2c2c; background:#fff5f5; border:1px solid #fc8181; }
.risk-Critical { color:white;   background:#c53030; border:1px solid #9b2c2c; }
.risk-badge { display:inline-block; padding:3px 14px; border-radius:20px; font-weight:700; font-size:1rem; }
.sec { font-family:'DM Serif Display',serif; font-size:1.25rem; color:#1a202c;
       border-bottom:2px solid #e2e8f0; padding-bottom:0.35rem; margin:1.4rem 0 0.9rem 0; }
</style>
""", unsafe_allow_html=True)


#  Sample data (Apparel + Beauty context) 
SAMPLE_DESC = """
Premium Women's Floral Kurta — Summer Collection
Crafted from 100% breathable cotton fabric with soft touch finish.
True to size fit — our size chart is highly accurate.
Vibrant colours exactly as shown in pictures.
Lightweight at just 180g, comfortable for all-day wear.
Genuine hand-embroidery, colours are colourfast and won't fade after wash.
Comes beautifully gift-wrapped in premium eco-friendly packaging.
Money-back guarantee if not satisfied.
""".strip()

SAMPLE_REVIEWS = """
The colour is completely different from the photo — looks nothing like the listing.
Runs very small, I normally wear M but had to return and order L.
Material feels cheap and synthetic, definitely not 100% cotton as described.
Embroidery started fraying after the first wash — very poor quality.
Beautiful kurta! Fits perfectly true to size and the fabric is so soft.
Colour faded badly after one wash despite the colourfast claim.
Packaging was damaged and the kurta had a stain when it arrived.
Totally misleading description — the embroidery is printed, not hand-done.
Love the design! Exactly as shown, great value for money.
Way too short for my height, size chart is completely wrong.
""".strip()

SAMPLE_DESC_BEAUTY = """
Organic Rose Face Serum — Anti-Aging Formula
Dermatologist tested, hypoallergenic formula safe for sensitive skin.
Paraben-free and cruelty-free. Fragrance-free formula.
Long-lasting hydration — 24-hour moisture lock guaranteed.
100% natural ingredients, no artificial preservatives.
Luxury glass bottle with premium packaging. Money-back guarantee.
""".strip()

SAMPLE_REVIEWS_BEAUTY = """
Caused a breakout on my sensitive skin — definitely not hypoallergenic.
The serum leaked inside the package, no bubble wrap protection at all.
Smells strongly of fragrance — the fragrance-free claim is false advertising.
Love this serum! Skin feels hydrated all day, exactly as described.
Not as described — contains synthetic ingredients despite the natural claim.
Irritated my skin badly after first use. Returning immediately.
Beautiful packaging but the product doesn't moisturise at all, waste of money.
Great product, absorbed quickly and skin looks better after 2 weeks.
The glass bottle was cracked on arrival — poor packaging.
Works well for dry skin, would recommend for normal skin types.
""".strip()


#  Helpers 
def risk_color(level):
    return {"Low":"#48bb78","Medium":"#f6ad55","High":"#fc8181","Critical":"#c53030"}.get(level,"#ccc")

def gauge(score, level):
    color = risk_color(level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix":"/100","font":{"size":26}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#ccc"},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"white",
            "steps":[
                {"range":[0,25],"color":"#f0fff4"},{"range":[25,50],"color":"#fffbeb"},
                {"range":[50,75],"color":"#fff5f5"},{"range":[75,100],"color":"#fed7d7"},
            ],
            "threshold":{"line":{"color":color,"width":4},"thickness":0.8,"value":score},
        },
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig.update_layout(height=210, margin=dict(t=20,b=0,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def reason_bar(reason_dist):
    items  = sorted([(k,v) for k,v in reason_dist.items() if v>0.01], key=lambda x:-x[1])
    labels = [k for k,v in items]
    vals   = [round(v*100,1) for k,v in items]
    colors = ["#fc8181" if v>15 else "#f6ad55" if v>8 else "#90cdf4" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=colors,
                           text=[f"{v}%" for v in vals], textposition="outside"))
    fig.update_layout(height=max(200,len(labels)*42),
                      margin=dict(t=10,b=10,l=10,r=60),
                      xaxis=dict(title="% of Reviews", range=[0,max(vals)*1.3]),
                      yaxis=dict(autorange="reversed"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans"))
    return fig

def radar(breakdown):
    cats  = list(breakdown.keys())
    vals  = list(breakdown.values())
    maxes = {"Claim Contradictions":40,"Return Reason Signal":35,"Negative Sentiment":25}
    pcts  = [v/maxes.get(c,40)*100 for c,v in zip(cats,vals)]
    fig   = go.Figure(go.Scatterpolar(
        r=pcts+[pcts[0]], theta=cats+[cats[0]],
        fill="toself", fillcolor="rgba(252,129,129,0.25)",
        line_color="#fc8181", line_width=2,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,100],tickfont=dict(size=9)),
                   angularaxis=dict(tickfont=dict(size=11))),
        showlegend=False, height=250,
        margin=dict(t=20,b=20,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def render_card(v):
    css   = {"Confirmed":"claim-confirmed","Contradicted":"claim-contradicted","Unverified":"claim-unverified"}.get(v.status,"")
    icon  = {"Confirmed":"✅","Contradicted":"❌","Unverified":"⚠️"}.get(v.status,"❓")
    conf  = int(v.confidence*100)
    evid  = "".join(f"<span style='font-size:0.8rem;color:#555;'>{e}<br></span>" for e in v.evidence[:1])
    st.markdown(f"""
    <div class="claim-row {css}">
      <div style="font-size:1.3rem;line-height:1">{icon}</div>
      <div style="flex:1">
        <strong>{v.claim_type}</strong>
        <span style="font-size:0.8rem;color:#888;margin-left:8px;">"{v.claim_text}"</span>
        <span style="float:right;font-size:0.78rem;color:#999;">{v.status} · {conf}%</span>
        {"<br>"+evid if evid else ""}
      </div>
    </div>""", unsafe_allow_html=True)


#  Hero 
st.markdown("""
<div class="hero">
  <h1>🔎 Return Risk Analyzer for Apparel & Beauty </h1>
  <p> Find the gap between what you promise and what customers experience.</p>
</div>""", unsafe_allow_html=True)

#  Sample selector 
sample = st.selectbox(
    "Load a sample product:",
    ["— custom input —", "Apparel: Women's Floral Kurta", "Beauty: Rose Face Serum"],
)
if "Apparel" in sample:
    default_desc, default_rev = SAMPLE_DESC, SAMPLE_REVIEWS
elif "Beauty" in sample:
    default_desc, default_rev = SAMPLE_DESC_BEAUTY, SAMPLE_REVIEWS_BEAUTY
else:
    default_desc, default_rev = "", ""

#  Input columns 
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("<div class='sec'>Product Description</div>", unsafe_allow_html=True)
    description = st.text_area("Description", value=default_desc, height=220,
                                label_visibility="collapsed")
with c2:
    st.markdown("<div class='sec'>Customer Reviews  <span style='font-size:0.8rem;color:#888;font-family:DM Sans'>(one per line)</span></div>", unsafe_allow_html=True)
    reviews_raw = st.text_area("Reviews", value=default_rev, height=220,
                                label_visibility="collapsed")

st.markdown("")
run = st.button("Analyze Return Risk", type="primary", use_container_width=True)

#  Run analysis 
if run:
    reviews = [r.strip() for r in reviews_raw.strip().splitlines() if r.strip()]
    if not description.strip() or len(reviews) < 2:
        st.warning("Please provide a description and at least 2 reviews.")
        st.stop()

    with st.spinner("Extracting claims, verifying reviews, computing risk score..."):
        result: AnalysisResult = get_analyzer().analyze(description, reviews)

    st.markdown("---")

    #  KPI row 
    k1, k2, k3, k4 = st.columns(4)
    css = f"risk-{result.risk_level}"
    for col, val, lbl in [
        (k1, str(result.risk_score), "Return Risk Score"),
        (k2, f'<span class="risk-badge {css}">{result.risk_level}</span>', "Risk Level"),
        (k3, str(len(result.claim_verdicts)), "Claims Detected"),
        (k4, f'<span style="color:#e53e3e">{sum(1 for v in result.claim_verdicts if v.status=="Contradicted")}</span>', "Contradicted"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown("<div class='sec'>Claim Verification</div>", unsafe_allow_html=True)
        for v in result.claim_verdicts:
            render_card(v)

        st.markdown("<div class='sec'>Suggested Fixes</div>", unsafe_allow_html=True)
        for fix in result.suggested_fixes:
            st.markdown(f"<div class='fix-card'>{fix}</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='sec'>Risk Score</div>", unsafe_allow_html=True)
        st.plotly_chart(gauge(result.risk_score, result.risk_level), use_container_width=True)

        st.markdown("<div class='sec'>Risk Breakdown</div>", unsafe_allow_html=True)
        st.plotly_chart(radar(result.risk_breakdown), use_container_width=True)

        st.markdown("<div class='sec'>Return Reason Distribution (ML)</div>", unsafe_allow_html=True)
        st.plotly_chart(reason_bar(result.return_reason_distribution), use_container_width=True)
        st.caption(f"Top predicted return reason: **{result.top_return_reason}**")
