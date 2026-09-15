from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'src'))
from pipeline import process_text

st.set_page_config(page_title='AI Employee Wellness', page_icon='◉', layout='wide')

st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

    :root {
        --ink: #25231f;
        --muted: #777066;
        --paper: #f4efe5;
        --paper-dark: #e9e0d1;
        --panel: #fbf8f1;
        --line: #d8cebd;
        --olive: #65705a;
        --olive-soft: #e2e7dc;
        --terracotta: #b9684f;
        --terracotta-soft: #f0ddd4;
        --mustard: #c59a45;
        --navy: #344656;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(197,154,69,.08), transparent 24%),
            radial-gradient(circle at 92% 18%, rgba(101,112,90,.08), transparent 25%),
            var(--paper);
    }

    [data-testid="stSidebar"] {
        background: var(--paper-dark);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    .brand {
        padding: 5px 2px 24px 2px;
    }

    .brand-mark {
        display: inline-flex;
        width: 42px;
        height: 42px;
        align-items: center;
        justify-content: center;
        border: 2px solid var(--ink);
        border-radius: 50%;
        background: var(--mustard);
        color: var(--ink);
        font-family: 'Fraunces', serif;
        font-size: 21px;
        margin-bottom: 12px;
    }

    .brand-title, .hero-title, .section-title {
        font-family: 'Fraunces', Georgia, serif;
    }

    .brand-title {
        font-size: 23px;
        font-weight: 700;
        line-height: 1.0;
        letter-spacing: -.02em;
    }

    .brand-sub {
        color: var(--muted);
        font-size: 12px;
        margin-top: 9px;
        line-height: 1.5;
    }

    .eyebrow {
        color: var(--terracotta);
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .hero-title {
        font-size: clamp(34px, 4vw, 58px);
        line-height: .96;
        letter-spacing: -.045em;
        margin: 0;
        max-width: 850px;
    }

    .hero-copy, .section-copy {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.65;
    }

    .hero-copy {
        max-width: 780px;
        margin-top: 15px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 700;
        letter-spacing: -.025em;
        margin: 8px 0 5px 0;
    }

    .metric-card {
        background: rgba(251,248,241,.82);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 20px;
        min-height: 124px;
        box-shadow: 3px 4px 0 rgba(52,70,86,.055);
    }

    .metric-label {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 500;
    }

    .metric-value {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 31px;
        font-weight: 700;
        margin-top: 9px;
    }

    .metric-note {
        color: var(--muted);
        font-size: 11px;
        margin-top: 4px;
    }

    .result-card {
        background: rgba(251,248,241,.86);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 4px 5px 0 rgba(52,70,86,.045);
    }

    .result-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 3px;
        background: var(--olive-soft);
        color: var(--olive);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: .05em;
    }

    .big-result {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 39px;
        font-weight: 700;
        margin-top: 9px;
    }

    .notice {
        background: var(--terracotta-soft);
        border-left: 4px solid var(--terracotta);
        padding: 13px 16px;
        border-radius: 2px;
        color: #5c4036;
        font-size: 13px;
        line-height: 1.55;
        margin: 15px 0;
    }

    .success-box {
        background: var(--olive-soft);
        border: 1px solid #c5d0ba;
        padding: 16px;
        border-radius: 3px;
        color: #40503b;
    }

    .footer {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 10px;
        padding: 28px 0 10px;
        border-top: 1px solid var(--line);
        margin-top: 35px;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 14px 16px;
    }

    .stButton > button {
        border-radius: 3px;
        border: 1px solid var(--ink);
        font-weight: 700;
        box-shadow: 2px 2px 0 var(--ink);
    }

    .stButton > button:hover {
        transform: translate(-1px, -1px);
        box-shadow: 3px 3px 0 var(--ink);
    }

    textarea {
        border-radius: 3px !important;
        background: var(--panel) !important;
    }

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] > div > div {
        border-radius: 3px;
    }

    /* Handwritten-style section marker */
    .hand-note {
        font-family: 'Fraunces', Georgia, serif;
        color: var(--terracotta);
        font-size: 15px;
        font-style: italic;
        transform: rotate(-1deg);
        display: inline-block;
        margin: 4px 0 12px 5px;
    }

    /* Make Streamlit's horizontal separators feel editorial */
    hr {
        border-color: var(--line) !important;
    }
    </style>''', unsafe_allow_html=True)

OUT = BASE / 'outputs'
def csv(name):
    p=OUT/name
    return pd.read_csv(p) if p.exists() else None
model_df=csv('model_comparison.csv')
isear_df=csv('isear_model_comparison.csv')

with st.sidebar:
    st.markdown('<div class="brand"><span class="mark">◉</span><div class="title">AI Employee<br>Wellness</div><p class="muted">Milestone 2 demonstration dashboard</p></div>', unsafe_allow_html=True)
    page=st.radio('Navigate',['Overview','Analyze Feedback','Model Comparison','ISEAR Evaluation','Validation Center','Technical Details'])
    st.divider(); st.caption('Research / internship prototype')

if page=='Overview':
    st.markdown('<div class="eyebrow">Milestone 02 · working prototype</div>',unsafe_allow_html=True)
    st.markdown('<div class="hero">Understand the signal<br>behind employee feedback.</div>',unsafe_allow_html=True)
    st.markdown('<p class="muted" style="font-size:15px;max-width:760px">Interactive demonstration of preprocessing, VADER sentiment, transformer emotion classification, confidence scoring and benchmark validation.</p>',unsafe_allow_html=True)
    cols=st.columns(4)
    vals=[('Primary model','BERT','Best Accuracy + Macro F1'),('ISEAR accuracy','54.90%','297 / 541 correct'),('ISEAR Macro F1','57.76%','BERT result'),('Edge cases','13 / 13','0 unexpected failures')]
    for c,(a,b,d) in zip(cols,vals):
        with c: st.markdown(f'<div class="card"><div class="muted">{a}</div><div class="metric">{b}</div><div class="muted">{d}</div></div>',unsafe_allow_html=True)
    a,b=st.columns([1.2,1])
    with a:
        st.markdown('<div class="card"><h3>Milestone 2 adds</h3><b>01 · Transformer models</b><br> BERT and DistilBERT integration and comparison.<br><br><b>02 · Emotion intelligence</b><br>Six emotion outputs with probability inspection.<br><br><b>03 · Confidence validation</b><br>Confidence, second-best emotion and margin.<br><br><b>04 · Benchmark evidence</b><br>Independent ISEAR held-out evaluation.</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><h3>System flow</h3><p>Employee feedback</p><p>↓</p><p>Preprocessing → VADER</p><p>↓</p><p>BERT / DistilBERT</p><p>↓</p><p>Emotion → Confidence → Result</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="notice"><b>Interpretation:</b> model emotion predictions are research outputs, not medical diagnoses or definitive assessments of employee mental health.</div>',unsafe_allow_html=True)

elif page=='Analyze Feedback':
    st.markdown('<div class="eyebrow">Live model demonstration</div>',unsafe_allow_html=True)
    st.markdown('# Analyze employee feedback')
    examples=['I am very happy with my achievement.','I am stressed and unhappy with my workload.','I feel frustrated but also excited about the new opportunity.','I am worried about my upcoming performance review.']
    choice=st.selectbox('Quick example',['Custom input']+examples)
    text=st.text_area('Feedback text','' if choice=='Custom input' else choice,height=150,placeholder='Type or paste feedback here...')
    if st.button('Analyze feedback',type='primary',use_container_width=True):
        if not text.strip(): st.error('Please enter some text.')
        else:
            try: r=process_text(text)
            except Exception as e: st.error(str(e)); r=None
            if r:
                emotion=r.get('emotion',r.get('primary_emotion','—')); sentiment=r.get('sentiment','—')
                conf=float(r.get('confidence',0)); second=r.get('second_best_emotion',r.get('second_best','—')); margin=float(r.get('confidence_margin',r.get('margin',0)))
                c1,c2=st.columns([.8,1.2])
                with c1:
                    st.markdown(f'<div class="card"><span class="pill">{sentiment}</span><h2>{str(emotion).title()}</h2><b>Confidence</b><br>{conf*100:.2f}%<br><br><b>Second best</b><br>{str(second).title()}<br><br><b>Margin</b><br>{margin*100:.2f}%</div>',unsafe_allow_html=True)
                with c2:
                    probs=r.get('probabilities',{})
                    if probs:
                        d=pd.DataFrame({'Emotion':[str(x).title() for x in probs],'Probability':[float(x)*100 for x in probs.values()]})
                        fig=px.bar(d.sort_values('Probability'),x='Probability',y='Emotion',orientation='h',text='Probability',title='Emotion probability distribution')
                        fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside'); fig.update_layout(height=390,plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',xaxis_title='Probability (%)')
                        st.plotly_chart(fig,use_container_width=True)
                st.markdown('<div class="notice">These are classifier softmax outputs, not scientifically calibrated probabilities.</div>',unsafe_allow_html=True)

elif page=='Model Comparison':
    st.markdown('<div class="eyebrow">Benchmark comparison</div>',unsafe_allow_html=True); st.markdown('# BERT vs DistilBERT')
    if model_df is None: st.error('model_comparison.csv not found.')
    else:
        st.dataframe(model_df.style.format({c:'{:.2%}' for c in ['Accuracy','Precision','Recall','Macro F1']}),use_container_width=True,hide_index=True)
        d=model_df.melt(id_vars='Model',value_vars=['Accuracy','Precision','Recall','Macro F1'],var_name='Metric',value_name='Score'); d.Score=d.Score*100
        fig=px.bar(d,x='Metric',y='Score',color='Model',barmode='group',text='Score',title='Independent held-out model comparison'); fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside'); fig.update_layout(yaxis_range=[0,100],plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="card"><span class="pill">SELECTED MODEL</span><h2>BERT</h2><p class="muted">Best Accuracy and Macro F1 on the independent held-out evaluation.</p></div>',unsafe_allow_html=True)

elif page=='ISEAR Evaluation':
    st.markdown('<div class="eyebrow">Independent benchmark</div>',unsafe_allow_html=True); st.markdown('# ISEAR evaluation')
    if isear_df is None: st.error('isear_model_comparison.csv not found.')
    else:
        c=st.columns(3); c[0].metric('Held-out samples','541'); c[1].metric('BERT accuracy','54.90%'); c[2].metric('BERT Macro F1','57.76%')
        st.dataframe(isear_df.style.format({c:'{:.2%}' for c in ['Accuracy','Precision','Recall','Macro F1','Average Confidence']}),use_container_width=True,hide_index=True)
        d=isear_df.melt(id_vars='Model',value_vars=['Accuracy','Precision','Recall','Macro F1'],var_name='Metric',value_name='Score'); d.Score=d.Score*100
        fig=px.bar(d,x='Metric',y='Score',color='Model',barmode='group',text='Score',title='ISEAR benchmark metrics'); fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside'); fig.update_layout(yaxis_range=[0,100],plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)'); st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div class="notice">ISEAR originally contains seven emotion categories. This evaluation uses the five categories supported by the current setup: joy, sadness, anger, fear and disgust. Shame and guilt are excluded.</div>',unsafe_allow_html=True)

elif page=='Validation Center':
    st.markdown('<div class="eyebrow">Evidence, not decoration</div>',unsafe_allow_html=True); st.markdown('# Validation center')
    checks=[('BERT integration','PASS','Saved transformer model'),('DistilBERT integration','PASS','Saved lightweight transformer model'),('Model evaluation','PASS','Accuracy / Precision / Recall / Macro F1'),('ISEAR validation','PASS','541 held-out samples'),('Confidence validation','PASS','Probability + margin analysis'),('Multi-label module','PASS','Threshold-based secondary-emotion logic'),('Pipeline integration','PASS','Text → preprocessing → VADER → BERT'),('Edge-case testing','PASS','13 tests; 0 unexpected failures'),('Code cleanup','PASS','Compilation + organization'),('Git repository','CLEAN','master synchronized with origin')]
    for name,status,detail in checks: st.markdown(f'<div class="card" style="padding:13px 18px"><b>{name}</b><span class="pill" style="float:right">{status}</span><br><span class="muted">{detail}</span></div>',unsafe_allow_html=True)
    st.success('Milestone 2 implementation status: validated and presentation-ready.')

else:
    st.markdown('<div class="eyebrow">Under the hood</div>',unsafe_allow_html=True); st.markdown('# Technical details')
    a,b=st.columns(2)
    with a: st.markdown('<div class="card"><h3>Models</h3>BERT — bert-base-uncased<br>DistilBERT — distilbert-base-uncased<br><br><h3>Emotion classes</h3>Joy · Sadness · Anger · Fear · Surprise · Disgust</div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><h3>Stack</h3>Python 3.10 · PyTorch · Transformers · VADER · Scikit-learn · Streamlit · Plotly</div>',unsafe_allow_html=True)
    st.code('Employee Feedback\n      ↓\nText Ingestion → Preprocessing\n      ↓\nVADER Sentiment + BERT/DistilBERT\n      ↓\nEmotion Probabilities\n      ↓\nPrimary Emotion + Confidence + Margin\n      ↓\nWellness Analysis Dashboard','text')
    st.markdown('<div class="notice">Use this prototype with appropriate consent and privacy controls. It is not a standalone system for diagnosing or making high-impact decisions about employees.</div>',unsafe_allow_html=True)

st.markdown('<p style="color:#6f716d;font-size:11px;margin-top:35px">AI Employee Wellness Management Platform · Milestone 2 · Internship Prototype</p>',unsafe_allow_html=True)
