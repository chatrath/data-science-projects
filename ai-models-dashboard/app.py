import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="AI Evolution: Open vs. Closed Models",
    page_icon="🤖",
    layout="wide"
)

# App Header and primary navigation
st.html("""
<style>
.ai-app-hero {
    display: flex; align-items: center; gap: 24px;
    padding: 30px 32px; margin-bottom: 22px;
    background: linear-gradient(115deg, #eff6ff 0%, #f0fdfa 62%, #ffffff 100%);
    border: 1px solid #dbeafe; border-radius: 24px;
    box-shadow: 0 8px 30px rgba(15, 23, 42, .04);
}
.ai-app-hero .hero-icon {
    flex: 0 0 64px; width: 64px; height: 64px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 19px; background: #0f172a; color: #67e8f9;
    box-shadow: 0 8px 18px rgba(15, 23, 42, .15);
}
.ai-app-hero h1 {
    color: #0f172a; font-size: clamp(25px, 2.5vw, 40px);
    letter-spacing: -.035em; line-height: 1.2; margin: 0 0 12px;
    font-weight: 750; padding: 0;
}
.ai-app-hero p { color: #475569; font-size: 16px; line-height: 1.65; margin: 0; max-width: 1000px; }
.stTabs > [role="tablist"] {
    display: flex; gap: 8px; width: fit-content; max-width: 100%;
    padding: 6px; height: auto; margin: 0 0 22px;
    border: 1px solid #dbe3ee; background: #f1f5f9; border-radius: 17px;
}
.stTabs > [role="tablist"] [role="tab"] {
    min-height: 50px; height: auto; padding: 12px 22px;
    border-radius: 12px; border: 1px solid transparent;
    color: #475569; background: transparent;
    transition: background .18s ease, color .18s ease, box-shadow .18s ease;
}
.stTabs > [role="tablist"] [role="tab"] p { font-size: 16px; font-weight: 600; }
.stTabs > [role="tablist"] [role="tab"]:hover { background: #e2e8f0; color: #0f172a; }
.stTabs > [role="tablist"] [role="tab"][aria-selected="true"] {
    background: #0f172a; color: #ffffff; border-color: #0f172a;
    box-shadow: 0 3px 10px rgba(15, 23, 42, .18);
}
.stTabs > [role="tablist"] [role="tab"]:focus-visible { outline: 3px solid #38bdf8; outline-offset: 3px; }
.stTabs > [role="tablist"] [data-baseweb="tab-highlight"],
.stTabs > [role="tablist"] [data-baseweb="tab-border"] { display: none; }
@media (max-width: 640px) {
    .ai-app-hero { gap: 14px; padding: 22px 18px; align-items: flex-start; border-radius: 18px; }
    .ai-app-hero .hero-icon { flex-basis: 44px; width: 44px; height: 44px; border-radius: 13px; }
    .ai-app-hero .hero-icon svg { width: 26px; height: 26px; }
    .stTabs > [role="tablist"] { width: 100%; }
    .stTabs > [role="tablist"] [role="tab"] { flex: 1; padding: 10px 12px; }
    .stTabs > [role="tablist"] [role="tab"] p { font-size: 14px; }
}
@media (prefers-reduced-motion: reduce) {
    .stTabs > [role="tablist"] [role="tab"] { transition: none; }
}
</style>
<header class="ai-app-hero">
    <div class="hero-icon" aria-hidden="true">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="12" r="9"/><ellipse cx="12" cy="12" rx="4" ry="9"/>
            <path d="M3 12h18M5 6.5h14M5 17.5h14"/>
        </svg>
    </div>
    <div>
        <h1>The Evolution &amp; Adoption of Open vs. Closed AI Models</h1>
        <p>Explore dated releases across open-weight and closed models, then compare deployment choices and the factors that shape their use.</p>
    </div>
</header>
""")

# Navigation Sidebar
tab1, tab2 = st.tabs(["📅 Models Evolution", "📊 Models Adoption"])

with tab1:
    st.header("The Generative AI Evolution Timeline")

    # Resolve relative to this script, independent of the terminal's folder.
    dataset_path = Path(__file__).resolve().parent / "timeline_data.csv"
    columns = ["date", "tool", "title", "kind", "text", "source"]
    try:
        timeline_df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        missing_columns = set(columns) - set(timeline_df.columns)
        if missing_columns:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing_columns)))
        timeline_df = timeline_df[columns].apply(lambda column: column.str.strip())
        if timeline_df.empty or (timeline_df == "").any().any():
            raise ValueError("Include at least one milestone and fill all six fields in every row.")
        dates = pd.to_datetime(timeline_df["date"], format="%Y-%m-%d", errors="raise")
        timeline_df["date"] = dates.dt.strftime("%Y-%m-%d")
        if not timeline_df["kind"].isin(["Closed", "Open-weight"]).all():
            raise ValueError("The kind column must contain Closed or Open-weight.")
        if not timeline_df["source"].str.match(r"https?://").all():
            raise ValueError("Source links must start with https:// or http://.")
        timeline_events = timeline_df.sort_values("date", kind="stable").to_dict("records")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load timeline_data.csv: {error}")
        st.stop()

    access = st.selectbox("Model access", ["All models", "Open-weight", "Closed"])
    available = timeline_df if access == "All models" else timeline_df[timeline_df["kind"] == access]
    families = sorted(available["tool"].unique())
    selected_families = st.multiselect("Model families", families, default=families, key=f"families_{access}")
    visible_df = available[available["tool"].isin(selected_families)]
    timeline_events = visible_df.sort_values("date", kind="stable").to_dict("records")
    range_start = timeline_df["date"].min()
    range_end = max(timeline_df["date"].max(), "2026-10-10")
    st.caption("Selected release milestones. Open-weight means weights are available; licenses and deployment options vary by model. Scroll horizontally to explore model families.")

    timeline_html = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<style>
:root{color-scheme:normal;background:transparent}html,body{height:auto!important;min-height:0!important;background:transparent}*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;color:#e2e8f0;background:transparent}
.shell{color-scheme:dark;overflow:hidden;padding:12px 14px 6px;border:1px solid #334155;border-radius:16px;background:linear-gradient(135deg,#111c30,#0b1220)}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button,select{background:#1e293b;color:#f1f5f9;border:1px solid #475569;border-radius:8px;padding:9px 13px;cursor:pointer;font:inherit}
button:hover{background:#334155}button:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #38bdf8;outline-offset:3px}
#play{background:#2563eb;border-color:#3b82f6}#clock{margin-left:auto;font-weight:700;color:#7dd3fc}
input[type=range]{width:100%;accent-color:#38bdf8;cursor:pointer}.ends{display:flex;justify-content:space-between;color:#94a3b8;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;margin-top:10px}
svg{width:100%;height:auto;display:block}svg text{font-family:system-ui,sans-serif}.milestone{cursor:pointer;transition:opacity .25s}.milestone:focus{outline:none}.milestone:focus circle{stroke:white;stroke-width:4}
.plot-scroll{overflow-x:auto;max-width:100%;border-radius:8px;scrollbar-color:#475569 #101827;scrollbar-width:thin}
.layout.has-detail{grid-template-columns:minmax(0,2fr) minmax(235px,1fr)}.card[hidden]{display:none}.card{align-self:start;background:#172236;border:1px solid #334155;border-radius:12px;padding:16px}
.eyebrow{font-size:12px;color:#94a3b8;text-transform:uppercase;letter-spacing:.09em}.card h3{margin:12px 0;font-size:23px}.card p{line-height:1.6;color:#cbd5e1}.card a{color:#7dd3fc}.badge{display:inline-block;font-size:12px;padding:4px 8px;border-radius:6px;background:#243449;margin-top:12px}
.legend{display:flex;gap:18px;font-size:12px;color:#cbd5e1;margin-top:4px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.fade{animation:fade .3s ease}@keyframes fade{from{opacity:.3;transform:translateY(5px)}to{opacity:1;transform:none}}
@media(max-width:720px){.layout,.layout.has-detail{grid-template-columns:1fr}.shell{color-scheme:dark;overflow:hidden;padding:12px}.card{min-height:0}#clock{margin-left:0}}
@media(prefers-reduced-motion:reduce){.fade{animation:none}.milestone{transition:none}}
</style></head><body><main class="shell">
<div class="controls"><button id="play" aria-label="Play timeline">▶ Play</button><button id="reset">↺ Reset</button><label for="speed">Speed</label><select id="speed"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></select><output id="clock"></output></div>
<label for="scrub" class="eyebrow">Explore time</label><input id="scrub" aria-label="Timeline date" type="range" min="0" max="1000" value="0"><div class="ends"><span id="range-start"></span><span id="range-end"></span></div>
<div class="layout"><div><div class="plot-scroll" tabindex="0" aria-label="Scrollable model timeline"><svg id="plot" viewBox="0 0 780 460" role="group" aria-label="Interactive release timeline: tools on X-axis, time increasing down Y-axis"></svg></div><div class="legend"><span><i class="dot" style="background:#60a5fa"></i>Closed</span><span><i class="dot" style="background:#34d399"></i>Open-weight</span></div></div><article id="detail" class="card" aria-live="polite" hidden></article></div>
</main><script>
const events = __EVENTS__;
const tools = [...new Set(events.map(event=>event.tool))].sort();
const start = Date.parse('__START__T00:00:00Z'), end = Math.max(Date.parse('__END__T00:00:00Z'),start+86400000);
const svg = document.getElementById('plot'), slider = document.getElementById('scrub'), detail = document.getElementById('detail');
const plotScroll=svg.closest('.plot-scroll');
const minChartWidth=Math.max(780,120+tools.length*240);
let chartWidth=Math.max(minChartWidth,plotScroll.clientWidth);
svg.setAttribute('viewBox',`0 0 ${chartWidth} 460`);svg.style.width=`${chartWidth}px`;svg.style.height='460px';
const ns='http://www.w3.org/2000/svg';const chartTop=36,bottom=410,left=100;let right=chartWidth-20;
const y = t => chartTop+(t-start)/(end-start)*(bottom-chartTop);
const x = tool => left+(right-left)*(tools.indexOf(tool)+1)/(tools.length+1);
const dateLabel = t => new Date(t).toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
function node(tag,attrs,parent=svg){const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);parent.appendChild(el);return el;}
function label(value,attrs,parent=svg){const el=node('text',attrs,parent);el.textContent=value;return el;}
const monthLabel=t=>new Date(t).toLocaleDateString('en-GB',{month:'short',year:'numeric',timeZone:'UTC'});
document.getElementById('range-start').textContent=monthLabel(start);
document.getElementById('range-end').textContent=monthLabel(end);
const ticks=[start,end];
for(let year=new Date(start).getUTCFullYear();year<=new Date(end).getUTCFullYear();year++){
for(const month of [0,6]){const t=Date.UTC(year,month,1);if(t>start&&t<end)ticks.push(t);}}
// Keep endpoint labels and omit intermediate ticks that crowd nearby dates.
const dateLabelGap=30;
let lastTickY=-Infinity;
const spacedTicks=ticks.sort((a,b)=>a-b).filter(t=>{
if(t===start||t===end){lastTickY=y(t);return true;}
if(y(t)-lastTickY<dateLabelGap||y(end)-y(t)<dateLabelGap)return false;
lastTickY=y(t);return true;
});
for(const t of spacedTicks){node('line',{x1:left,y1:y(t),x2:right,y2:y(t),stroke:'#26354a','data-full-width':'true'});
label(new Date(t).toLocaleDateString('en-GB',{month:'short',year:'numeric',timeZone:'UTC'}),{x:left-12,y:y(t)+4,'text-anchor':'end',fill:'#94a3b8','font-size':12});}
label('Time ↓',{x:8,y:18,fill:'#cbd5e1','font-size':13});label('AI tool / model family',{x:chartWidth/2,y:452,'text-anchor':'middle',fill:'#cbd5e1','font-size':13,id:'axis-caption'});
tools.forEach(tool=>{node('line',{x1:x(tool),y1:chartTop,x2:x(tool),y2:bottom,stroke:'#334155','stroke-dasharray':'4 5','data-tool-lane':tool});label(tool,{x:x(tool),y:432,'text-anchor':'middle',fill:'#e2e8f0','font-size':14,'font-weight':600,'data-tool-label':tool});});
const cursor=node('line',{x1:left,y1:chartTop,x2:right,y2:chartTop,stroke:'#38bdf8','stroke-width':2,'stroke-dasharray':'7 4','data-full-width':'true'});
// The current date is shown above the chart to keep milestone labels clear.
let selected=-1,playing=false,last=0,position=0,rendered=-2;
events.forEach(event=>{event.time=Date.parse(event.date+'T00:00:00Z');event.labelY=y(event.time);});
// Keep labels apart while dots retain their exact release dates.
tools.forEach(tool=>{let nextY=bottom+8;events.filter(event=>event.tool===tool).slice().reverse().forEach(event=>{event.labelY=Math.min(y(event.time),nextY-24);nextY=event.labelY;});});
const marks=events.map((event,i)=>{const group=node('g',{class:'milestone',transform:`translate(${x(event.tool)} 0)`,role:'button',tabindex:0,'aria-label':event.title+' — '+event.date});
node('circle',{cx:0,cy:y(event.time),r:8,fill:event.kind==='Closed'?'#60a5fa':'#34d399',stroke:'#101827','stroke-width':2},group);
if(Math.abs(event.labelY-y(event.time))>2)node('line',{x1:8,y1:y(event.time),x2:14,y2:event.labelY,stroke:'#64748b','stroke-width':1},group);
label(event.title,{x:14,y:event.labelY+4,fill:'#e2e8f0','font-size':12,stroke:'#101827','stroke-width':4,'stroke-linejoin':'round','paint-order':'stroke'},group);
const title=node('title',{},group);title.textContent=event.title+' — '+dateLabel(event.time);
const choose=()=>{pause();selected=i;render();};group.addEventListener('click',e=>{e.stopPropagation();choose();});group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});return group;});
function showDetail(i){if(rendered===i)return;rendered=i;detail.replaceChildren();detail.classList.remove('fade');void detail.offsetWidth;detail.classList.add('fade');
detail.hidden=i<0;detail.closest('.layout').classList.toggle('has-detail',i>=0);if(i<0)return;
const e=events[i];
const header=document.createElement('div');header.className='eyebrow';header.textContent=dateLabel(e.time)+' · '+e.tool;
const badge=document.createElement('span');badge.className='badge';badge.textContent=e.kind;
const title=document.createElement('h3');title.textContent=e.title;const body=document.createElement('p');body.textContent=e.text;
const source=document.createElement('a');source.href=e.source;source.target='_blank';source.rel='noopener noreferrer';source.textContent='Read release announcement ↗';detail.append(header,badge,title,body,source);}
function render(){const now=start+position*(end-start);slider.value=Math.round(position*1000);slider.setAttribute('aria-valuetext',dateLabel(now));document.getElementById('clock').textContent=dateLabel(now);
cursor.setAttribute('y1',y(now));cursor.setAttribute('y2',y(now));
marks.forEach((mark,i)=>{const visible=events[i].time<=now;mark.style.opacity=visible?'1':'0';mark.style.pointerEvents=visible?'auto':'none';mark.setAttribute('tabindex',visible?'0':'-1');mark.setAttribute('aria-hidden',String(!visible));mark.querySelector('circle').setAttribute('r',i===selected?11:8);});showDetail(selected);}
function advance(p){position=Math.max(0,Math.min(1,p));if(selected>=0&&events[selected].time>start+position*(end-start))selected=-1;render();}
function pause(){playing=false;document.getElementById('play').textContent='▶ Play';document.getElementById('play').setAttribute('aria-label','Play timeline');}
function tick(timestamp){if(!playing)return;const dt=last?timestamp-last:0;last=timestamp;advance(position+dt/24000*Number(document.getElementById('speed').value));if(position>=1){pause();return;}requestAnimationFrame(tick);}
document.getElementById('play').onclick=()=>{if(playing){pause();return;}if(position>=1)advance(0);playing=true;last=0;document.getElementById('play').textContent='Ⅱ Pause';document.getElementById('play').setAttribute('aria-label','Pause timeline');requestAnimationFrame(tick);};
document.getElementById('reset').onclick=()=>{pause();selected=-1;advance(0);};slider.oninput=()=>{pause();advance(Number(slider.value)/1000);};
svg.addEventListener('click',event=>{const point=svg.createSVGPoint();point.x=event.clientX;point.y=event.clientY;const local=point.matrixTransform(svg.getScreenCTM().inverse());pause();advance((local.y-chartTop)/(bottom-chartTop));});
function dismissDetail(){if(selected<0)return;selected=-1;render();}
// Milestone clicks stop propagation; every other click dismisses the story.
document.addEventListener('click',dismissDetail);
// The timeline lives in an iframe: clicks elsewhere in the app move focus out.
window.addEventListener('blur',dismissDetail);
// Fit the grid to the graph area, including when the details panel opens/closes.
new ResizeObserver(entries=>{
chartWidth=Math.max(minChartWidth,entries[0].contentRect.width);right=chartWidth-20;
svg.setAttribute('viewBox',`0 0 ${chartWidth} 460`);svg.style.width=`${chartWidth}px`;
svg.querySelectorAll('[data-full-width]').forEach(line=>line.setAttribute('x2',right));
document.getElementById('axis-caption').setAttribute('x',chartWidth/2);
svg.querySelectorAll('[data-tool-lane]').forEach(line=>{const position=x(line.getAttribute('data-tool-lane'));line.setAttribute('x1',position);line.setAttribute('x2',position);});
svg.querySelectorAll('[data-tool-label]').forEach(label=>label.setAttribute('x',x(label.getAttribute('data-tool-label'))));
marks.forEach((mark,i)=>mark.setAttribute('transform',`translate(${x(events[i].tool)} 0)`));
}).observe(plotScroll);
document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();dismissDetail();}});advance(0);
</script></body></html>
"""
    import json
    if timeline_events:
        st.iframe(
            timeline_html.replace("__EVENTS__", json.dumps(timeline_events).replace("<", "\\u003c"))
            .replace("__START__", range_start).replace("__END__", range_end),
            height="content",
            alt="Interactive AI evolution timeline with playback controls",
        )
    else:
        st.info("Select at least one model family to display its timeline.")

with tab2:
    st.header("AI Adoption Numbers Across Industries")
    st.markdown("Published survey findings on open-model adoption and enterprise model use.")
    survey_path = Path(__file__).resolve().parent / "adoption_survey_data.csv"
    try:
        survey_df = pd.read_csv(survey_path, encoding="utf-8-sig")
        required = {"survey", "sector", "model_category", "percentage", "sample_size", "survey_start", "survey_end", "source", "notes"}
        if not required.issubset(survey_df.columns):
            raise ValueError("Missing survey dataset columns.")
        survey_df["percentage"] = pd.to_numeric(survey_df["percentage"], errors="raise")
        survey_df["sample_size"] = pd.to_numeric(survey_df["sample_size"], errors="raise")
        if survey_df.empty or not survey_df["percentage"].between(0, 100).all():
            raise ValueError("Include survey observations with percentages between 0 and 100.")
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not load adoption_survey_data.csv: {error}")
        st.stop()

    from adoption_insights import server_key, context_for, narration, insights
    ai_key = server_key()

    view = st.radio("Survey view", ["Sector adoption (McKinsey)", "Enterprise API market share (Menlo)"], horizontal=True)
    summary_panel = st.empty()
    sector_view = view == "Sector adoption (McKinsey)"
    if sector_view:
        chart_df = survey_df[survey_df["survey"] == "McKinsey Open Source AI Survey"].sort_values("percentage", ascending=False)
        st.caption("McKinsey, Mozilla Foundation and Patrick J. McGovern Foundation • 703 respondents across 41 countries • Survey: 9 Dec 2024–24 Jan 2025 • Published April 2025.")
        base = alt.Chart(chart_df).encode(
            x=alt.X("percentage:Q", title="Respondents reporting regular use (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("sector:N", title=None, sort=chart_df["sector"].tolist(), axis=alt.Axis(labelLimit=290)),
            tooltip=[alt.Tooltip("sector:N", title="Sector"), alt.Tooltip("percentage:Q", title="Open-model use (%)", format=".0f"), alt.Tooltip("sample_size:Q", title="Sector respondents", format=".0f")]
        )
        bars = base.mark_bar(color="#34d399", cornerRadiusEnd=3)
        chart_height = 320
    else:
        chart_df = survey_df[survey_df["survey"] == "Menlo State of Generative AI 2025"]
        st.caption("Menlo Ventures • 495 US enterprise AI decision-makers • Survey: 7–25 Nov 2025 • Estimates as of December 2025.")
        base = alt.Chart(chart_df).encode(
            x=alt.X("percentage:Q", title="Estimated enterprise LLM API market share (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("model_category:N", title=None, sort=["Closed", "Open-weight / open-source"], axis=alt.Axis(labelLimit=290)),
            tooltip=[alt.Tooltip("model_category:N", title="Model access"), alt.Tooltip("percentage:Q", title="Market share (%)", format=".0f"), alt.Tooltip("notes:N", title="Measure")]
        )
        bars = base.mark_bar(cornerRadiusEnd=3).encode(color=alt.Color("model_category:N", title=None, scale=alt.Scale(domain=["Closed", "Open-weight / open-source"], range=["#60a5fa", "#34d399"]), legend=None))
        chart_height = 150

    labels = base.mark_text(align="left", dx=7, color="#e2e8f0", fontSize=12).encode(text=alt.Text("percentage:Q", format=".0f"))
    chart = (bars + labels).properties(height=chart_height, background="transparent").configure_view(stroke=None).configure_axis(
        labelColor="#cbd5e1", titleColor="#e2e8f0", gridColor="#26354a",
        domainColor="#334155", tickColor="#334155", labelFont="system-ui",
        titleFont="system-ui", labelFontSize=12, titleFontSize=13,
        labelPadding=8, titlePadding=14
    ).configure_axisY(grid=False)

    ai_context = context_for(chart_df, view)
    ai_narration, ai_signature = narration(summary_panel, ai_context, ai_key)

    st.html("""
    <style>
    .st-key-adoption-chart {
        background: linear-gradient(135deg, #111c30, #0b1220);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 12px 14px 6px;
        margin-top: 44px;
        overflow: visible;
    }
    .st-key-adoption-chart [data-testid="stElementToolbar"] {
        top: -54px !important;
        right: 0 !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        height: auto !important;
    }
    .st-key-adoption-chart [data-testid="stElementToolbarButtonContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .st-key-adoption-chart [data-testid="stElementToolbarButton"]:not(:has(button[aria-label="Download as PNG"])) {
        display: none;
    }
    .st-key-adoption-chart button[aria-label="Download as PNG"] {
        width: auto;
        height: 34px;
        padding: 6px 12px;
        gap: 6px;
        background: #1e293b;
        color: #f1f5f9;
        border: 1px solid #475569;
        border-radius: 8px;
    }
    .st-key-adoption-chart button[aria-label="Download as PNG"]::after {
        content: "Download as PNG";
        font-size: 14px;
        white-space: nowrap;
    }
    </style>
    """)
    with st.container(key="adoption-chart"):
        st.altair_chart(chart, width="stretch", theme=None)
    if sector_view:
        st.caption(f"Source: [McKinsey report, Exhibit 3, page 6]({chart_df.iloc[0]['source']}). Sector sample sizes are shown on hover. This is a survey of AI-experienced participants, not a census of all businesses.")
    else:
        st.caption(f"Source: [Menlo Ventures 2025 enterprise report]({chart_df.iloc[0]['source']}). Survey inputs are weighted by application scale and triangulated with financial information. Self-hosted usage is not captured by an API market-share comparison.")
    st.download_button("Download survey dataset (CSV)", survey_path.read_bytes(), file_name="adoption_survey_data.csv", mime="text/csv")

    st.divider()
    insights(ai_context, ai_key, ai_narration, ai_signature)

    from model_comparison import render_comparison
    render_comparison(timeline_df, Path(__file__).resolve().parent)

# Footer
st.markdown("---")
st.caption("Path C Vibe-Coded App • Built with Streamlit & AI Guidance")


