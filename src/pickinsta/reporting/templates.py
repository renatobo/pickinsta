"""Static HTML document templates for pickinsta galleries."""

SHARED_HEADER_CSS = """\
  .brand { color: var(--accent); font-size: 1.35rem; font-weight: 700; letter-spacing: -.02em; }
  .page-title { font-size: 1.2rem; font-weight: 600; letter-spacing: -.02em; }
  .gh-link { color: var(--text-dim); transition: color .15s; }
  .gh-link:hover { color: var(--text); }
  .breadcrumb {
    display: inline-flex; align-items: baseline; flex-wrap: wrap;
    gap: .15rem; font-size: .95rem; color: var(--text-dim);
  }
  .breadcrumb a { color: var(--text-dim); text-decoration: none; }
  .breadcrumb a:hover { color: var(--text); text-decoration: underline; }
  .breadcrumb .sep { margin: 0 .3rem; }\
"""

DEDUP_GALLERY_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg: #0f0f0f; --surface: #1a1a1a; --surface2: #242424; --border: #333;
    --text: #e0e0e0; --text-dim: #888; --accent: #d32f2f; --accent-dim: #b71c1c; --gold: #ffd54f; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5; }}
  header {{ padding: 1rem 2rem; border-bottom: 1px solid var(--border); background: var(--surface); }}
  header h1 {{ font-size: 1rem; font-weight: 600; display: flex; align-items: center; flex-wrap: wrap; gap: .45rem; }}
{shared_header_css}
  .info {{ padding: .5rem 2rem; font-size: .8rem; color: var(--text-dim); border-bottom: 1px solid var(--border); background: var(--surface); }}
  .layout {{ display: flex; height: calc(100vh - 80px); }}
  .grid-panel {{ flex: 1; overflow-y: auto; padding: .75rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: .5rem; }}
  .tile {{ position: relative; cursor: pointer; border-radius: 6px; overflow: hidden;
    border: 2px solid transparent; transition: border-color .15s, transform .15s; }}
  .tile:hover {{ border-color: var(--accent); transform: scale(1.02); }}
  .tile.active {{ border-color: var(--gold); }}
  .tile img {{ width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block; }}
  .tile-burst {{ position: absolute; bottom: 4px; left: 4px; background: rgba(0,0,0,.75);
    color: #81d4fa; font-size: .6rem; padding: 1px 5px; border-radius: 4px; }}
  .detail-panel {{ width: 400px; min-width: 400px; overflow-y: auto; background: var(--surface);
    border-left: 1px solid var(--border); padding: 1rem; display: none; }}
  .detail-panel.open {{ display: flex; flex-direction: column; }}
  .detail-panel h2 {{ font-size: .9rem; font-weight: 600; margin-bottom: .5rem; word-break: break-all; }}
  .version-tabs {{ display: flex; gap: 3px; margin-bottom: .5rem; }}
  .version-tabs button {{ flex: 1; padding: .3rem; font-size: .7rem; background: var(--surface2);
    color: var(--text-dim); border: 1px solid var(--border); border-radius: 4px; cursor: pointer; }}
  .version-tabs button.active {{ background: var(--accent-dim); color: #fff; border-color: var(--accent); }}
  .preview-img {{ width: 100%; border-radius: 4px; background: var(--surface2); }}
  .exif-row {{ display: flex; flex-wrap: wrap; gap: .3rem .75rem; font-size: .75rem;
    color: var(--text-dim); margin-top: .5rem; }}
  .exif-row span {{ white-space: nowrap; }}
  .exif-val {{ color: var(--text); font-weight: 500; }}

  @media (max-width: 900px) {{
    body {{
      padding: 0;
    }}

    header {{
      padding: .75rem 1rem;
    }}

    header h1 {{
      font-size: 1.1rem;
      flex-wrap: wrap;
      gap: .35rem;
    }}

    .info {{
      padding: .5rem 1rem;
    }}

    .layout {{
      flex-direction: column;
      height: auto;
    }}

    .grid-panel {{
      max-height: 48vh;
      padding: .75rem 1rem 1rem;
    }}

    .grid {{
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    }}

    .detail-panel {{
      width: 100%;
      min-width: 0;
      border-left: 0;
      border-top: 1px solid var(--border);
      max-height: none;
    }}

    .version-tabs {{
      flex-wrap: wrap;
    }}

    .version-tabs button {{
      min-width: 0;
    }}

    .preview-img {{
      max-height: 42vh;
    }}
  }}
</style>
</head>
<body>
<header><h1><span class="brand">pickinsta</span> <span class="page-title">{title} &mdash; dedup</span></h1></header>
<div class="info">{count} unique images</div>
<div class="layout">
  <div class="grid-panel"><div class="grid" id="grid">{tiles}</div></div>
  <div class="detail-panel" id="detail">
    <h2 id="detail-title"></h2>
    <div class="version-tabs" id="version-tabs"></div>
    <a id="preview-link" href="" target="_blank"><img class="preview-img" id="preview-img" src="" alt=""></a>
    <div class="exif-row" id="detail-exif"></div>
  </div>
</div>
<script>
const DATA = {json_data};
function sel(i) {{
  document.querySelectorAll('.tile').forEach(t => t.classList.remove('active'));
  const tile = document.querySelector(`.tile[data-idx="${{i}}"]`);
  if (tile) tile.classList.add('active');
  document.getElementById('detail').classList.add('open');
  const d = DATA[i];
  document.getElementById('detail-title').textContent = d.filename;
  const tabs = document.getElementById('version-tabs');
  const img = document.getElementById('preview-img');
  const link = document.getElementById('preview-link');
  const versions = [['Full', d.full], ['HD', d.hd], ['Cropped', d.cropped]].filter(v => v[1]);
  tabs.innerHTML = '';
  versions.forEach(([label, file], j) => {{
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.onclick = () => {{ img.src = file; link.href = file;
      tabs.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active'); }};
    if (j === 0) btn.classList.add('active');
    tabs.appendChild(btn);
  }});
  if (versions.length) {{ img.src = versions[0][1]; link.href = versions[0][1]; }}
  const exifDiv = document.getElementById('detail-exif');
  if (d.exif && Object.keys(d.exif).length) {{
    exifDiv.replaceChildren();
    ['camera','lens','focal','aperture','shutter','iso','date'].forEach(k => {{
      if (!d.exif[k]) return;
      const wrapper = document.createElement('span');
      const value = document.createElement('span');
      value.className = 'exif-val';
      value.textContent = String(d.exif[k]);
      wrapper.appendChild(value);
      exifDiv.appendChild(wrapper);
    }});
  }} else {{ exifDiv.innerHTML = ''; }}
}}
if (DATA.length) sel(0);
</script>
</body>
</html>
"""

GALLERY_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #080808;
    --surface: #0d0d0d;
    --surface2: #141414;
    --surface3: #1c1c1c;
    --border: #1f1f1f;
    --border2: #2a2a2a;
    --text: #bdbdbd;
    --text-bright: #e8e8e8;
    --text-dim: #424242;
    --accent: #c8251d;
    --accent-dim: #7a1510;
    --accent-glow: rgba(200,37,29,.18);
    --gold: #d4a73a;
    --gold-dim: rgba(212,167,58,.15);
    --cyan: #4fc3c8;
    --panel-w: 480px;
    --header-h: 48px;
    --bar-h: 36px;
    --font-mono: 'DM Mono', 'SF Mono', 'Fira Code', monospace;
    --font-display: 'Bebas Neue', 'Impact', sans-serif;
    --font-editorial: 'Fraunces', Georgia, serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ height: 100%; overflow: hidden; }}
  body {{
    font-family: var(--font-mono);
    background: var(--bg); color: var(--text);
    line-height: 1.5; font-size: 13px;
  }}

  /* ── Header ── */
  header {{
    height: var(--header-h);
    padding: 0 1.25rem;
    display: flex; align-items: center;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    position: relative; z-index: 10;
    gap: .75rem;
  }}
  header::after {{
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, var(--accent) 0%, transparent 60%);
  }}
  header h1 {{
    display: flex; align-items: baseline; flex-wrap: wrap;
    gap: .4rem; flex: 1;
  }}
{shared_header_css}
  .brand {{
    font-family: var(--font-editorial);
    font-size: 1.25rem; font-weight: 600;
    color: var(--accent); letter-spacing: -.01em;
    line-height: 1;
  }}
  .page-title {{
    font-family: var(--font-mono);
    font-size: .7rem; font-weight: 400;
    color: var(--text-dim); letter-spacing: .06em;
    text-transform: uppercase;
  }}
  .gh-link {{
    color: var(--text-dim); transition: color .2s; flex-shrink: 0;
    display: flex; align-items: center;
  }}
  .gh-link:hover {{ color: var(--text); }}
  .breadcrumb {{
    display: inline-flex; align-items: baseline; flex-wrap: wrap;
    gap: 0; font-size: .7rem; color: var(--text-dim);
    letter-spacing: .04em; text-transform: uppercase;
  }}
  .breadcrumb a {{ color: var(--text-dim); text-decoration: none; }}
  .breadcrumb a:hover {{ color: var(--text); }}
  .breadcrumb .sep {{ margin: 0 .2rem; opacity: .4; }}

  /* ── Stats bar ── */
  .stats-bar {{
    height: var(--bar-h);
    display: flex; align-items: stretch;
    background: var(--surface); border-bottom: 1px solid var(--border);
    overflow-x: auto; overflow-y: hidden;
  }}
  .stat {{
    display: flex; flex-direction: column; justify-content: center;
    padding: 0 1.25rem; border-right: 1px solid var(--border);
    white-space: nowrap; flex-shrink: 0;
  }}
  .stat-label {{
    font-size: .55rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .1em; line-height: 1;
  }}
  .stat-value {{
    font-family: var(--font-display);
    font-size: .95rem; color: var(--text-bright);
    line-height: 1.1; letter-spacing: .02em;
  }}

  /* ── Layout ── */
  .layout {{
    display: flex;
    height: calc(100vh - var(--header-h) - var(--bar-h));
  }}

  /* ── Grid panel ── */
  .grid-panel {{
    flex: 1; overflow-y: auto; padding: .6rem;
    position: relative;
  }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: .4rem;
  }}

  /* ── Tile ── */
  .tile {{
    position: relative; cursor: pointer; overflow: hidden;
    border-radius: 3px;
    outline: 1px solid var(--border);
    outline-offset: 0;
    transition: outline-color .15s, transform .15s;
  }}
  .tile:hover {{ outline-color: var(--accent); transform: scale(1.025); z-index: 2; }}
  .tile.active {{ outline: 2px solid var(--gold); outline-offset: 0; }}
  .tile img {{
    width: 100%; aspect-ratio: 3/4; object-fit: cover; display: block;
  }}
  .tile-overlay {{
    position: absolute; inset: 0;
    background: linear-gradient(
      to bottom,
      rgba(0,0,0,.55) 0%,
      transparent 35%,
      transparent 65%,
      rgba(0,0,0,.65) 100%
    );
    pointer-events: none;
  }}
  .tile-rank {{
    position: absolute; top: 4px; left: 5px;
    font-family: var(--font-display);
    font-size: 1.4rem; line-height: 1;
    color: var(--gold);
    text-shadow: 0 1px 4px rgba(0,0,0,.8);
  }}
  .tile-score {{
    position: absolute; bottom: 4px; right: 5px;
    font-family: var(--font-mono); font-size: .58rem;
    color: rgba(255,255,255,.7);
    text-shadow: 0 1px 3px rgba(0,0,0,.9);
  }}
  .tile-uncertain {{
    position: absolute; top: 4px; right: 4px;
    color: #e67e22; opacity: .9;
    filter: drop-shadow(0 1px 2px rgba(0,0,0,.8));
  }}
  .tile-burst {{
    position: absolute; bottom: 4px; left: 5px;
    font-family: var(--font-mono); font-size: .55rem;
    color: var(--cyan);
    text-shadow: 0 1px 3px rgba(0,0,0,.9);
  }}

  /* ── Detail panel ── */
  .detail-panel {{
    width: var(--panel-w); min-width: var(--panel-w);
    overflow-y: auto; background: var(--surface);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column;
    transform: translateX(100%);
    transition: transform .25s cubic-bezier(.4,0,.2,1);
  }}
  .detail-panel.open {{
    transform: translateX(0);
  }}
  .panel-header {{
    padding: .75rem 1rem .5rem;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    position: sticky; top: 0; z-index: 5;
    background: var(--surface);
  }}
  .panel-header h2 {{
    font-family: var(--font-mono); font-size: .72rem;
    font-weight: 400; color: var(--text-dim);
    letter-spacing: .04em; text-transform: uppercase;
    word-break: break-all; margin-bottom: .3rem;
  }}
  .panel-header h2 strong {{
    font-family: var(--font-display);
    font-size: 1.3rem; color: var(--gold);
    letter-spacing: .04em; font-weight: 400;
    vertical-align: baseline; margin-right: .25rem;
  }}
  .version-tabs {{
    display: flex; gap: 2px;
  }}
  .version-tabs button {{
    flex: 1; padding: .2rem .4rem; font-size: .6rem;
    font-family: var(--font-mono); letter-spacing: .08em;
    text-transform: uppercase;
    background: var(--surface2); color: var(--text-dim);
    border: 1px solid var(--border2); border-radius: 2px;
    cursor: pointer; transition: background .15s, color .15s;
  }}
  .version-tabs button.active {{
    background: var(--accent-dim); color: #fff; border-color: var(--accent);
  }}
  .version-tabs button:hover:not(.active) {{
    background: var(--surface3); color: var(--text);
  }}

  /* ── Panel body ── */
  .panel-body {{
    flex: 1; display: flex; flex-direction: column;
    overflow-y: auto;
  }}
  .preview-row {{
    display: flex; gap: .5rem; padding: .6rem 1rem;
    border-bottom: 1px solid var(--border);
  }}
  .preview-col {{
    flex: 1; display: flex; flex-direction: column; gap: .3rem;
  }}
  .preview-col-label {{
    font-size: .55rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--text-dim);
  }}
  .preview-img-wrap img {{
    width: 100%; object-fit: contain;
    border-radius: 2px; background: var(--surface2);
    display: block;
  }}

  /* ── Info sections ── */
  .info-section {{
    padding: .6rem 1rem; border-bottom: 1px solid var(--border);
  }}
  .section-label {{
    font-size: .55rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--text-dim); margin-bottom: .4rem;
  }}
  .one-line {{
    font-family: var(--font-editorial);
    font-style: italic; color: var(--text);
    font-size: .85rem; line-height: 1.5;
    font-weight: 300;
  }}
  .exif-chips {{
    display: flex; flex-wrap: wrap; gap: .25rem;
  }}
  .exif-chip {{
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 2px; padding: .15rem .45rem;
    font-size: .65rem; color: var(--text);
    white-space: nowrap;
  }}
  .yolo-grid {{
    display: grid; grid-template-columns: auto 1fr;

    gap: .15rem .75rem; font-size: .68rem;
  }}
  .yolo-grid dt {{ color: var(--text-dim); letter-spacing: .05em; text-transform: uppercase; font-size: .58rem; }}
  .yolo-grid dd {{ font-weight: 500; color: var(--text-bright); }}

  /* ── Score gauges ── */
  .score-group-label {{
    font-size: .55rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--text-dim); margin-bottom: .5rem;
  }}
  .score-row {{
    display: grid;
    grid-template-columns: 80px 1fr 36px;
    align-items: center; gap: .5rem;
    margin-bottom: .35rem;
  }}
  .score-label {{
    font-size: .6rem; color: var(--text-dim);
    text-align: right; letter-spacing: .03em;
    text-transform: uppercase; white-space: nowrap;
  }}
  .score-track {{
    position: relative; height: 2px;
    background: var(--surface3);
  }}
  .score-fill {{
    position: absolute; top: 0; left: 0; bottom: 0;
    background: var(--accent);
    transition: width .4s cubic-bezier(.4,0,.2,1);
  }}
  .score-fill.highlight {{
    background: linear-gradient(90deg, var(--accent-dim), var(--accent));
    box-shadow: 0 0 6px var(--accent-glow);
  }}
  .score-dot {{
    position: absolute; top: 50%; right: 0;
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); border: 1px solid var(--bg);
    transform: translate(50%, -50%);
    transition: right .4s cubic-bezier(.4,0,.2,1);
  }}
  .score-val {{
    font-family: var(--font-display);
    font-size: .9rem; color: var(--text-bright);
    letter-spacing: .02em; text-align: right;
  }}
  .score-divider {{
    height: 1px; background: var(--border); margin: .4rem 0 .6rem;
  }}

  /* ── Burst info ── */
  .burst-info {{
    display: flex; gap: .75rem; flex-wrap: wrap;
  }}
  .burst-chip {{
    font-size: .68rem; color: var(--cyan);
  }}
  .burst-chip span {{ color: var(--text-bright); }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border2); border-radius: 2px; }}
  .exif-row span {{ white-space: nowrap; }}
  .exif-val {{ color: var(--text); font-weight: 500; }}
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}

  @media (max-width: 900px) {{
    body {{
      padding: 0;
    }}

    header {{
      padding: .75rem 1rem;
    }}

    header h1 {{
      font-size: 1.1rem;
      flex-wrap: wrap;
      gap: .35rem;
    }}

    .info {{
      padding: .5rem 1rem;
    }}

    .layout {{
      flex-direction: column;
      height: auto;
    }}

    .grid-panel {{
      max-height: 48vh;
      padding: .75rem 1rem 1rem;
    }}

    .grid {{
      grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    }}

    .detail-panel {{
      width: 100%;
      min-width: 0;
      border-left: 0;
      border-top: 1px solid var(--border);
      max-height: none;
    }}

    .preview-row {{
      flex-direction: column;
    }}

    .preview-col {{
      width: 100%;
    }}

    .version-tabs {{
      flex-wrap: wrap;
    }}

    .version-tabs button {{
      min-width: 0;
    }}

    .preview-img-wrap img {{
      max-height: 42vh;
    }}

    .score-label {{
      width: 74px;
    }}
  }}
</style>
</head>
<body>
<header>
  <h1>
    <a class="gh-link" href="https://github.com/renatobo/pickinsta" target="_blank" title="pickinsta on GitHub">
      <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
    </a>
    <span class="brand">pickinsta</span>
    {breadcrumb}<span class="page-title">{title}</span>
  </h1>
</header>
<div class="stats-bar" id="stats-bar">
  {summary_stats}
</div>
<div class="layout">
  <div class="grid-panel">
    <div class="grid" id="grid">
      {tiles}
    </div>
  </div>
  <div class="detail-panel" id="detail">
    <div class="panel-header">
      <h2 id="detail-title"></h2>
      <div class="version-tabs" id="version-tabs"></div>
    </div>
    <div class="panel-body">
      <div class="preview-row">
        <div class="preview-col">
          <div class="preview-col-label">Preview</div>
          <div class="preview-img-wrap">
            <a id="version-link" href="" target="_blank"><img id="version-img" src="" alt=""></a>
          </div>
        </div>
        <div class="preview-col" id="yolo-col" style="display:none">
          <div class="preview-col-label">YOLO</div>
          <div class="preview-img-wrap">
            <img id="yolo-img" src="" alt="">
          </div>
          <div id="detail-yolo" style="margin-top:.5rem"></div>
        </div>
      </div>
      <div class="info-section" id="ai-section">
        <div class="section-label">AI Assessment</div>
        <p class="one-line" id="detail-oneline"></p>
      </div>
      <div class="info-section">
        <div class="section-label">Scores</div>
        <div id="detail-scores"></div>
      </div>
      <div class="info-section" id="exif-section" style="display:none">
        <div class="section-label">EXIF</div>
        <div class="exif-chips" id="detail-exif"></div>
      </div>
      <div class="info-section" id="burst-section" style="display:none">
        <div class="section-label">Burst</div>
        <div class="burst-info" id="detail-burst"></div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = {json_data};
function selectImage(idx) {{
  document.querySelectorAll('.tile').forEach(t => t.classList.remove('active'));
  const tile = document.querySelector(`.tile[data-idx="${{idx}}"]`);
  if (tile) tile.classList.add('active');
  const panel = document.getElementById('detail');
  panel.classList.add('open');
  const d = DATA[idx];
  const titleEl = document.getElementById('detail-title');
  const rankEl = document.createElement('strong');
  rankEl.textContent = `#${{d.rank}}`;
  titleEl.replaceChildren(rankEl, document.createTextNode(String(d.filename || '')));
  document.getElementById('detail-oneline').textContent = d.one_line || '';
  const tabs = document.getElementById('version-tabs');
  const img = document.getElementById('version-img');
  const link = document.getElementById('version-link');
  const versions = [
    ['Full', d.output_full],
    ['HD', d.output_hd],
    ['Cropped', d.output_cropped],
  ].filter(v => v[1]);
  tabs.innerHTML = '';
  versions.forEach(([label, file], i) => {{
    const btn = document.createElement('button');
    btn.textContent = label;
    btn.onclick = () => {{
      img.src = file;
      link.href = file;
      tabs.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }};
    if (i === 0) btn.classList.add('active');
    tabs.appendChild(btn);
  }});
  if (versions.length) {{ img.src = versions[0][1]; link.href = versions[0][1]; }}
  // YOLO
  const yoloCol = document.getElementById('yolo-col');
  const yoloImg = document.getElementById('yolo-img');
  const yoloDiv = document.getElementById('detail-yolo');
  if (d.yolo_debug_img || d.yolo) {{
    yoloCol.style.display = 'flex';
    yoloImg.src = d.yolo_debug_img || '';
    yoloImg.style.display = d.yolo_debug_img ? 'block' : 'none';
    if (d.yolo) {{
      const y = d.yolo;
      const yoloList = document.createElement('dl');
      yoloList.className = 'yolo-grid';
      [['Class', y.class_name], ['Conf', y.confidence ? (y.confidence * 100).toFixed(0) + '%' : '\u2014'],
       ['Shot', y.shot_type], ['Facing', y.facing]].forEach(([k,v]) => {{
        const term = document.createElement('dt');
        const value = document.createElement('dd');
        term.textContent = k;
        value.textContent = String(v || '\u2014');
        yoloList.append(term, value);
      }});
      yoloDiv.replaceChildren(yoloList);
    }} else {{ yoloDiv.innerHTML = ''; }}
  }} else {{ yoloCol.style.display = 'none'; }}
  // Burst
  const burstSection = document.getElementById('burst-section');
  const burstDiv = document.getElementById('detail-burst');
  if (d.burst && d.burst.count > 1) {{
    burstSection.style.display = 'block';
    burstDiv.replaceChildren();
    [['Best of ', d.burst.count, ' shots'], ['Via ', d.burst.selected_by, '']].forEach(([prefix, value, suffix]) => {{
      const chip = document.createElement('span');
      const emphasized = document.createElement('span');
      chip.className = 'burst-chip';
      chip.append(document.createTextNode(String(prefix)), emphasized, document.createTextNode(String(suffix)));
      emphasized.textContent = String(value);
      burstDiv.appendChild(chip);
    }});
  }} else {{ burstSection.style.display = 'none'; }}
  // EXIF
  const exifSection = document.getElementById('exif-section');
  const exifDiv = document.getElementById('detail-exif');
  if (d.exif && Object.keys(d.exif).length) {{
    exifSection.style.display = 'block';
    const fmts = {{ camera: v => v, lens: v => v, focal: v => v + 'mm', aperture: v => 'f/' + v,
                    shutter: v => v + 's', iso: v => 'ISO ' + v, date: v => v }};
    exifDiv.replaceChildren();
    Object.entries(fmts).forEach(([k, fmt]) => {{
      if (!d.exif[k]) return;
      const chip = document.createElement('span');
      chip.className = 'exif-chip';
      chip.textContent = String(fmt(d.exif[k]));
      exifDiv.appendChild(chip);
    }});
  }} else {{ exifSection.style.display = 'none'; }}
  // Scores
  const criteria = ['subject_clarity','lighting','color_pop','emotion','scroll_stop','crop_4x5'];
  const scoresDiv = document.getElementById('detail-scores');
  let html = '';
  html += scoreBar('Final', d.final_score, 1, true, true);
  html += scoreBar('Technical', d.technical_composite, 1, true, false);
  html += scoreBar('Vision', d.vision_total, 60, false, false);
  if (d.vision_detail) {{
    html += '<div class="score-divider"></div>';
    criteria.forEach(c => {{
      if (d.vision_detail[c] !== undefined)
        html += scoreBar(c.replace(/_/g,' '), d.vision_detail[c], 10, false, false);
    }});
  }}
  scoresDiv.innerHTML = html;
}}
function scoreBar(label, value, max, isFloat, highlight) {{
  const pct = Math.min(100, (value / max) * 100).toFixed(1);
  const display = isFloat ? value.toFixed(3) : Math.round(value);
  const cls = highlight ? 'score-fill highlight' : 'score-fill';
  return `<div class="score-row">
    <span class="score-label">${{label}}</span>
    <div class="score-track">
      <div class="${{cls}}" style="width:${{pct}}%">
        <div class="score-dot"></div>
      </div>
    </div>
    <span class="score-val">${{display}}</span>
  </div>`;
}}
if (DATA.length) selectImage(0);
</script>
</body>
</html>
"""


INDEX_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f0f0f; --surface: #1a1a1a; --surface2: #242424;
    --border: #333; --text: #e0e0e0; --text-dim: #888;
    --accent: #d32f2f;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
    padding: 2rem;
  }}
  h1 {{
    font-size: 1rem; font-weight: 600; margin-bottom: 1.5rem;
    display: flex; align-items: center; flex-wrap: wrap; gap: .45rem;
  }}
{shared_header_css}
  .folder-list {{
    list-style: none;
  }}
  .folder-list li {{
    margin-bottom: .5rem;
  }}
  .folder-link {{
    display: flex; align-items: center; gap: .75rem;
    padding: .75rem 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; text-decoration: none; color: var(--text);
    transition: border-color .15s, background .15s;
  }}
  .folder-link:hover {{
    border-color: var(--accent); background: var(--surface2);
  }}
  .folder-name {{
    flex: 1; font-weight: 500;
  }}
  .folder-count {{
    font-size: .85rem; color: var(--text-dim);
    white-space: nowrap;
  }}
  .folder-thumb {{
    width: 48px; height: 48px; border-radius: 4px;
    object-fit: cover; background: var(--surface2); flex-shrink: 0;
  }}
</style>
</head>
<body>
<h1>
  <a class="gh-link" href="https://github.com/renatobo/pickinsta" target="_blank" title="pickinsta on GitHub">
    <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
  </a>
  <span class="brand">pickinsta</span> {breadcrumb}<span class="page-title">{title}</span>
</h1>
<ul class="folder-list">
{rows}
</ul>
</body>
</html>
"""
