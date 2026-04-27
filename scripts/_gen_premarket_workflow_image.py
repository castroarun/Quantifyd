"""Generate quantifyd-premarket-brief.png for the portfolio card.
1492 x 940, matches other project images. Dark theme, cyan + purple
accents, embedded fonts via SVG <text>."""
from pathlib import Path
import cairosvg

OUT = Path(r'C:\Users\Castro\Documents\Projects\castronix-portfolio\assets\projects\quantifyd-premarket-brief.png')
OUT.parent.mkdir(parents=True, exist_ok=True)

# All styling baked in as attributes — no CSS classes (cairosvg-friendly).
SVG = r'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1492 940" width="1492" height="940">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#22D3EE"/></marker>
    <marker id="ahP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#A78BFA"/></marker>
    <marker id="ahW" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#F59E0B"/></marker>
    <linearGradient id="bgGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#0A0A0F"/><stop offset="100%" stop-color="#0F1117"/></linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1492" height="940" fill="url(#bgGrad)"/>

  <!-- Subtle grid -->
  <g stroke="rgba(255,255,255,0.025)" stroke-width="1">
    {grid}
  </g>

  <!-- Title -->
  <text x="60" y="80" font-family="Inter, Helvetica, Arial, sans-serif" font-size="32" font-weight="700" fill="#FAFAFA" letter-spacing="-1">Pre-Market Brief Pipeline</text>
  <text x="60" y="115" font-family="JetBrains Mono, Courier, monospace" font-size="16" fill="#22D3EE" letter-spacing="1.5">QUANTIFYD · 08:00 IST · MON-FRI</text>

  <!-- Stage banners (above the diagram) -->
  <text x="456" y="158" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#22D3EE" letter-spacing="3" font-weight="700">STAGE 1 · BUILD</text>
  <text x="746" y="158" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#22D3EE" letter-spacing="3" font-weight="700">STAGE 2 · SYNTHESIZE</text>
  <text x="1240" y="158" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#22D3EE" letter-spacing="3" font-weight="700">STAGE 3 · DELIVER</text>

  <!-- Cloud routine box (top center) -->
  <rect x="540" y="180" width="412" height="120" rx="14" fill="rgba(167,139,250,0.08)" stroke="rgba(167,139,250,0.5)" stroke-width="2"/>
  <text x="746" y="218" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#A78BFA" letter-spacing="2">CLAUDE CODE · CLOUD ROUTINE</text>
  <text x="746" y="250" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="20" font-weight="600" fill="#FAFAFA">Sandboxed · sentiment + narrative</text>
  <text x="746" y="280" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#A78BFA" font-weight="600">08:02 IST · spawned</text>

  <!-- VPS container -->
  <rect x="60" y="400" width="1372" height="320" rx="16" fill="rgba(22,27,34,0.5)" stroke="rgba(255,255,255,0.07)" stroke-width="1.5"/>
  <text x="92" y="440" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#525252" letter-spacing="2">VPS · CONTABO · PYTHON SERVICE</text>

  <!-- Sources box -->
  <rect x="92" y="470" width="220" height="220" rx="10" fill="transparent" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
  <text x="202" y="502" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#525252" letter-spacing="2">SOURCES</text>
  <text x="202" y="532" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="15" font-weight="600" fill="#FAFAFA">14 yfinance tickers</text>
  <text x="202" y="556" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="14" fill="#A1A1A1">NSE F&amp;O ban CSV</text>
  <text x="202" y="580" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="14" fill="#A1A1A1">holdings_events.db</text>
  <text x="202" y="604" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="14" fill="#A1A1A1">3 RSS feeds</text>
  <text x="202" y="628" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="14" fill="#A1A1A1">BSE corp-actions</text>

  <!-- Builder -->
  <rect x="356" y="510" width="200" height="140" rx="10" fill="#161B22" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
  <text x="456" y="544" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#22D3EE" font-weight="600" letter-spacing="1">08:00 IST</text>
  <text x="456" y="578" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#FAFAFA">Builder</text>
  <text x="456" y="606" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#525252">parallel fetch</text>
  <text x="456" y="626" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#525252">~6s end-to-end</text>

  <!-- JSON cache -->
  <rect x="600" y="510" width="200" height="140" rx="10" fill="#161B22" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
  <text x="700" y="568" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#FAFAFA">JSON cache</text>
  <text x="700" y="596" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#525252">_latest.json</text>
  <text x="700" y="616" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#525252">/api/.../raw</text>

  <!-- Render -->
  <rect x="844" y="510" width="200" height="140" rx="10" fill="#161B22" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
  <text x="944" y="568" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="22" font-weight="700" fill="#FAFAFA">Render</text>
  <text x="944" y="596" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#525252">Mock C v5 HTML</text>
  <text x="944" y="616" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#525252">/synthesized</text>

  <!-- Dispatch -->
  <rect x="1088" y="510" width="180" height="140" rx="10" fill="#161B22" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>
  <text x="1178" y="544" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="15" fill="#22D3EE" font-weight="600" letter-spacing="1">08:03 IST</text>
  <text x="1178" y="582" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#FAFAFA">Dispatch</text>
  <text x="1178" y="612" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="14" fill="#525252">Gmail + WhatsApp</text>

  <!-- Gmail destination -->
  <rect x="1304" y="478" width="148" height="68" rx="10" fill="rgba(34,197,94,0.1)" stroke="rgba(34,197,94,0.6)" stroke-width="2"/>
  <text x="1378" y="510" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#FAFAFA">Gmail</text>
  <text x="1378" y="532" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#22C55E">HTML email</text>

  <!-- WhatsApp destination -->
  <rect x="1304" y="612" width="148" height="68" rx="10" fill="rgba(34,197,94,0.1)" stroke="rgba(34,197,94,0.6)" stroke-width="2"/>
  <text x="1378" y="644" text-anchor="middle" font-family="Inter, Helvetica, Arial, sans-serif" font-size="20" font-weight="700" fill="#FAFAFA">WhatsApp</text>
  <text x="1378" y="666" text-anchor="middle" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#22C55E">Twilio summary</text>

  <!-- Horizontal arrows -->
  <path d="M 312 580 L 350 580" fill="none" stroke="#22D3EE" stroke-width="2" marker-end="url(#ah)"/>
  <path d="M 556 580 L 594 580" fill="none" stroke="#22D3EE" stroke-width="2" marker-end="url(#ah)"/>
  <path d="M 800 580 L 838 580" fill="none" stroke="#22D3EE" stroke-width="2" marker-end="url(#ah)"/>
  <path d="M 1044 580 L 1082 580" fill="none" stroke="#22D3EE" stroke-width="2" marker-end="url(#ah)"/>
  <!-- Dispatch fork: up to Gmail, down to WhatsApp -->
  <path d="M 1268 552 C 1286 552, 1286 512, 1304 512" fill="none" stroke="#22D3EE" stroke-width="2.5" marker-end="url(#ah)"/>
  <path d="M 1268 608 C 1286 608, 1286 646, 1304 646" fill="none" stroke="#22D3EE" stroke-width="2.5" marker-end="url(#ah)"/>

  <!-- Cloud round-trip arrows.  Labels placed cleanly in the gap between cloud bottom (y=300) and VPS top (y=400) -->
  <!-- JSON cache → Cloud -->
  <path d="M 700 510 C 700 420, 700 360, 700 302" fill="none" stroke="#A78BFA" stroke-width="2.5" marker-end="url(#ahP)"/>
  <text x="540" y="350" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#525252">GET /raw</text>
  <text x="540" y="372" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#A78BFA" font-weight="600">08:02 IST</text>

  <!-- Cloud → Render -->
  <path d="M 944 302 C 944 360, 944 420, 944 510" fill="none" stroke="#A78BFA" stroke-width="2.5" marker-end="url(#ahP)"/>
  <text x="966" y="350" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#525252">POST /synthesized</text>
  <text x="966" y="372" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#A78BFA" font-weight="600">08:02 - 08:03</text>

  <!-- Fallback dashed -->
  <path d="M 700 650 C 700 720, 850 750, 944 720" fill="none" stroke="#F59E0B" stroke-width="2" stroke-dasharray="8 6" marker-end="url(#ahW)"/>
  <text x="700" y="775" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#525252">if no synthesis arrived</text>
  <text x="700" y="795" font-family="JetBrains Mono, Courier, monospace" font-size="14" fill="#F59E0B" font-weight="600">08:08 IST · fallback</text>

  <!-- Footer legend -->
  <rect x="60" y="850" width="1372" height="48" rx="8" fill="rgba(22,27,34,0.4)" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
  <line x1="100" y1="874" x2="130" y2="874" stroke="#22D3EE" stroke-width="3"/>
  <text x="140" y="879" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#A1A1A1">happy path · VPS internal</text>
  <line x1="450" y1="874" x2="480" y2="874" stroke="#A78BFA" stroke-width="3"/>
  <text x="490" y="879" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#A1A1A1">cloud round-trip</text>
  <line x1="730" y1="874" x2="760" y2="874" stroke="#F59E0B" stroke-width="3" stroke-dasharray="4 3"/>
  <text x="770" y="879" font-family="JetBrains Mono, Courier, monospace" font-size="13" fill="#A1A1A1">08:08 fallback (if needed)</text>
  <text x="1392" y="879" text-anchor="end" font-family="JetBrains Mono, Courier, monospace" font-size="12" fill="#525252">end-to-end ~3 min · ₹0/mo</text>
</svg>
'''

# Build subtle grid lines
grid_lines = []
for y in range(60, 940, 60):
    grid_lines.append(f'<line x1="0" y1="{y}" x2="1492" y2="{y}"/>')
for x in range(60, 1492, 60):
    grid_lines.append(f'<line x1="{x}" y1="0" x2="{x}" y2="940"/>')
svg = SVG.replace('{grid}', '\n    '.join(grid_lines))

# Write standalone SVG (also useful as an asset)
svg_path = OUT.with_suffix('.svg')
svg_path.write_text(svg, encoding='utf-8')

# Render to PNG
cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=str(OUT), output_width=1492, output_height=940)
print(f'wrote {OUT.relative_to(OUT.parent.parent.parent)} ({OUT.stat().st_size} bytes)')
print(f'wrote {svg_path.name} (also as standalone SVG)')
