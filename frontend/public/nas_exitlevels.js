/* NAS live EXIT-LEVELS panel — self-contained overlay.
   For every open leg, shows its current exit: ST <trail> for SuperTrend-managed
   naked legs (live st_value), or SL <price> for fixed-stop legs. Market-safe
   runtime overlay (interim until the in-table column ships in the rebuild). */
(function () {
  if (window.__nasExitLevels) return; window.__nasExitLevels = true;
  var V = [
    ["nas", "Squeeze·OTM"], ["nas-atm", "Squeeze·ATM"], ["nas-atm2", "Squeeze·ATM2"], ["nas-atm4", "Squeeze·ATM4"],
    ["nas-916-otm", "9:16·OTM"], ["nas-916-atm", "9:16·ATM"], ["nas-916-atm2", "9:16·ATM2"], ["nas-916-atm4", "9:16·ATM4"]
  ];

  var panel = document.createElement("div");
  panel.style.cssText = "position:fixed;bottom:16px;right:16px;z-index:2147483646;width:250px;max-height:60vh;overflow:auto;" +
    "background:#1b1b1a;border:1px solid #333;border-radius:10px;padding:10px 12px;color:#fafaf9;" +
    "box-shadow:0 8px 28px rgba(0,0,0,.5);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5";
  document.body.appendChild(panel);
  var collapsed = false;

  function row(label, exit, isST) {
    return '<div style="display:flex;justify-content:space-between;gap:8px">' +
      '<span style="color:#d4d4d4">' + label + '</span>' +
      '<span style="color:' + (isST ? "#22c55e" : "#f59e0b") + ';font-weight:600">' + exit + '</span></div>';
  }

  function render(groups) {
    var hdr = '<div style="display:flex;justify-content:space-between;align-items:center;cursor:pointer;margin-bottom:6px" id="nas_el_hdr">' +
      '<span style="font-weight:700;letter-spacing:.04em;color:#a3a3a3;font-size:11px">EXIT LEVELS</span>' +
      '<span style="color:#666;font-size:11px">' + (collapsed ? "+" : "−") + '</span></div>';
    var body = "";
    if (!collapsed) {
      var keys = Object.keys(groups);
      if (!keys.length) body = '<div style="color:#666">no open legs</div>';
      keys.forEach(function (g) {
        body += '<div style="color:#737373;font-size:10px;margin-top:6px;font-weight:700">' + g + '</div>';
        groups[g].forEach(function (l) { body += row(l.label, l.exit, l.isST); });
      });
      body += '<div style="color:#525252;font-size:10px;margin-top:8px;border-top:1px solid #2a2a2a;padding-top:6px">' +
        '<span style="color:#22c55e">ST</span> = SuperTrend trail (5m) &nbsp; <span style="color:#f59e0b">SL</span> = fixed stop</div>';
    }
    panel.innerHTML = hdr + body;
    document.getElementById("nas_el_hdr").onclick = function () { collapsed = !collapsed; render(groups); };
  }

  function poll() {
    var stmap = {};
    fetch("/api/nas/ticker/status").then(function (r) { return r.json(); }).then(function (t) {
      ["atm_naked_st", "atm4_naked_st", "atm2_naked_st"].forEach(function (f) {
        var d = t[f]; if (d && d.active && d.tradingsymbol && d.st_value != null) stmap[d.tradingsymbol] = d.st_value;
      });
    }).catch(function () {}).then(function () {
      return Promise.all(V.map(function (v) {
        return fetch("/api/" + v[0] + "/state").then(function (r) { return r.json(); })
          .then(function (d) { return { v: v, d: d }; }).catch(function () { return null; });
      }));
    }).then(function (res) {
      var groups = {};
      res.forEach(function (r) {
        if (!r || !r.d || !r.d.positions) return;
        var name = r.v[1], p = r.d.positions, legs = [];
        ["ce", "pe"].forEach(function (s) {
          (p[s] || []).forEach(function (x) {
            var ts = x.tradingsymbol, label = (x.instrument_type || "") + " " + Math.round(x.strike || 0);
            if (stmap[ts] != null) legs.push({ label: label, exit: "ST " + Number(stmap[ts]).toFixed(1), isST: true });
            else {
              var sl = x.sl_price;
              var ex = (sl == null) ? "—" : (sl >= 999999 ? "ST(warming)" : "SL " + Number(sl).toFixed(1));
              legs.push({ label: label, exit: ex, isST: false });
            }
          });
        });
        if (legs.length) groups[name] = legs;
      });
      render(groups);
    }).catch(function () {});
  }

  setInterval(poll, 8000);
  poll();
})();
