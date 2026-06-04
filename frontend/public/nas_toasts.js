/* NAS new-trade ALERTS — persistent black panel (reminds until you clear it).
   Polls each system's /api/<key>/state every 12s and adds a row on any NEW
   entry/exit/adjustment, tagged with the system. Rows STAY until dismissed
   (click a row to remove it, or "clear all"). Market-safe runtime overlay. */
(function () {
  if (window.__nasAlerts) return; window.__nasAlerts = true;
  var V = [
    ["nas", "Squeeze·OTM", "#f59e0b"], ["nas-atm", "Squeeze·ATM", "#f59e0b"],
    ["nas-atm2", "Squeeze·ATM2", "#f59e0b"], ["nas-atm4", "Squeeze·ATM4", "#f59e0b"],
    ["nas-916-otm", "9:16·OTM", "#22c55e"], ["nas-916-atm", "9:16·ATM", "#22c55e"],
    ["nas-916-atm2", "9:16·ATM2", "#22c55e"], ["nas-916-atm4", "9:16·ATM4", "#22c55e"]
  ];
  var seen = null, alerts = [];   // alerts: {name,color,text,isExit,time}

  var panel = document.createElement("div");
  panel.style.cssText = "position:fixed;top:16px;right:16px;z-index:2147483647;width:300px;max-height:70vh;display:none;" +
    "flex-direction:column;background:#1b1b1a;border:1px solid #333;border-radius:10px;" +
    "box-shadow:0 8px 30px rgba(0,0,0,.5);font-family:ui-sans-serif,system-ui,sans-serif;overflow:hidden";
  document.body.appendChild(panel);

  function nowHHMM() { var d = new Date(); return ("0"+d.getHours()).slice(-2)+":"+("0"+d.getMinutes()).slice(-2); }

  function render() {
    if (!alerts.length) { panel.style.display = "none"; panel.innerHTML = ""; return; }
    panel.style.display = "flex";
    var head = '<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 12px;border-bottom:1px solid #2a2a2a">' +
      '<span style="font-size:11px;font-weight:700;letter-spacing:.05em;color:#a3a3a3">TRADE ALERTS · ' + alerts.length + '</span>' +
      '<span id="nas_clr" style="font-size:11px;color:#ef4444;cursor:pointer;font-weight:600">clear all</span></div>';
    var body = '<div style="overflow:auto;padding:6px 0">';
    alerts.forEach(function (a, i) {
      body += '<div class="nas_al" data-i="' + i + '" style="padding:7px 12px;border-left:3px solid ' + a.color +
        ';cursor:pointer;display:flex;justify-content:space-between;gap:8px" onmouseover="this.style.background=\'#262626\'" onmouseout="this.style.background=\'\'">' +
        '<div><div style="font-size:10px;font-weight:700;color:' + a.color + '">' + (a.isExit ? "● " : "▲ ") + a.name + '</div>' +
        '<div style="font-size:12.5px;color:#fafaf9;margin-top:2px">' + a.text + '</div></div>' +
        '<div style="font-size:10px;color:#666;white-space:nowrap">' + a.time + '</div></div>';
    });
    body += '</div>';
    panel.innerHTML = head + body;
    document.getElementById("nas_clr").onclick = function () { alerts = []; render(); };
    Array.prototype.forEach.call(panel.querySelectorAll(".nas_al"), function (el) {
      el.onclick = function () { alerts.splice(+el.getAttribute("data-i"), 1); render(); };
    });
  }

  function reasonLabel(r) {
    var m = { SL_HIT: "SL HIT", SL_EXIT_BOTH: "SL-BOTH", MANUAL_ROLL_REBAL: "MANUAL ROLL",
      MANUAL_CLOSE: "MANUAL CLOSE", time_exit: "TIME EXIT", ST_EXIT: "ST EXIT",
      ROLL_OUT: "ROLL OUT", ROLL_IN: "ROLL IN", ASYNC_PARTIAL_FILL_ROLLBACK: "ROLLBACK" };
    return m[r] || (r || "EXIT").toUpperCase();
  }
  function leg(x) { return (x.instrument_type || "") + " " + Math.round(x.strike || 0); }

  function poll() {
    Promise.all(V.map(function (v) {
      return fetch("/api/" + v[0] + "/state").then(function (r) { return r.json(); })
        .then(function (d) { return { v: v, d: d }; }).catch(function () { return null; });
    })).then(function (res) {
      var ns = {}, fresh = [];
      res.forEach(function (r) {
        if (!r || !r.d || !r.d.positions) return;
        var name = r.v[1], color = r.v[2], p = r.d.positions, ent = {}, ex = {};
        ["ce", "pe"].forEach(function (s) {
          (p[s] || []).forEach(function (x) {
            var k = name + "|E|" + x.tradingsymbol + "|" + x.entry_time; ns[k] = 1;
            if (seen && !seen[k]) (ent[x.entry_time] = ent[x.entry_time] || []).push(x);
          });
        });
        (p.closed_today || []).forEach(function (x) {
          var k = name + "|X|" + x.tradingsymbol + "|" + x.exit_time; ns[k] = 1;
          if (seen && !seen[k]) (ex[x.exit_time] = ex[x.exit_time] || []).push(x);
        });
        Object.keys(ent).forEach(function (t) {
          fresh.push({ name: name, color: color, text: "ENTRY · " + ent[t].map(leg).join(" + "), isExit: false, time: nowHHMM() });
        });
        Object.keys(ex).forEach(function (t) {
          var L = ex[t];
          fresh.push({ name: name, color: color, text: reasonLabel(L[0].exit_reason) + " · " + L.map(leg).join(" + "), isExit: true, time: nowHHMM() });
        });
      });
      if (seen && fresh.length) {
        alerts = fresh.concat(alerts).slice(0, 25);   // newest first, cap 25
        render();
      }
      seen = ns;
    }).catch(function () {});
  }

  setInterval(poll, 12000);
  poll();
})();
