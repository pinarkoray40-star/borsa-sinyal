// buyuk.js - "Büyük Hareket" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır.
(function () {
  const sec = document.getElementById("p-buyuk");
  if (!sec) return;
  sec.innerHTML = `
    <div class="good"><b>Büyük Hareket</b> (günlük veri): son 60 günde <b>oynak</b> hisse, kapanışı önceki 20 günün zirvesine eşit/üstünde (<b>kırılım</b>), son 5 günün hacmi 60 günlük ortalamanın %70'inin altında (<b>hacim kuruması</b>) ve fiyat 60 günlük zirvenin en az %90'ında. Dördü birlikte sağlanınca sinyal. İşlem hacmi en düşük %30 hisse hariç, aynı hissede 21 gün yeni sinyal yok.</div>
    <div class="good"><b>Takip kuralı:</b> giriş = sinyal günü kapanışı. 42 işlem günü içinde kapanış önce <b>+%30</b> olursa HEDEF, önce <b>−%15</b> olursa STOP, ikisi de olmazsa 42. günde SÜRE DOLDU. 5 yıl ve 526 hissede yapılan testte (%0,4 komisyon dahil) 807 işlemde işlem başına ortalama <b>+%8,9</b>, medyan <b>+%11,1</b>, işlemlerin %40'ı hedefe, %35'i stopa gitti; hem ilk hem ikinci yarı dönemde rastgele hisseden iyiydi.</div>
    <div class="warn"><b>Önce bunu bil.</b> Üç işlemden biri stop olur (−%15), bu normaldir: tek hisseye değil birkaç sinyale bölerek gir. Test güncel hisse listesiyle yapıldı (batanlar yok) ve sadece kapanış fiyatı kullandı: gerçekte stop, gap yüzünden daha kötü dolabilir; oynak/küçük hisselerde alım-satım farkı da komisyondan büyük olabilir. Test son 5 yılda, çoğu güçlü boğa döneminde yapıldı. Buradaki <b>"geçmiş"</b> sinyaller kuralı yazarken kullandığımız dönemin içindedir, asıl kanıt <b>"canlı"</b> olanlardır (18.11.2026'dan sonra çıkanlar). Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="bstats"></div>
    <div class="chips" id="bchips"></div>
    <div id="blist"></div>
    <button class="more" id="bmore" hidden>Daha fazla göster</button>
    <div class="msg" id="bmsg">Yükleniyor…</div>`;

  const B = { chip: "Hepsi", shown: 30, last: null };
  const ST = { "açık": ["Açık", "BEKLE"], "hedef": ["Hedef", "AL"], "stop": ["Stop", "SAT"], "süre": ["Süre doldu", "BEKLE"] };
  const dmy = s => s ? s.split("-").reverse().join(".") : "";
  const el = id => document.getElementById(id);

  function row(e, P) {
    const [lbl, c] = ST[e.st];
    const why = e.st === "açık"
      ? `Takip sürüyor: gün ${e.days}/${P.hold}. Hedef ${nf(e.tp_px)}, stop ${nf(e.sl_px)} (kapanışla kontrol edilir).`
      : `Çıkış: ${dmy(e.xdate)} (${e.days}. gün), ${lbl.toLowerCase()}, kapanışla ${pct(e.pnl, 1)}.`;
    return `<details class="row"><summary>
      <div><span class="badge ${c}">${lbl}</span><span class="sym">${e.symbol}</span>${e.live ? '<span class="tag">canlı</span>' : ""}
        <div class="meta">${dmy(e.date)} · giriş ${nf(e.entry)} · hedef ${nf(e.tp_px)} · stop ${nf(e.sl_px)}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: sinyal günü kapanışı).</li></ul></div></details>`;
  }

  function render() {
    const D = data && data.buyuk;
    if (!data) return;
    if (!D) { el("bmsg").style.display = "block"; el("bmsg").textContent = "Büyük Hareket verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["Hedef", cnt(e => e.st === "hedef"), e => e.st === "hedef"], ["Stop", cnt(e => e.st === "stop"), e => e.st === "stop"],
      ["Süre doldu", cnt(e => e.st === "süre"), e => e.st === "süre"], ["Canlı", cnt(e => e.live), e => e.live]];
    el("bchips").innerHTML = chips.map(([t, n]) => `<button aria-pressed="${t === B.chip}" data-t="${t}">${t} (${n})</button>`).join("");
    const f = (chips.find(c => c[0] === B.chip) || chips[0])[2];
    const list = D.entries.filter(f);
    el("blist").innerHTML = list.slice(0, B.shown).map(e => row(e, P)).join("");
    el("bmore").hidden = list.length <= B.shown;
    el("bmsg").style.display = list.length ? "none" : "block";
    el("bmsg").textContent = D.n ? "Bu filtrede sinyal yok." : "Henüz sinyal yok. Dört şart birlikte nadiren sağlanır, çıktıkça burada birikir.";
    const part = D.dropped_partial ? " Bugünün mumu henüz kapanmadığı için son kapanmış güne (" + dmy(D.last_bar) + ") bakıldı." : "";
    const done = S.n_done ? `<table><tr><th></th><th>Kapanan işlemler</th></tr>
        <tr><td>Hedefe giden</td><td>%${nf(S.hedef, 0)}</td></tr>
        <tr><td>Stop olan</td><td>%${nf(S.stop, 0)}</td></tr>
        <tr><td>Süresi dolan</td><td>%${nf(S.sure, 0)}</td></tr>
        <tr><td>Ortalama getiri</td><td><b class="${sgn(S.avg)}">${pct(S.avg, 1)}</b></td></tr>
        <tr><td>Medyan getiri</td><td>${pct(S.median, 1)}</td></tr>
        <tr><td>Kazanan işlem</td><td>%${nf(S.win, 0)}</td></tr></table>` : "";
    el("bstats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda sinyal<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" : "açık işlem yok"}</div></div>
      ${done}
      <div class="edge">Kapanan işlemler hedef, stop ya da 42. günle bitmiş olanlardır. Az işlemde (30'dan az) sonuçlara güvenme.${part}</div></div>`;
  }

  el("bchips").addEventListener("click", e => { const t = e.target.dataset.t; if (t) { B.chip = t; B.shown = 30; render(); } });
  el("bmore").onclick = () => { B.shown += 30; render(); };

  const _tab = window.tab;
  window.tab = function (which) {
    _tab(which);
    el("p-buyuk").hidden = which !== "buyuk";
    el("t-buyuk").setAttribute("aria-selected", which === "buyuk");
  };
  el("t-buyuk").onclick = () => window.tab("buyuk");

  // data her yenilemede yeni nesne olur; değişince yeniden çiz
  setInterval(() => { if (data && data !== B.last) { B.last = data; render(); } }, 500);
})();
