// ayakta.js - "Ayakta" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır. buyuk2.js'den SONRA yüklenmeli.
(function () {
  const sec = document.getElementById("p-ayakta");
  if (!sec || !document.getElementById("t-ayakta")) return;
  sec.innerHTML = `
    <div class="good"><b>Ayakta</b> (günlük veri): <b>endeks zayıfken kendi başına yükselen hisse.</b> İki şart birlikte aranır: BIST 100 son 60 işlem gününde en az <b>%3 düşmüş</b> olmalı (endeks zayıf değilse bu sekme sinyal vermez) ve hisse aynı 60 günde <b>en az %10 yükselmiş</b> olmalı (mutlak getiri, işlem hacmi en düşük %30 hisse hariç). Giriş = sinyal günü kapanışı, <b>84 işlem günü tutulur, stop yok</b>. Aynı hissede 84 gün yeni sinyal yok.</div>
    <div class="good"><b>Geçmiş test</b> 5 yıl, 525 hisse: endeks 60 günde düşmüşken böyle seçilen hisseler, 84 gün sonra aynı günün tüm likit hisse ortalamasından ortalama <b>+5,5 puan</b> önde çıktı (t=3,4); iki yarı dönemde de pozitifti (+5,1 ve +6,0).</div>
    <div class="warn"><b>Önce bunu bil.</b> (1) Bu sekme "kötü piyasada da kazandırır" demek değil: testte endeks düşmeye devam ettiğinde bu hisseler de düştü (21 günde yaklaşık %6,5, 42 günde yaklaşık %11), sadece evrenden az düştüler. Fark asıl <b>endeks toparlanınca</b> geldi. (2) Ortalamayı az sayıda büyük kazanan çekiyor: medyan fark yaklaşık -%5, hisselerin sadece yaklaşık %46'sı aynı günün ortalamasını geçti. (3) Zayıf piyasa dönemi az sayıda ve birbirine yapışık; bu yüzden t değeri olduğundan sağlam görünür. (4) Şartı geçmişe bakarak seçtik, gerçek fayda genelde daha küçük çıkar; test güncel hisse listesiyle yapıldı (batanlar yok, iyimser) ve sadece kapanış fiyatı kullandı. Asıl kanıt <b>"canlı"</b> işaretli sinyallerdir (20.11.2026 ve sonrası), birkaç ay izlenecek. Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="aystate"></div>
    <div id="aystats"></div>
    <div class="chips" id="aychips"></div>
    <div id="aylist"></div>
    <button class="more" id="aymore" hidden>Daha fazla göster</button>
    <div class="msg" id="aymsg">Yükleniyor…</div>`;

  const B = { chip: "Hepsi", shown: 30, last: null };
  const ST = { "açık": ["Açık", "BEKLE"], "süre": ["Süre doldu", "BEKLE"] };
  const dmy = s => s ? s.split("-").reverse().join(".") : "";
  const el = id => document.getElementById(id);

  function row(e, P) {
    const [lbl, c] = ST[e.st] || ST["açık"];
    const why = e.st === "açık"
      ? `Takip sürüyor: gün ${e.days}/${P.hold}. Stop yok, ${P.hold}. işlem gününün kapanışında çıkılır.`
      : `Çıkış: ${dmy(e.xdate)} (${e.days}. gün), ${lbl.toLowerCase()}, kapanışla ${pct(e.pnl, 1)}.`;
    const ir = e.ir == null ? "?" : pct(e.ir, 1);
    return `<details class="row"><summary>
      <div><span class="badge ${c}">${lbl}</span><span class="sym">${e.symbol}</span>${e.live ? '<span class="tag">canlı</span>' : ""}
        <div class="meta">${dmy(e.date)} · giriş ${nf(e.entry)} · hisse 60 günde ${pct(e.sr, 1)} · endeks ${ir}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: sinyal günü kapanışı).</li><li>Sinyal gününde hisse son 60 günde ${pct(e.sr, 1)}, BIST 100 ise ${ir}.</li></ul></div></details>`;
  }

  function render() {
    const D = data && data.ayakta;
    if (!data) return;
    if (!D) { el("aymsg").style.display = "block"; el("aymsg").textContent = "Ayakta verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["Süre doldu", cnt(e => e.st === "süre"), e => e.st === "süre"], ["Canlı", cnt(e => e.live), e => e.live]];
    el("aychips").innerHTML = chips.map(([t, n]) => `<button aria-pressed="${t === B.chip}" data-t="${t}">${t} (${n})</button>`).join("");
    const f = (chips.find(c => c[0] === B.chip) || chips[0])[2];
    const list = D.entries.filter(f);
    el("aylist").innerHTML = list.slice(0, B.shown).map(e => row(e, P)).join("");
    el("aymore").hidden = list.length <= B.shown;
    el("aymsg").style.display = list.length ? "none" : "block";
    el("aymsg").textContent = D.n ? "Bu filtrede sinyal yok." : "Henüz sinyal yok. Endeks zayıf olmadıkça bu sekme sinyal üretmez; zayıflayınca burada birikir.";
    const I = D.index || {};
    el("aystate").innerHTML = I.ret == null ? "" :
      `<div class="card"><b>Endeks durumu:</b> BIST 100 son ${I.w} işlem gününde <b class="${sgn(I.ret)}">${pct(I.ret, 1)}</b>. ` +
      (I.weak ? "Şart sağlanıyor (en az %3 düşüş): yeni sinyal üretilebilir." : "Şart sağlanmıyor (düşüş %3'ten az): yeni sinyal çıkmaz.") + "</div>";
    const part = D.dropped_partial ? " Bugünün mumu henüz kapanmadığı için son kapanmış güne (" + dmy(D.last_bar) + ") bakıldı." : "";
    const done = S.n_done ? `<table><tr><th></th><th>Kapanan işlemler (84. gün)</th></tr>
        <tr><td>Ortalama getiri</td><td><b class="${sgn(S.avg)}">${pct(S.avg, 1)}</b></td></tr>
        <tr><td>Medyan getiri</td><td>${pct(S.median, 1)}</td></tr>
        <tr><td>Kazanan işlem</td><td>%${nf(S.win, 0)}</td></tr>
        <tr><td>En iyi / en kötü</td><td>${pct(S.best, 1)} / ${pct(S.worst, 1)}</td></tr></table>` : "";
    el("aystats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda sinyal<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" : "açık işlem yok"}</div></div>
      ${done}
      <div class="edge">Kapanan işlemler 84. günle bitmiş olanlardır. Az işlemde (30'dan az) sonuçlara güvenme.${part}</div></div>`;
  }

  el("aychips").addEventListener("click", e => { const t = e.target.dataset.t; if (t) { B.chip = t; B.shown = 30; render(); } });
  el("aymore").onclick = () => { B.shown += 30; render(); };

  const _tab = window.tab;
  window.tab = function (which) {
    _tab(which);
    el("p-ayakta").hidden = which !== "ayakta";
    el("t-ayakta").setAttribute("aria-selected", which === "ayakta");
  };
  el("t-ayakta").onclick = () => window.tab("ayakta");

  // data her yenilemede yeni nesne olur; değişince yeniden çiz
  setInterval(() => { if (data && data !== B.last) { B.last = data; render(); } }, 500);
})();
