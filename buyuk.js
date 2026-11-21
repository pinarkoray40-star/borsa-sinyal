// buyuk.js - "Büyük Hareket" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır.
(function () {
  const sec = document.getElementById("p-buyuk");
  if (!sec) return;
  sec.innerHTML = `
    <div class="good"><b>Büyük Hareket</b> (günlük veri): son 60 günde <b>oynak</b> hisse, kapanışı önceki 20 günün zirvesine eşit/üstünde (<b>kırılım</b>), son 5 günün hacmi 60 günlük ortalamanın %70'inin altında (<b>hacim kuruması</b>) ve fiyat 60 günlük zirvenin en az %90'ında. Dördü birlikte sağlanınca sinyal. İşlem hacmi en düşük %30 hisse hariç, aynı hissede 21 gün yeni sinyal yok.</div>
    <div class="good"><b>Takip kuralı:</b> giriş = sinyal günü kapanışı. Kapanış, girişten sonraki en yüksek kapanıştan (zirve) <b>%25 aşağı</b> inerse <b>İZLEYEN STOP</b> ile çıkılır; olmazsa 84. işlem gününde SÜRE DOLDU. <b>Sabit hedef yok</b>: yükselen hisse koşturulur, stop zirveyle birlikte yukarı çıkar. 5 yıl ve 526 hissede yapılan testte (%0,4 komisyon dahil, 779 işlem) işlem başına ortalama <b>+%31</b>; aynı günlerde rastgele seçilen likit hisselere göre fark <b>+%13</b> (iki yarı dönemde de pozitif, en iyi 3 işlem çıkınca da +%29 ortalama). Eski kural (+%30 hedef / −%15 stop, 42 gün) rastgeleye göre sadece +%3 fark veriyordu.</div>
    <div class="warn"><b>Önce bunu bil.</b> Kazanç az sayıda büyük işlemden gelir: medyan işlem sadece <b>+%8</b>, kazanma oranı <b>%55</b>. Testte üst üste 37 işlem kazançsız geçti, en kötü işlem −%40 oldu (stop gap yüzünden −%25'ten aşağı kayabilir), kötü %5'lik dilim −%27 ve altı. Para ortalama 54 gün bağlı kalır. Tek hisseye değil birkaç sinyale bölerek gir ve uzun kötü seriye dayanabilecek kadar küçük pozisyon al. Test güncel hisse listesiyle yapıldı (batanlar yok, getiriyi iyimser gösterir) ve sadece kapanış fiyatı kullandı: gerçekte gap ve alım-satım farkı sonucu düşürür. Son 5 yılın çoğu güçlü boğa dönemiydi; son 6 ayda piyasa zayıf ve sonuçlar eksi. Buradaki <b>"geçmiş"</b> sinyaller kuralı yazarken kullandığımız dönemin içindedir, asıl kanıt <b>"canlı"</b> olanlardır (18.11.2026'dan sonra çıkanlar). Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="bstats"></div>
    <div class="chips" id="bchips"></div>
    <div id="blist"></div>
    <button class="more" id="bmore" hidden>Daha fazla göster</button>
    <div class="msg" id="bmsg">Yükleniyor…</div>`;

  const B = { chip: "Hepsi", shown: 30, last: null };
  const ST = { "açık": ["Açık", "BEKLE"], "stop": ["İzleyen stop", "SAT"], "süre": ["Süre doldu", "BEKLE"] };
  const dmy = s => s ? s.split("-").reverse().join(".") : "";
  const el = id => document.getElementById(id);

  function row(e, P) {
    const [lbl, c] = ST[e.st];
    const kv = e.bench == null ? "" : ` · evren ${pct(e.bench, 1)} · fark <span class="${sgn(e.diff)}">${pct(e.diff, 1)}</span>`;
    const why = e.st === "açık"
      ? `Takip sürüyor: gün ${e.days}/${P.hold}. Zirve kapanış ${nf(e.peak_px)}, stop seviyesi ${nf(e.stop_px)} (zirve çıkarsa stop da yukarı çıkar, kapanışla kontrol edilir).`
      : `Çıkış: ${dmy(e.xdate)} (${e.days}. gün), ${lbl.toLowerCase()}, kapanışla ${pct(e.pnl, 1)}.`;
    return `<details class="row"><summary>
      <div><span class="badge ${c}">${lbl}</span><span class="sym">${e.symbol}</span>${e.live ? '<span class="tag">canlı</span>' : ""}
        <div class="meta">${dmy(e.date)} · giriş ${nf(e.entry)} · zirve ${nf(e.peak_px)} · stop ${nf(e.stop_px)}${kv}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: sinyal günü kapanışı).</li>${e.bench == null ? "" : `<li>Kıyas: aynı gün giriş yapılsa tüm likit hisselerin ortalaması aynı sürede ${pct(e.bench, 1)} olurdu; bu işlem ondan ${pct(e.diff, 1)} puan ${e.diff >= 0 ? "iyi" : "kötü"}.</li>`}</ul></div></details>`;
  }

  function render() {
    const D = data && data.buyuk;
    if (!data) return;
    if (!D) { el("bmsg").style.display = "block"; el("bmsg").textContent = "Büyük Hareket verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["İzleyen stop", cnt(e => e.st === "stop"), e => e.st === "stop"],
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
        <tr><td>İzleyen stopla çıkan</td><td>%${nf(S.stop, 0)}</td></tr>
        <tr><td>Süresi dolan</td><td>%${nf(S.sure, 0)}</td></tr>
        <tr><td>Ortalama getiri</td><td><b class="${sgn(S.avg)}">${pct(S.avg, 1)}</b></td></tr>
        <tr><td>Medyan getiri</td><td>${pct(S.median, 1)}</td></tr>
        <tr><td>Kazanan işlem</td><td>%${nf(S.win, 0)}</td></tr>
        <tr><td>En iyi / en kötü</td><td>${pct(S.best, 1)} / ${pct(S.worst, 1)}</td></tr>
        ${S.bench_avg == null ? "" : `<tr><td>Aynı günlerde evren ortalaması</td><td>${pct(S.bench_avg, 1)}</td></tr>
        <tr><td>Fark (puan)</td><td><b class="${sgn(S.diff_avg)}">${pct(S.diff_avg, 1)}</b></td></tr>
        <tr><td>Evreni geçen işlem</td><td>%${nf(S.beat, 0)}</td></tr>`}</table>` : "";
    el("bstats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda sinyal<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" + (S.open_bench == null ? "" : `<br>evren aynı sürede ${pct(S.open_bench, 1)}, fark <span class="${sgn(S.open_diff)}">${pct(S.open_diff, 1)}</span>`) : "açık işlem yok"}</div></div>
      ${done}
      <div class="edge">Kapanan işlemler izleyen stop ya da 84. günle bitmiş olanlardır. Evren = işlem hacmi en düşük %30 hariç tüm hisselerin, aynı gün girişle aynı sürede eşit ağırlıklı ortalaması. Az işlemde (30'dan az) sonuçlara güvenme.${part}</div></div>`;
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
