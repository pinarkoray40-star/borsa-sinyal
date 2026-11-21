// buyuk2.js - "Büyük Hareket 2" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır. buyuk.js'den SONRA yüklenmeli.
(function () {
  const sec = document.getElementById("p-buyuk2");
  if (!sec || !document.getElementById("t-buyuk2")) return;
  sec.innerHTML = `
    <div class="good"><b>Büyük Hareket 2</b> (günlük veri): <b>Büyük Hareket</b> sekmesiyle aynı sinyal (oynak hisse, 20 günlük zirve kırılımı, hacim kuruması, 60 günlük zirveye yakınlık) <b>artı bir ek şart: dayanıklılık.</b> Hisse, son 120 işlem gününde BIST 100'ün düştüğü günlerde endeksten ortalama daha iyi gitmiş olmalı: bu ölçüde tüm hisseler arasında ilk %30'da (sıra %70 ve üstü). Yani sadece "yükselmeye hazır" değil, "düşüşte de ayakta kalan" hisseler. Aynı hissede 21 gün yeni sinyal yok.</div>
    <div class="good"><b>Takip kuralı</b> Büyük Hareket ile aynı: giriş = sinyal günü kapanışı. Kapanış, girişten sonraki en yüksek kapanıştan <b>%25 aşağı</b> inerse <b>İZLEYEN STOP</b> ile çıkılır; olmazsa 84. işlem gününde SÜRE DOLDU. Sabit hedef yok. 5 yıl ve 526 hissede yapılan geçmiş testte bu ek şart, Büyük Hareket'in aynı günlerde rastgele likit hisselere göre farkını kabaca <b>+14 puandan +20 puana</b> çıkardı: Büyük Hareket sinyallerinin yaklaşık üçte biri (dayanıklı olmayanlar) hiçbir şey katmıyordu.</div>
    <div class="warn"><b>Önce bunu bil.</b> Bu şartı geçmiş sonuçlara bakarak seçtik; böyle seçilen filtrenin gerçek faydası geçmişte görülenden genelde <b>daha küçük</b> çıkar. Test küçük gruplarla (200 ile 330 işlem) ve güncel hisse listesiyle yapıldı (batanlar yok, getiriyi iyimser gösterir), sadece kapanış fiyatı kullandı. Büyük Hareket'in tüm uyarıları burada da geçerli: kazanç az sayıda büyük işlemden gelir, uzun kötü seriler olur, bazı işlemler stop'tan aşağı kayabilir. Bu sekme Büyük Hareket'e dokunmadan yanında <b>canlı olarak izlemek</b> için eklendi. Asıl kanıt <b>"canlı"</b> işaretli sinyallerdir (20.11.2026'dan sonra çıkanlar); ikisini birkaç ay yan yana izle. Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="b2stats"></div>
    <div class="chips" id="b2chips"></div>
    <div id="b2list"></div>
    <button class="more" id="b2more" hidden>Daha fazla göster</button>
    <div class="msg" id="b2msg">Yükleniyor…</div>`;

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
        <div class="meta">${dmy(e.date)} · giriş ${nf(e.entry)} · zirve ${nf(e.peak_px)} · stop ${nf(e.stop_px)} · dayanıklılık sırası %${e.dnr}${kv}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: sinyal günü kapanışı).</li>${e.bench == null ? "" : `<li>Kıyas: aynı gün giriş yapılsa tüm likit hisselerin ortalaması aynı sürede ${pct(e.bench, 1)} olurdu; bu işlem ondan ${pct(e.diff, 1)} puan ${e.diff >= 0 ? "iyi" : "kötü"}.</li>`}<li>Dayanıklılık sırası %${e.dnr}: son 120 günde endeks düşerken, hisselerin %${e.dnr}'inden daha iyi gitmiş.</li></ul></div></details>`;
  }

  function render() {
    const D = data && data.buyuk2;
    if (!data) return;
    if (!D) { el("b2msg").style.display = "block"; el("b2msg").textContent = "Büyük Hareket 2 verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["İzleyen stop", cnt(e => e.st === "stop"), e => e.st === "stop"],
      ["Süre doldu", cnt(e => e.st === "süre"), e => e.st === "süre"], ["Canlı", cnt(e => e.live), e => e.live]];
    el("b2chips").innerHTML = chips.map(([t, n]) => `<button aria-pressed="${t === B.chip}" data-t="${t}">${t} (${n})</button>`).join("");
    const f = (chips.find(c => c[0] === B.chip) || chips[0])[2];
    const list = D.entries.filter(f);
    el("b2list").innerHTML = list.slice(0, B.shown).map(e => row(e, P)).join("");
    el("b2more").hidden = list.length <= B.shown;
    el("b2msg").style.display = list.length ? "none" : "block";
    el("b2msg").textContent = D.n ? "Bu filtrede sinyal yok." : "Henüz sinyal yok. Beş şart birlikte nadiren sağlanır, çıktıkça burada birikir.";
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
    el("b2stats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda sinyal<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" + (S.open_bench == null ? "" : `<br>evren aynı sürede ${pct(S.open_bench, 1)}, fark <span class="${sgn(S.open_diff)}">${pct(S.open_diff, 1)}</span>`) : "açık işlem yok"}</div></div>
      ${done}
      <div class="edge">Kapanan işlemler izleyen stop ya da 84. günle bitmiş olanlardır. Evren = işlem hacmi en düşük %30 hariç tüm hisselerin, aynı gün girişle aynı sürede eşit ağırlıklı ortalaması. Az işlemde (30'dan az) sonuçlara güvenme.${part}</div></div>`;
  }

  el("b2chips").addEventListener("click", e => { const t = e.target.dataset.t; if (t) { B.chip = t; B.shown = 30; render(); } });
  el("b2more").onclick = () => { B.shown += 30; render(); };

  const _tab = window.tab;
  window.tab = function (which) {
    _tab(which);
    el("p-buyuk2").hidden = which !== "buyuk2";
    el("t-buyuk2").setAttribute("aria-selected", which === "buyuk2");
  };
  el("t-buyuk2").onclick = () => window.tab("buyuk2");

  // data her yenilemede yeni nesne olur; değişince yeniden çiz
  setInterval(() => { if (data && data !== B.last) { B.last = data; render(); } }, 500);
})();
