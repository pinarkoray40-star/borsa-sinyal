// buyuk3.js - "Büyük Hareket 3" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır. buyuk2.js'den SONRA yüklenmeli.
(function () {
  const sec = document.getElementById("p-buyuk3");
  if (!sec || !document.getElementById("t-buyuk3")) return;
  sec.innerHTML = `
    <div class="good"><b>Büyük Hareket 3</b> (günlük veri): <b>Büyük Hareket 2</b> sekmesiyle aynı sinyal (oynak hisse, 20 günlük zirve kırılımı, hacim kuruması, 60 günlük zirveye yakınlık, endeks düşerken dayanıklılık) <b>artı bir ek şart: alt sektör zayıf olmamalı.</b> Hissenin alt sektöründeki <b>diğer</b> hisselerin son 20 günlük ortalama getirisi, tüm hisselerin ortalamasından yüksek olmalı. Alt sektörü piyasadan zayıf olan hissenin sinyali elenir. Alt sektörü bilinmeyen ya da çok küçük olan (6 hisseden az) hisselerin sinyali elenmez. Aynı hissede 21 gün yeni sinyal yok.</div>
    <div class="good"><b>Takip kuralı</b> diğer Büyük Hareket sekmeleriyle aynı: giriş = sinyal günü kapanışı. Kapanış, girişten sonraki en yüksek kapanıştan <b>%25 aşağı</b> inerse <b>İZLEYEN STOP</b> ile çıkılır; olmazsa 84. işlem gününde SÜRE DOLDU. Sabit hedef yok. 5 yıllık geçmiş testte bu şart, Büyük Hareket 2'yi 10 ve 20 pozisyonlu simülasyonda hem %0,8 hem %1,5 maliyetle iyileştirdi (örnek: 20 pozisyon, yıllık getiri 117'den 133'e, en büyük düşüş -25'ten -20'ye). 30 pozisyonda fark yoktu.</div>
    <div class="warn"><b>Önce bunu bil.</b> Bu şartı geçmiş sonuçlara bakarak seçtik, dört çeşidin en iyisini aldık; böyle seçilen filtrenin gerçek faydası geçmişte görülenden genelde <b>daha küçük</b> çıkar. Alt sektör sınıflaması bugünün Yahoo verisidir ve test sadece bugün listede olan hisselerle yapıldı (batanlar yok, getiriyi iyimser gösterir). Büyük Hareket'in tüm uyarıları burada da geçerli: kazanç az sayıda büyük işlemden gelir, uzun kötü seriler olur, bazı işlemler stop'tan aşağı kayabilir. Bu sekme diğerlerine dokunmadan yanlarında <b>canlı olarak izlemek</b> için eklendi. Asıl kanıt <b>"canlı"</b> işaretli sinyallerdir (27.11.2026'dan sonra çıkanlar); Büyük Hareket 2 ile birkaç ay yan yana izle. Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="b3stats"></div>
    <div class="chips" id="b3chips"></div>
    <div id="b3list"></div>
    <button class="more" id="b3more" hidden>Daha fazla göster</button>
    <div class="msg" id="b3msg">Yükleniyor…</div>`;

  const B = { chip: "Hepsi", shown: 30, last: null };
  const ST = { "açık": ["Açık", "BEKLE"], "stop": ["İzleyen stop", "SAT"], "süre": ["Süre doldu", "BEKLE"] };
  const dmy = s => s ? s.split("-").reverse().join(".") : "";
  const el = id => document.getElementById(id);
  const puan = v => (v > 0 ? "+" : "") + nf(v, 1) + " puan";

  function row(e, P) {
    const [lbl, c] = ST[e.st];
    const kv = e.bench == null ? "" : ` · evren ${pct(e.bench, 1)} · fark <span class="${sgn(e.diff)}">${pct(e.diff, 1)}</span>`;
    const sk = e.ind ? ` · ${e.ind}${e.sg == null ? "" : " " + puan(e.sg)}` : " · alt sektör bilinmiyor";
    const why = e.st === "açık"
      ? `Takip sürüyor: gün ${e.days}/${P.hold}. Zirve kapanış ${nf(e.peak_px)}, stop seviyesi ${nf(e.stop_px)} (zirve çıkarsa stop da yukarı çıkar, kapanışla kontrol edilir).`
      : `Çıkış: ${dmy(e.xdate)} (${e.days}. gün), ${lbl.toLowerCase()}, kapanışla ${pct(e.pnl, 1)}.`;
    const sektor = e.sg != null
      ? `Alt sektör (${e.ind}): son 20 günde diğer hisselerin ortalaması piyasadan ${puan(e.sg)} fark ediyordu (sinyal günü).`
      : (e.ind ? `Alt sektör (${e.ind}) çok küçük olduğu için gücü ölçülemedi; sinyal elenmedi.` : `Alt sektör bilgisi yok; sinyal elenmedi.`);
    return `<details class="row"><summary>
      <div><span class="badge ${c}">${lbl}</span><span class="sym">${e.symbol}</span>${e.live ? '<span class="tag">canlı</span>' : ""}
        <div class="meta">${dmy(e.date)} · giriş ${nf(e.entry)} · zirve ${nf(e.peak_px)} · stop ${nf(e.stop_px)} · dayanıklılık sırası %${e.dnr}${sk}${kv}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: sinyal günü kapanışı).</li>${e.bench == null ? "" : `<li>Kıyas: aynı gün giriş yapılsa tüm likit hisselerin ortalaması aynı sürede ${pct(e.bench, 1)} olurdu; bu işlem ondan ${pct(e.diff, 1)} puan ${e.diff >= 0 ? "iyi" : "kötü"}.</li>`}<li>Dayanıklılık sırası %${e.dnr}: son 120 günde endeks düşerken, hisselerin %${e.dnr}'inden daha iyi gitmiş.</li><li>${sektor}</li></ul></div></details>`;
  }

  function render() {
    const D = data && data.buyuk3;
    if (!data) return;
    if (!D) { el("b3msg").style.display = "block"; el("b3msg").textContent = "Büyük Hareket 3 verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["İzleyen stop", cnt(e => e.st === "stop"), e => e.st === "stop"],
      ["Süre doldu", cnt(e => e.st === "süre"), e => e.st === "süre"], ["Canlı", cnt(e => e.live), e => e.live]];
    el("b3chips").innerHTML = chips.map(([t, n]) => `<button aria-pressed="${t === B.chip}" data-t="${t}">${t} (${n})</button>`).join("");
    const f = (chips.find(c => c[0] === B.chip) || chips[0])[2];
    const list = D.entries.filter(f);
    el("b3list").innerHTML = list.slice(0, B.shown).map(e => row(e, P)).join("");
    el("b3more").hidden = list.length <= B.shown;
    el("b3msg").style.display = list.length ? "none" : "block";
    el("b3msg").textContent = D.n ? "Bu filtrede sinyal yok." : "Henüz sinyal yok. Altı şart birlikte nadiren sağlanır, çıktıkça burada birikir.";
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
    el("b3stats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda sinyal<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" + (S.open_bench == null ? "" : `<br>evren aynı sürede ${pct(S.open_bench, 1)}, fark <span class="${sgn(S.open_diff)}">${pct(S.open_diff, 1)}</span>`) : "açık işlem yok"}</div></div>
      <div class="edge">Aynı dönemde Büyük Hareket 2 (alt sektör şartı olmadan): ${D.n_bh2} sinyal. Büyük Hareket 3: ${D.n}. Alt sektör bilgisi ${D.sektor_bilinen} hissede var.</div>
      ${done}
      <div class="edge">Kapanan işlemler izleyen stop ya da 84. günle bitmiş olanlardır. Evren = işlem hacmi en düşük %30 hariç tüm hisselerin, aynı gün girişle aynı sürede eşit ağırlıklı ortalaması. Az işlemde (30'dan az) sonuçlara güvenme.${part}</div></div>`;
  }

  el("b3chips").addEventListener("click", e => { const t = e.target.dataset.t; if (t) { B.chip = t; B.shown = 30; render(); } });
  el("b3more").onclick = () => { B.shown += 30; render(); };

  const _tab = window.tab;
  window.tab = function (which) {
    _tab(which);
    el("p-buyuk3").hidden = which !== "buyuk3";
    el("t-buyuk3").setAttribute("aria-selected", which === "buyuk3");
  };
  el("t-buyuk3").onclick = () => window.tab("buyuk3");

  // data her yenilemede yeni nesne olur; değişince yeniden çiz
  setInterval(() => { if (data && data !== B.last) { B.last = data; render(); } }, 500);
})();
