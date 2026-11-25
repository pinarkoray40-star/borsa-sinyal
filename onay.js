// onay.js - "Onaylı" sekmesi. index.html'deki data, $, nf, pct, sgn, spark ve tab'ı kullanır. buyuk2.js'den SONRA yüklenmeli.
(function () {
  const sec = document.getElementById("p-onay");
  if (!sec || !document.getElementById("t-onay")) return;
  sec.innerHTML = `
    <div class="good"><b>Onaylı</b> (günlük veri): <b>Büyük Hareket 2</b> ile aynı sinyal (oynak hisse, 20 günlük zirve kırılımı, hacim kuruması, 60 günlük zirveye yakınlık, dayanıklılık sırası %70 ve üstü). <b>Tek fark giriş zamanı:</b> sinyal günü hemen girmiyoruz. Sinyalden sonraki <b>teyit gününde</b> (şu an ertesi işlem günü) kapanış, sinyal günü kapanışının <b>üstündeyse</b> o kapanıştan giriyoruz. Altındaysa sinyal atlanıyor. Amaç, sinyalin hemen ertesi gün zayıflayan hisseleri elemek.</div>
    <div class="good"><b>Takip kuralı</b> Büyük Hareket 2 ile aynı: kapanış, girişten sonraki en yüksek kapanıştan <b>%25 aşağı</b> inerse <b>İZLEYEN STOP</b> ile çıkılır; olmazsa 84. işlem gününde SÜRE DOLDU. Sabit hedef yok. Sayım giriş gününden (teyit günü) başlar.</div>
    <div class="warn"><b>Önce bunu bil.</b> Bu fikir geçmişte test edilmeden, "hemen zayıflayanı eleyelim" mantığıyla eklendi. Büyük kazananlar çoğu zaman sinyalden hemen sonra hızlı çıkar; teyit beklemek onları daha pahalıdan almaya ya da hiç yakalayamamaya yol açabilir. Ortalamayı düşürebilir. Aşağıdaki geçmiş sonuçlar son 6 ayın aynı sinyalleridir; küçük örnek, zayıf piyasa. Asıl kanıt <b>"canlı"</b> işaretli sinyallerdir (25.11.2026'dan sonra çıkanlar). Büyük Hareket 2 ile yan yana izle. Kâr/zarar komisyonsuz, kapanışlara göredir. Yatırım tavsiyesi değildir.</div>
    <div id="onpend"></div>
    <div id="onstats"></div>
    <div class="chips" id="onchips"></div>
    <div id="onlist"></div>
    <button class="more" id="onmore" hidden>Daha fazla göster</button>
    <div class="msg" id="onmsg">Yükleniyor…</div>`;

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
        <div class="meta">giriş ${dmy(e.date)} (sinyal ${dmy(e.sdate)}, sinyal fiyatı ${nf(e.sig_px)}) · giriş ${nf(e.entry)} · zirve ${nf(e.peak_px)} · stop ${nf(e.stop_px)} · dayanıklılık sırası %${e.dnr}${kv}</div></div>
      <div class="px"><span class="${sgn(e.pnl)}">${pct(e.pnl, 1)}</span><div class="meta">${e.st === "açık" ? "şu an" : "çıkış"}</div></div>
    </summary>
    <div class="detail">${e.path.length > 1 ? spark(e.path.map(x => x == null ? 0 : x)) : ""}
      <ul class="why"><li>${why}</li><li>Bu süre içindeki en yüksek ${pct(e.max, 1)}, en düşük ${pct(e.min, 1)} (giriş: onay günü kapanışı).</li><li>Sinyal ${dmy(e.sdate)} günü ${nf(e.sig_px)} kapanışla geldi; onay günü kapanış ${nf(e.entry)} ile sinyal fiyatının üstünde kaldı, bu yüzden girildi (sinyal fiyatına göre ${pct((e.entry / e.sig_px - 1) * 100, 1)}).</li>${e.bench == null ? "" : `<li>Kıyas: aynı gün giriş yapılsa tüm likit hisselerin ortalaması aynı sürede ${pct(e.bench, 1)} olurdu; bu işlem ondan ${pct(e.diff, 1)} puan ${e.diff >= 0 ? "iyi" : "kötü"}.</li>`}<li>Dayanıklılık sırası %${e.dnr}: son 120 günde endeks düşerken, hisselerin %${e.dnr}'inden daha iyi gitmiş.</li></ul></div></details>`;
  }

  function render() {
    const D = data && data.onay;
    if (!data) return;
    if (!D) { el("onmsg").style.display = "block"; el("onmsg").textContent = "Onaylı verisi şu an yok. Birkaç dakika sonra tekrar dene."; return; }
    const P = D.params, S = D.stats;
    const cnt = f => D.entries.filter(f).length;
    const chips = [["Hepsi", D.entries.length, () => true], ["Açık", cnt(e => e.st === "açık"), e => e.st === "açık"],
      ["İzleyen stop", cnt(e => e.st === "stop"), e => e.st === "stop"],
      ["Süre doldu", cnt(e => e.st === "süre"), e => e.st === "süre"], ["Canlı", cnt(e => e.live), e => e.live]];
    el("onchips").innerHTML = chips.map(([t, n]) => `<button aria-pressed="${t === B.chip}" data-t="${t}">${t} (${n})</button>`).join("");
    const f = (chips.find(c => c[0] === B.chip) || chips[0])[2];
    const list = D.entries.filter(f);
    el("onlist").innerHTML = list.slice(0, B.shown).map(e => row(e, P)).join("");
    el("onmore").hidden = list.length <= B.shown;
    el("onmsg").style.display = list.length ? "none" : "block";
    el("onmsg").textContent = D.n ? "Bu filtrede sinyal yok." : "Henüz onaylı giriş yok. Sinyal nadir gelir ve teyit gerekir, çıktıkça burada birikir.";

    const pend = D.pending && D.pending.length
      ? `<div class="card"><b>Onay bekleyen sinyaller (${D.pending.length})</b>
          <div class="meta">Sinyal geldi, teyit günü henüz kapanmadı. Teyit günü kapanış sinyal fiyatının üstündeyse giriş sayılır, değilse atlanır.</div>
          ${D.pending.map(x => `<div class="meta">${x.symbol} · sinyal ${dmy(x.date)} · sinyal fiyatı ${nf(x.signal_px)} · dayanıklılık sırası %${x.dnr}</div>`).join("")}</div>`
      : "";
    el("onpend").innerHTML = pend;

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
    const skip = S.skipped
      ? `<div class="edge"><b>Elenen sinyaller:</b> ${S.n_signals} sinyalden ${S.skipped} tanesi onay vermedi, girilmedi.${S.skip_done ? ` Bunlar onaylılarla aynı günden (onay günü kapanışından) girilseydi sonuçlanan ${S.skip_done} işlemin ortalaması ${pct(S.skip_avg, 1)}, medyanı ${pct(S.skip_median, 1)}, kazanan oranı %${nf(S.skip_win, 0)} olurdu${S.n_done ? `; onaylı girilen kapanmış işlemlerin ortalaması ${pct(S.avg, 1)}, medyanı ${pct(S.median, 1)}, kazanan oranı %${nf(S.win, 0)}` : ""}. Elenenler girilenlerden belirgin kötüyse onay işe yarıyor demektir; benzerse sadece iyi işlemleri de kaçırıyoruz demektir.` : ""}</div>`
      : "";
    const bh2 = S.bh2_done
      ? `<div class="edge"><b>Asıl kıyas:</b> aynı sinyallerin hepsine sinyal günü kapanışından girseydik (Büyük Hareket 2 kuralı), sonuçlanan ${S.bh2_done} işlemin ortalaması ${pct(S.bh2_avg, 1)}, medyanı ${pct(S.bh2_median, 1)}, kazanan oranı %${nf(S.bh2_win, 0)} olurdu${S.n_done ? `; onaylı giriş: ortalama ${pct(S.avg, 1)}, medyan ${pct(S.median, 1)}, kazanan %${nf(S.win, 0)}` : ""}. Onaylı giriş bunlardan belirgin iyiyse bekleme işe yarıyor demektir; benzerse ya da kötüyse gerek yok.</div>`
      : "";
    el("onstats").innerHTML = `<div class="card"><div class="grid2">
        <div>Son 6 ayda onaylı giriş<b>${D.n}</b>canlı: ${D.n_live} · kapanan: ${S.n_done}</div>
        <div>Açık işlemler<b class="${sgn(S.open_avg)}">${S.n_open ? pct(S.open_avg, 1) : "—"}</b>${S.n_open ? S.n_open + " işlem, ortalama şu anki durum" + (S.open_bench == null ? "" : `<br>evren aynı sürede ${pct(S.open_bench, 1)}, fark <span class="${sgn(S.open_diff)}">${pct(S.open_diff, 1)}</span>`) : "açık işlem yok"}</div></div>
      ${done}
      ${skip}
      ${bh2}
      <div class="edge">Teyit süresi: sinyalden ${P.confirm} işlem günü sonra. Kapanan işlemler izleyen stop ya da 84. günle bitmiş olanlardır. Evren = işlem hacmi en düşük %30 hariç tüm hisselerin, aynı gün girişle aynı sürede eşit ağırlıklı ortalaması. Az işlemde (30'dan az) sonuçlara güvenme. Tek bir büyük kazanan ortalamayı çok değiştirir, medyana da bak.${part}</div></div>`;
  }

  el("onchips").addEventListener("click", e => { const t = e.target.dataset.t; if (t) { B.chip = t; B.shown = 30; render(); } });
  el("onmore").onclick = () => { B.shown += 30; render(); };

  const _tab = window.tab;
  window.tab = function (which) {
    _tab(which);
    el("p-onay").hidden = which !== "onay";
    el("t-onay").setAttribute("aria-selected", which === "onay");
  };
  el("t-onay").onclick = () => window.tab("onay");

  // data her yenilemede yeni nesne olur; değişince yeniden çiz
  setInterval(() => { if (data && data !== B.last) { B.last = data; render(); } }, 500);
})();
