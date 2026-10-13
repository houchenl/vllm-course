"use strict";
(() => {
  const format = n => new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 5}).format(n);
  for (const figure of document.querySelectorAll("[data-calculator]")) {
    const form = figure.querySelector("form");
    const output = figure.querySelector("output");
    function update() {
      const values = {};
      let valid = true;
      for (const input of form.elements) {
        if (!input.name) continue;
        const value = Number(input.value);
        if (input.value === "" || !input.checkValidity() || !Number.isFinite(value)) valid = false;
        values[input.name] = value;
      }
      const kind = figure.dataset.calculator;
      const integerKeys = kind === "budget" ? ["budget", "a", "b", "c"] :
        kind === "pages" ? ["tokens", "requests", "block"] :
        kind === "collective" ? ["ranks", "tokens", "hidden", "bytes"] : [];
      if (integerKeys.some(key => !Number.isSafeInteger(values[key]))) valid = false;
      if (!valid) { output.textContent = "请输入范围内的有限数值；数量必须为整数。"; return; }
      if (kind === "roofline") {
        const {flops, bytes, compute, bandwidth} = values;
        const traffic = bytes * 2 ** 20;
        const intensity = flops * 1e9 / traffic;
        const tc = flops * 1e9 / (compute * 1e12) * 1e6;
        const tm = traffic / (bandwidth * 1e9) * 1e6;
        output.textContent = `I=${format(intensity)} FLOP/byte；计算下界 ${format(tc)} μs；访存下界 ${format(tm)} μs；双下界 ${format(Math.max(tc, tm))} μs。`;
      } else if (kind === "budget") {
        let remaining = values.budget;
        const results = ["a", "b", "c"].map(key => {
          const count = Math.min(values[key], remaining);
          remaining -= count;
          return `${key.toUpperCase()}=${format(count)}`;
        });
        output.textContent = `本轮计划 ${results.join("、")}；剩余 ${format(remaining)}。只演示固定顺序分配。`;
      } else if (kind === "pages") {
        const {tokens, requests, block, kib} = values;
        const blocks = Math.ceil(tokens / block) * requests;
        const slots = blocks * block;
        const logical = tokens * requests * kib;
        const allocated = slots * kib;
        if (![blocks, slots, tokens * requests].every(Number.isSafeInteger)) {
          output.textContent = "超过精确整数范围，请缩小输入。"; return;
        }
        output.textContent = `${format(blocks)} 块、${format(slots)} 槽；逻辑 ${format(logical)} KiB、取整 ${format(allocated)} KiB、尾块 ${format(allocated - logical)} KiB。`;
      } else if (kind === "collective") {
        const {ranks, tokens, hidden, bytes} = values;
        const payload = tokens * hidden * bytes;
        if (!Number.isSafeInteger(payload)) { output.textContent = "超过精确整数范围，请缩小输入。"; return; }
        output.textContent = `payload=${format(payload / 1024)} KiB；理想 ring 每 rank 流量≈${format(2 * (ranks - 1) / ranks * payload / 1024)} KiB。`;
      }
    }
    form.addEventListener("input", update);
    form.addEventListener("submit", event => event.preventDefault());
    update();
  }
})();
