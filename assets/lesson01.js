"use strict";
(() => {
  const states = [
    {title: "步骤 1：Prefill", input: "x₁、x₂、x₃、x₄", cache: ["x₁", "x₂", "x₃", "x₄"], output: "y₁", description: "已计算 4 个位置，已知输入与输出共 5 个 token；下一次需要计算 y₁。"},
    {title: "步骤 2：第一次 Decode", input: "y₁", cache: ["x₁", "x₂", "x₃", "x₄", "y₁"], output: "y₂", description: "y₁ 已输入模型并拥有 KV。已计算 5 个位置；刚采样的 y₂ 尚未拥有自己的 KV。"},
    {title: "步骤 3：第二次 Decode", input: "y₂", cache: ["x₁", "x₂", "x₃", "x₄", "y₁", "y₂"], output: "y₃", description: "已得到 3 个输出，经历 1 次完整 Prefill 与 2 次 Decode。KV 有 6 个位置（P+N−1）；最后输出 y₃ 尚未写入 KV。"}
  ];
  let step = 0;
  const byId = id => document.getElementById(id);
  function token(text, output = false) {
    const span = document.createElement("span");
    span.className = output ? "token output" : "token";
    span.textContent = text;
    return span;
  }
  function renderStep() {
    const state = states[step];
    byId("step-title").textContent = state.title;
    byId("step-input").textContent = "本次输入：" + state.input;
    byId("kv-tokens").replaceChildren(...state.cache.map(value => token(value)));
    byId("sample-token").replaceChildren(token(state.output, true));
    byId("step-explanation").textContent = state.description;
    byId("step-counter").textContent = `${step + 1} / ${states.length}`;
    byId("step-prev").disabled = step === 0;
    byId("step-next").disabled = step === states.length - 1;
  }
  byId("step-prev").addEventListener("click", () => {if (step > 0) {step--; renderStep();}});
  byId("step-next").addEventListener("click", () => {if (step < states.length - 1) {step++; renderStep();}});

  const fields = ["kv-layers", "kv-heads", "kv-dim", "kv-bytes", "kv-tokens-input", "kv-requests", "kv-block"];
  const format = value => new Intl.NumberFormat("en-US", {maximumFractionDigits: 8}).format(value);
  function calculate() {
    const inputs = fields.map(byId);
    const values = inputs.map(input => Number(input.value));
    const valid = inputs.every((input, i) => input.value !== "" && input.checkValidity() && Number.isSafeInteger(values[i]) && values[i] > 0);
    if (!valid) {
      byId("kv-results").hidden = true;
      byId("kv-error").textContent = "请输入范围内的正整数；元素字节数从下拉框选择。";
      return;
    }
    const [layers, heads, dim, bytes, positions, requests, block] = values;
    const perToken = 2 * layers * heads * dim * bytes;
    const slotsPerRequest = Math.ceil(positions / block) * block;
    const ideal = perToken * positions * requests;
    const rounded = perToken * slotsPerRequest * requests;
    if (![perToken, ideal, rounded].every(Number.isSafeInteger)) {
      byId("kv-results").hidden = true;
      byId("kv-error").textContent = "计算超过浏览器的精确整数范围，请缩小输入；本工具不返回不可靠结果。";
      return;
    }
    byId("kv-results").hidden = false;
    byId("kv-error").textContent = "";
    byId("kv-per-token").textContent = format(perToken / 2 ** 10) + " KiB";
    byId("kv-ideal").textContent = format(ideal / 2 ** 30) + " GiB";
    byId("kv-rounded").textContent = format(rounded / 2 ** 30) + " GiB";
    byId("kv-waste").textContent = `每请求 ${format(slotsPerRequest)} 个槽位；尾块空槽载荷共 ${format((rounded - ideal) / 2 ** 20)} MiB。`;
  }
  function preset(values) {fields.forEach((id, i) => {byId(id).value = values[i];}); calculate();}
  byId("kv-form").addEventListener("input", calculate);
  byId("kv-form").addEventListener("change", calculate);
  byId("kv-form").addEventListener("submit", event => event.preventDefault());
  byId("preset-toy").addEventListener("click", () => preset([32, 8, 128, 2, 4096, 8, 16]));
  byId("preset-qwen").addEventListener("click", () => preset([28, 8, 128, 2, 1024, 4, 16]));
  byId("preset-tail").addEventListener("click", () => preset([32, 8, 128, 2, 4097, 8, 16]));
  calculate();
})();
