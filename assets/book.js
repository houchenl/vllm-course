"use strict";
(() => {
  const printButton = document.getElementById("print-chapter");
  if (printButton) printButton.addEventListener("click", () => window.print());
  const toc = document.querySelector(".toc details");
  if (toc && window.matchMedia("(max-width: 760px)").matches) toc.open = false;
  const headings = [...document.querySelectorAll("main section[id]")];
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      document.querySelectorAll(".toc a").forEach(link => {
        const active = link.getAttribute("href") === "#" + visible[0].target.id;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }, {rootMargin: "-5% 0px -65% 0px", threshold: 0});
    headings.forEach(heading => observer.observe(heading));
  }
  const illustrationButtons = [...document.querySelectorAll(".illustration-open")];
  if (illustrationButtons.length) {
    const dialog = document.createElement("dialog");
    dialog.className = "illustration-dialog no-print";
    dialog.setAttribute("aria-labelledby", "illustration-dialog-title");
    dialog.innerHTML = '<h3 id="illustration-dialog-title">教学插图</h3><div class="controls"><button type="button" data-size="fit">适应窗口</button><button type="button" data-size="actual">原始尺寸</button><button type="button" data-close autofocus>关闭</button></div><div class="illustration-zoom-scroll"><img alt=""></div>';
    document.body.append(dialog);
    const image = dialog.querySelector("img");
    const scroller = dialog.querySelector(".illustration-zoom-scroll");
    dialog.querySelector("[data-close]").addEventListener("click", () => dialog.close());
    dialog.querySelectorAll("[data-size]").forEach(button => {
      button.addEventListener("click", () => {
        image.classList.toggle("actual-size", button.dataset.size === "actual");
        scroller.scrollTo(0, 0);
      });
    });
    dialog.addEventListener("click", event => {if (event.target === dialog) dialog.close();});
    illustrationButtons.forEach(button => {
      button.addEventListener("click", () => {
        const original = button.querySelector("img");
        image.src = original.currentSrc || original.src;
        image.alt = original.alt;
        image.classList.remove("actual-size");
        dialog.querySelector("h3").textContent = button.getAttribute("aria-label").replace(/^放大/, "");
        dialog.showModal();
        scroller.scrollTo(0, 0);
      });
    });
  }
  let printAnswers = [];
  let printing = false;
  window.addEventListener("beforeprint", () => {
    if (printing) return;
    printing = true;
    printAnswers = [...document.querySelectorAll("details.answer")].map(details => [details, details.open]);
    printAnswers.forEach(([details]) => {details.open = true;});
  });
  window.addEventListener("afterprint", () => {
    if (!printing) return;
    printAnswers.forEach(([details, open]) => {details.open = open;});
    printAnswers = [];
    printing = false;
  });
})();
