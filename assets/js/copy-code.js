document.addEventListener("DOMContentLoaded", function () {
  console.log("copy-code.js loaded");

  const codeBlocks = document.querySelectorAll("pre");

  async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-9999px";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }

    document.body.removeChild(textarea);
    if (!ok) throw new Error("Fallback copy failed");
    return true;
  }

  function bindCopyHandler(button, code) {
    if (!button || !code) return;
    if (button.dataset.copyBound === "true") return;
    button.dataset.copyBound = "true";

    button.addEventListener("click", async function (e) {
      console.log("copy button clicked");

      e.preventDefault();
      e.stopPropagation();

      const text = code.innerText || code.textContent || "";

      try {
        await copyText(text);

        button.classList.remove("copied");
        void button.offsetWidth;
        button.classList.add("copied");

        clearTimeout(button._copiedTimer);
        button._copiedTimer = setTimeout(() => {
          button.classList.remove("copied");
        }, 1000);
      } catch (err) {
        console.error("复制失败：", err);
      }
    });
  }

  codeBlocks.forEach((pre) => {
    const code = pre.querySelector("code");
    if (!code) return;

    const wrapper =
      pre.closest(".highlighter-rouge") ||
      pre.closest(".highlight") ||
      pre.closest("figure.highlight") ||
      pre;

    let button = wrapper.querySelector(".copy-btn");

    if (!button) {
      button = document.createElement("button");
      button.className = "copy-btn";
      button.type = "button";
      button.textContent = "复制代码";
      button.setAttribute("aria-label", "复制代码");
      wrapper.appendChild(button);
    }

    bindCopyHandler(button, code);
  });
});