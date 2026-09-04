function initializeReveal() {
  const targets = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    targets.forEach((target) => target.classList.add("visible"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12 });

  targets.forEach((target) => observer.observe(target));
}

function initializeCitation() {
  const button = document.querySelector("#copy-citation");
  const text = document.querySelector("#citation-text")?.innerText;
  if (!button || !text) return;

  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select and copy";
    }
    window.setTimeout(() => { button.textContent = "Copy BibTeX"; }, 1800);
  });
}

initializeReveal();
initializeCitation();
