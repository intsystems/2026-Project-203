(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const evidence = new Map(data.evidence.map(item => [item.id, item]));
  const cards = Array.from(document.querySelectorAll(".point-card"));
  const marks = Array.from(document.querySelectorAll(".source-highlight"));
  const passageLinks = Array.from(document.querySelectorAll(".passage-link"));
  const previous = document.getElementById("previous-passage");
  const next = document.getElementById("next-passage");
  const back = document.getElementById("back-to-summary");
  const status = document.getElementById("active-point");
  const counter = document.getElementById("passage-position");
  const toggle = document.getElementById("toggle-highlights");
  let activePoint = null;
  let activeEvidence = null;
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  const summaryId = point => `summary-${point.replaceAll(".", "-")}`;
  const pointLabel = point => point === "overview" ? "Overall assessment" : point;
  const byPoint = point => data.evidence.filter(item => item.point === point);
  const pane = document.querySelector(".original-pane");
  const workspaceToolbar = document.querySelector(".workspace-toolbar");
  if (typeof ResizeObserver === "function") new ResizeObserver(() => {
    pane.style.setProperty("--workspace-toolbar-height", Math.ceil(workspaceToolbar.getBoundingClientRect().height) + "px");
  }).observe(workspaceToolbar);

  function showView(responses) {
    document.getElementById("review-view").hidden = responses;
    document.getElementById("responses").hidden = !responses;
    document.getElementById("view-review").toggleAttribute("aria-current", !responses);
    document.getElementById("view-responses").toggleAttribute("aria-current", responses);
    const current = document.getElementById(responses ? "view-responses" : "view-review");
    current.setAttribute("aria-current", "page");
  }

  function select(point, evidenceId = null) {
    activePoint = point;
    activeEvidence = evidenceId;
    const matching = byPoint(point);
    cards.forEach(card => card.classList.toggle("is-active", card.dataset.point === point));
    const card = document.getElementById(summaryId(point));
    if (card) card.querySelector("details").open = true;
    marks.forEach(mark => {
      mark.classList.toggle("is-point", mark.dataset.points.split(" ").includes(point));
      mark.classList.toggle("is-target", Boolean(evidenceId) && mark.dataset.evidence.split(" ").includes(evidenceId));
    });
    passageLinks.forEach(link => {
      const current = link.hash === `#${evidenceId}`;
      link.classList.toggle("is-active", current);
      if (current) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
    status.textContent = point === "overview" ? "Overall assessment" : point ? `${pointLabel(point)} · ${data.pointTitles[point]}` : "Choose a summary point";
    const index = matching.findIndex(item => item.id === evidenceId);
    counter.textContent = index >= 0 ? `Passage ${index + 1} of ${matching.length} · ${evidence.get(evidenceId).location}` : `${matching.length} linked passages`;
    previous.disabled = matching.length < 2;
    next.disabled = matching.length < 2;
    back.href = point ? `#${summaryId(point)}` : "#summary";
    back.textContent = point ? `↩ ${pointLabel(point)}` : "↩ Summary";
    back.setAttribute("aria-label", point ? `Back to summary: ${pointLabel(point)}` : "Back to summary");
  }

  function followHash(shouldScroll) {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); }
    catch (_) { return; }
    const item = evidence.get(id);
    let target = null;
    if (item) {
      showView(false);
      select(item.point, item.id);
      target = document.getElementById(item.id);
      if (shouldScroll) document.getElementById(summaryId(item.point)).scrollIntoView({behavior: "auto", block: "nearest"});
    } else {
      const card = document.getElementById(id);
      if (card && card.classList.contains("response-card")) {
        showView(true);
        const retained = activePoint === card.dataset.responsePoint ? activeEvidence : null;
        select(card.dataset.responsePoint, retained);
        document.getElementById(summaryId(card.dataset.responsePoint)).scrollIntoView({behavior: "auto", block: "nearest"});
        target = card;
      } else if (card && card.dataset.assembledPoint) {
        showView(true);
        select(card.dataset.assembledPoint);
        document.getElementById(summaryId(card.dataset.assembledPoint)).scrollIntoView({behavior: "auto", block: "nearest"});
        target = card;
      } else if (id === "responses" || id === "assembled-draft" || id === "feedback-plan" || card && card.closest(".feedback-plan")) {
        showView(true);
        if (id === "responses") pane.scrollTop = 0;
        target = card;
      } else if (card && card.classList.contains("assembled-heading")) {
        showView(true);
        target = card;
      } else if (id === "original-review" || id.startsWith("field-")) {
        showView(false);
        if (id === "original-review") pane.scrollTop = 0;
        target = card;
      } else if (card && card.classList.contains("point-card")) {
        const retained = activePoint === card.dataset.point ? activeEvidence : null;
        select(card.dataset.point, retained);
        target = card;
      } else if (card) {
        target = card;
      }
    }
    if (target && shouldScroll) {
      target.scrollIntoView({behavior: motion, block: item ? "center" : "start"});
      target.focus({preventScroll: true});
    }
  }

  document.addEventListener("click", event => {
    const anchor = event.target.closest('a[href^="#"]');
    if (!anchor || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    const id = anchor.hash.slice(1);
    if (!document.getElementById(id)) return;
    event.preventDefault();
    if (location.hash !== anchor.hash) history.pushState(null, "", anchor.hash);
    followHash(true);
  });

  function step(direction) {
    if (!activePoint) return;
    const matching = byPoint(activePoint);
    if (!matching.length) return;
    const current = matching.findIndex(item => item.id === activeEvidence);
    const offset = current < 0 ? (direction > 0 ? -1 : 0) : current;
    const item = matching[(offset + direction + matching.length) % matching.length];
    history.pushState(null, "", `#${item.id}`);
    followHash(true);
  }
  previous.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  toggle.addEventListener("click", () => {
    const disabled = document.body.classList.toggle("highlights-off");
    toggle.setAttribute("aria-pressed", String(!disabled));
    toggle.textContent = disabled ? "Highlights off" : "Highlights on";
  });
  document.querySelector(".print-button").addEventListener("click", () => window.print());
  window.addEventListener("hashchange", () => followHash(true));
  window.addEventListener("popstate", () => followHash(true));
  document.addEventListener("responses-ready", () => followHash(Boolean(location.hash)));
  followHash(Boolean(location.hash));
})();
