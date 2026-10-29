(() => {
  "use strict";
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const clone = value => JSON.parse(JSON.stringify(value));
  const key = "rebuttal-v1:" + data.workspaceId + ":" + data.stem;
  const fileMode = location.protocol === "file:";
  const sourceFile = data.sourceFile || "response_fragments.json";
  const limit = data.limit;
  const status = document.getElementById("save-status");
  const saveButton = document.getElementById("save-to-repo");
  const Draft = window.RebuttalDraft;
  let state = clone(data.response);
  let baseRevision = state.revision;
  let dirty = false;
  let ready = false;
  let connected = false;
  let saving = false;
  let conflicted = false;
  let remoteConflict = null;
  let changes = 0;
  let timer;
  let storageAvailable = true;
  let lastResult;
  let directFolder = null;
  let fileSaveActive = false;
  let serverOutdated = false;
  let draftSelection = null;

  const linkPattern = /\b(?:https?:\/\/|ftp:\/\/|www\.|mailto:)|\[[^\]\n]+\]\([^\n)]+\)|\b(?:[a-z0-9-]+\.)+(?:com|org|net|edu|gov|io|ai|cc|dev|ru|uk)\b/i;
  const placeholders = /\b(?:TODO|TBD|FIXME|PLACEHOLDER)\b/i;
  const fragment = id => state.fragments.find(item => item.id === id);

  function sameShape(review) {
    return review && review.target === data.response.target &&
      Array.isArray(review.fragments) && Array.isArray(review.sections) &&
      review.fragments.map(item => item.id).join("|") === data.response.fragments.map(item => item.id).join("|") &&
      review.sections.map(item => item.id).join("|") === data.response.sections.map(item => item.id).join("|") &&
      typeof review.intro === "string" && typeof review.closing === "string" &&
      review.fragments.every(item => ["text", "notes", "joiner", "section"].every(field => typeof item[field] === "string") &&
        ["include", "needs_check"].every(field => typeof item[field] === "boolean") &&
        review.sections.some(section => section.id === item.section)) &&
      review.sections.every(item => typeof item.lead === "string") && Draft.valid(review) &&
      (!review.frame_variants || ["intro", "closing"].every(field =>
        JSON.stringify((review.frame_variants[field] || []).map(item => item.id)) ===
        JSON.stringify((data.response.frame_variants?.[field] || []).map(item => item.id)))) &&
      (!review.reply_topics || review.reply_topics.map(item => item.id).join("|") === data.response.reply_topics.map(item => item.id).join("|") &&
        review.reply_topics.every((item, index) => ["points", "question", "source", "occurrence"].every(field =>
          JSON.stringify(item[field]) === JSON.stringify(data.response.reply_topics[index][field]))));
  }

  function cache() {
    try {
      localStorage.setItem(key, JSON.stringify({base_revision: baseRevision, dirty, review: state}));
      storageAvailable = true;
    }
    catch (_) {
      storageAvailable = false;
      status.textContent = "Browser storage unavailable · download a backup";
      document.getElementById("file-mode-note").hidden = false;
    }
  }

  function readCache() {
    try {
      const stored = JSON.parse(localStorage.getItem(key) || "null");
      return stored && sameShape(stored.review) ? stored : null;
    } catch (_) { return null; }
  }

  function appendStrong(element, text) {
    let offset = 0;
    for (const match of text.matchAll(/\*\*([^\n]+?)\*\*/g)) {
      element.append(document.createTextNode(text.slice(offset, match.index)));
      const strong = document.createElement("strong");
      strong.textContent = match[1];
      element.append(strong);
      offset = match.index + match[0].length;
    }
    element.append(document.createTextNode(text.slice(offset)));
  }

  function renderMath(element, text) {
    const lines = text.split("\n");
    const cells = line => typeof line === "string" && /^\s*\|.*\|\s*$/.test(line) && !line.includes("\\|")
      ? line.trim().slice(1, -1).split("|").map(cell => cell.trim()) : null;
    const content = document.createDocumentFragment();
    let plain = [];
    const flush = () => {
      if (plain.length) appendStrong(content, plain.join("\n"));
      plain = [];
    };
    for (let i = 0; i < lines.length;) {
      const header = cells(lines[i]);
      const separator = cells(lines[i + 1]);
      if (!header || !separator || header.length !== separator.length ||
          !separator.every(cell => /^:?-{3,}:?$/.test(cell))) {
        plain.push(lines[i++]);
        continue;
      }
      flush();
      const table = document.createElement("table");
      table.className = "preview-table";
      const appendRow = (parent, values, tag) => {
        const row = document.createElement("tr");
        values.forEach((value, column) => {
          const cell = document.createElement(tag);
          appendStrong(cell, value);
          if (tag === "th") cell.scope = "col";
          cell.style.textAlign = separator[column].endsWith(":")
            ? (separator[column].startsWith(":") ? "center" : "right") : "left";
          row.append(cell);
        });
        parent.append(row);
      };
      const head = table.createTHead();
      appendRow(head, header, "th");
      const body = table.createTBody();
      i += 2;
      for (let row; i < lines.length && (row = cells(lines[i])) && row.length === header.length; i++) {
        appendRow(body, row, "td");
      }
      const wrapper = document.createElement("div");
      wrapper.className = "preview-table-scroll";
      wrapper.append(table);
      content.append(wrapper);
    }
    flush();
    element.replaceChildren(content);
    if (typeof renderMathInElement !== "function") return;
    renderMathInElement(element, {
      delimiters: [{left: "$$", right: "$$", display: true}, {left: "\\[", right: "\\]", display: true},
                   {left: "\\(", right: "\\)", display: false}, {left: "$", right: "$", display: false}],
      throwOnError: false, trust: false, maxExpand: 1000,
      errorCallback: () => {},
    });
  }

  function assemble(review = state) {
    const body = Draft.body(review);
    const selected = review.fragments.filter(item => item.include);
    const omitted = review.fragments.filter(item => !item.include).map(item => item.id);
    const pending = review.fragments.filter(item => item.needs_check).map(item => item.id);
    const issues = [];
    if (!body.trim() || !selected.length) issues.push("Select at least one response fragment.");
    if (limit !== null && body.length > limit) issues.push(`The assembled draft exceeds ${limit.toLocaleString("en-US")} characters.`);
    if (data.forbidExternalLinks && linkPattern.test(body)) issues.push("The assembled draft contains a link or address.");
    if (placeholders.test(body) || body.includes("\uFEFF")) issues.push("The assembled draft contains a placeholder or byte-order mark.");
    for (const item of selected) {
      if (!item.text.trim()) issues.push(item.id + " has an empty response.");
      if (item.needs_check) issues.push(item.id + " still needs an evidence check.");
    }
    return {body, characters: body.length, words: body.trim() ? body.trim().split(/\s+/).length : 0, issues, can_export: !issues.length,
      included: selected.map(item => item.id), omitted, pending};
  }

  function preview() {
    const container = document.getElementById("assembled-preview");
    container.replaceChildren();
    Draft.spans(state).forEach((span, index) => {
      const block = span.block;
      if (block.kind === "heading") {
        const item = state.reply_topics.find(item => item.id === block.topic);
        const section = document.createElement("section"); section.className = "assembled-heading";
        section.id = "assembled-topic-" + item.id;
        const heading = document.createElement("h3"); renderMath(heading, item.heading);
        const editor = document.createElement("a"); editor.href = "#topic-" + item.id; editor.textContent = "Edit this heading";
        const question = document.createElement("a"); question.href = "#" + data.topicPassages[item.id]; question.textContent = "Reviewer’s question";
        const full = document.createElement("button"); full.type = "button"; full.dataset.editDraft = index; full.textContent = "Edit in full text";
        section.append(heading, editor, question, full); container.append(section); return;
      }
      const row = document.createElement("section");
      row.className = "assembled-answer";
      const objection = document.createElement("div");
      objection.className = "assembled-objection";
      const content = document.createElement("div");
      content.className = "assembled-reply";
      const points = block.points || [];
      row.dataset.previewPoints = points.join(" ");
      const anchor = (href, label) => {
        const node = document.createElement("a"); node.href = href; node.textContent = label; return node;
      };
      if (points.length) {
        points.forEach(id => {
          const concern = document.createElement("div");
          concern.id = "assembled-" + id.replaceAll(".", "-");
          concern.dataset.assembledPoint = id; concern.tabIndex = -1;
          const heading = document.createElement("h3");
          heading.textContent = id + " · " + data.pointTitles[id];
          const links = document.createElement("div"); links.className = "assembled-links";
          const evidence = data.evidence.find(item => item.point === id);
          links.append(anchor("#summary-" + id.replaceAll(".", "-"), "Summary"),
            anchor("#" + evidence.id, "Exact words"), anchor("#response-" + id.replaceAll(".", "-"), "Edit answer"));
          concern.append(heading, links); objection.append(concern);
        });
      } else {
        const heading = document.createElement("h3");
        heading.textContent = block.kind === "intro" ? "Opening" : block.kind === "closing" ? "Closing" : "Additional text";
        objection.append(heading);
      }
      if (points.length > 1) {
        const shared = document.createElement("p"); shared.className = "shared-answer-note";
        shared.textContent = "These concerns share this answer. Editing it changes their linked response fields.";
        objection.append(shared);
      }
      const rendered = document.createElement("div"); rendered.className = "response-preview";
      renderMath(rendered, Draft.text(state, block));
      const edit = document.createElement("button"); edit.type = "button"; edit.className = "edit-draft-answer";
      edit.dataset.editDraft = index; edit.textContent = "Edit in full text";
      content.append(rendered, edit); row.append(objection, content); container.append(row);
    });
  }

  function syncFields() {
    document.querySelectorAll("[data-point][data-field]").forEach(input => {
      const value = fragment(input.dataset.point)[input.dataset.field];
      if (input.type === "checkbox") input.checked = value;
      else if (input.value !== value) input.value = value;
    });
    document.querySelectorAll("[data-review-field]").forEach(input => {
      const value = state[input.dataset.reviewField];
      if (input.value !== value) input.value = value;
    });
    document.querySelectorAll("[data-topic]").forEach(input => {
      const value = state.reply_topics.find(item => item.id === input.dataset.topic).heading;
      if (input.value !== value) input.value = value;
    });
    document.querySelectorAll("[data-topic-link]").forEach(link => {
      link.textContent = state.reply_topics.find(item => item.id === link.dataset.topicLink).heading || "Edit the numbered topic heading";
    });
    document.querySelectorAll("[data-variant-selector]").forEach(select => {
      const field = select.dataset.variantSelector;
      const options = state.frame_variants[field];
      options.forEach((option, index) => { select.options[index].textContent = option.label; });
      const selected = options.find(option => option.id === select.value) || options[0];
      select.value = selected.id;
      const input = document.querySelector(`[data-variant-field="${field}"]`);
      if (input.value !== selected.text) input.value = selected.text;
    });
  }

  function refresh() {
    Draft.upgrade(state, data.response);
    Draft.ensure(state);
    syncFields();
    for (const item of state.fragments) {
      renderMath(document.querySelector(`[data-preview-for="${item.id}"]`), item.text);
      document.querySelector(`[data-count-for="${item.id}"]`).textContent = `${item.text.length} characters`;
      const check = document.querySelector(`[data-check-for="${item.id}"]`);
      check.textContent = item.needs_check ? "Needs evidence check" : item.include ? "Included in draft" : "Not included";
      check.classList.toggle("needs-check", item.needs_check);
      const shared = state.draft.find(block => block.kind === "answer" && block.points.includes(item.id) && block.points.length > 1);
      const notice = document.querySelector(`[data-shared-for="${item.id}"]`);
      notice.hidden = !shared;
      if (shared) notice.querySelector("span").textContent = "Shared answer with " + shared.points.filter(id => id !== item.id).join(", ") + ". Edits here update that answer everywhere. ";
      const previewLink = document.querySelector(`[data-assembled-link="${item.id}"]`);
      previewLink.hidden = !item.include;
      previewLink.href = item.include ? "#assembled-" + item.id.replaceAll(".", "-") : "#assembled-draft";
    }
    const result = lastResult = assemble();
    const editor = document.getElementById("assembled-text");
    // Do not replace the active field's value: preserve its caret and native undo.
    if (editor.value !== result.body) editor.value = result.body;
    renderMath(document.getElementById("rendered-draft"), result.body);
    for (const id of ["draft-count", "preview-character-count"]) {
      const count = document.getElementById(id);
      count.textContent = limit === null ? `${result.characters} characters · limit not specified` : `${result.characters} / ${limit.toLocaleString("en-US")} characters`;
      count.classList.toggle("over-limit", limit !== null && result.characters > limit);
      count.classList.toggle("near-limit", limit !== null && result.characters >= limit * 0.93 && result.characters <= limit);
    }
    document.getElementById("preview-word-count").textContent = `${result.words} words`;
    for (const topic of state.reply_topics || []) {
      const pending = topic.points.filter(id => fragment(id).needs_check);
      const omitted = topic.points.filter(id => !fragment(id).include);
      const notes = [];
      if (pending.length) notes.push("Needs evidence: " + pending.join(", "));
      if (omitted.length) notes.push("Not included: " + omitted.join(", "));
      document.querySelector(`[data-topic-status="${topic.id}"]`).textContent = notes.join(" · ");
    }
    const issueList = document.getElementById("draft-issues");
    issueList.replaceChildren(...result.issues.map(issue => {
      const item = document.createElement("li"); item.textContent = issue; return item;
    }));
    const coverage = document.getElementById("draft-coverage");
    coverage.replaceChildren();
    coverage.append(`${result.included.length} of ${state.fragments.length} concerns included.`);
    for (const [label, ids] of [[" Not included: ", result.omitted], [" Needs evidence: ", result.pending]]) {
      if (!ids.length) continue;
      coverage.append(label);
      ids.forEach((id, index) => {
        const anchor = document.createElement("a");
        anchor.href = "#response-" + id.replaceAll(".", "-"); anchor.textContent = id;
        coverage.append(anchor, index + 1 < ids.length ? ", " : ".");
      });
    }
    preview();
    document.getElementById("download-markdown").disabled = !result.can_export;
    if (!result.can_export) document.getElementById("export-status").textContent = "Edits are retained. The Markdown file keeps its last valid draft until the issues above are resolved.";
    else if (fileMode || !connected) document.getElementById("export-status").textContent = "Save to repo writes your fragments and rebuttals/" + state.target + ". Download provides another copy. The local launcher enables automatic file saving.";
    else document.getElementById("export-status").textContent = "Automatic Markdown target: rebuttals/" + state.target + ". This is an author draft; review the selected concern coverage before posting.";
    return result;
  }

  function hydrate() {
    refresh();
  }

  function conflict(remote) {
    conflicted = true;
    remoteConflict = remote;
    document.getElementById("save-conflict").hidden = false;
    status.textContent = "Save conflict · browser edits preserved";
  }

  function receipt(result, source) {
    const details = result.written ? "Updated rebuttals/" + state.target + "." : "The Markdown file retains its last valid draft; see the issues below.";
    document.getElementById("repo-save-receipt").textContent = "Saved to " + source + " at " + new Date().toLocaleTimeString() + ": " + sourceFile + " and a previous-version backup. " + details;
  }

  async function save(force = false) {
    if (!ready || fileMode || serverOutdated || saving || fileSaveActive || conflicted || !dirty && !force) return;
    saving = true;
    saveButton.disabled = true;
    const sentVersion = changes;
    status.textContent = "Saving…";
    try {
      const response = await fetch("/api/reviews/" + data.stem, {
        method: "PUT", headers: {"Content-Type": "application/json", "X-Rebuttal-Workspace": "1"},
        body: JSON.stringify({base_revision: baseRevision, review: state}),
      });
      const result = await response.json();
      if (response.status === 409) { conflict(result.review); return; }
      if (!response.ok) throw new Error(result.error || "Save failed");
      baseRevision = state.revision = result.revision;
      connected = true;
      if (changes === sentVersion) dirty = false;
      cache();
      status.textContent = dirty ? "Saving newer edits…" : result.result.written ? "Saved to files · Markdown updated" : "Fragments saved · Markdown needs attention";
      receipt(result.result, "repository files");
      refresh();
    } catch (_) {
      connected = false;
      status.textContent = "Saved in browser · file save unavailable";
      document.getElementById("file-mode-note").hidden = false;
      refresh();
    } finally {
      saving = false;
      saveButton.disabled = !ready;
      if (connected && dirty && !conflicted) timer = window.setTimeout(save, 150);
    }
  }

  async function writeHandle(handle, text) {
    const stream = await handle.createWritable();
    try { await stream.write(text); await stream.close(); }
    catch (error) { try { await stream.abort(); } catch (_) {} throw error; }
  }

  async function readRepository(folder) {
    let handle;
    try { handle = await folder.getFileHandle(sourceFile); }
    catch (_) {
      try {
        folder = await folder.getDirectoryHandle(data.workspaceFolder);
        handle = await folder.getFileHandle(sourceFile);
      } catch (_) { throw new Error("Choose the paper workspace folder, or its parent folder."); }
    }
    const text = await (await handle.getFile()).text();
    const workspace = JSON.parse(text);
    const targets = [...data.expectedTargets].sort();
    if (workspace.version !== 1 || !workspace.reviews ||
        Object.values(workspace.reviews).map(review => review.target).sort().join("|") !== targets.join("|") ||
        !sameShape(workspace.reviews[data.stem]) || !Number.isInteger(workspace.reviews[data.stem].revision)) {
      throw new Error("This folder does not contain the expected response workspace.");
    }
    return {folder, handle, text, workspace};
  }

  async function directSave() {
    if (!ready || saving || fileSaveActive || conflicted) return;
    if (typeof window.showDirectoryPicker !== "function") {
      status.textContent = "Run start_workspace.cmd to save to the repository in this browser.";
      document.getElementById("file-mode-note").hidden = false;
      document.querySelector(".saving-details").open = true;
      return;
    }
    fileSaveActive = true;
    saveButton.disabled = true;
    clearTimeout(timer);
    let committed = false;
    try {
      // Call the picker from the button gesture, before waiting on other work.
      if (!directFolder) directFolder = await window.showDirectoryPicker({id: "rebuttal-" + data.workspaceId.slice(0, 16), mode: "readwrite"});
      status.textContent = "Saving to the selected repository folder…";
      const transaction = async () => {
        const repository = await readRepository(directFolder);
        directFolder = repository.folder;
        const current = repository.workspace.reviews[data.stem];
        if (current.revision !== baseRevision) { conflict(current); return; }
        const sentVersion = changes;
        const snapshot = clone(state);
        const updated = clone(current);
        for (const field of ["intro", "closing"]) updated[field] = snapshot[field];
        if (snapshot.frame_variants) updated.frame_variants = clone(snapshot.frame_variants);
        updated.sections.forEach((section, index) => { section.lead = snapshot.sections[index].lead; });
        updated.fragments.forEach((item, index) => {
          for (const field of ["text", "notes", "joiner", "include", "needs_check", "section"]) item[field] = snapshot.fragments[index][field];
        });
        updated.draft = clone(snapshot.draft);
        if (!updated.reply_topics) updated.reply_topics = clone(snapshot.reply_topics);
        else updated.reply_topics.forEach((topic, index) => { topic.heading = snapshot.reply_topics[index].heading; });
        updated.revision += 1;
        const result = assemble(updated);
        const history = await directFolder.getDirectoryHandle("response_history", {create: true});
        const prior = await history.getDirectoryHandle(data.stem, {create: true});
        const stamp = new Date().toISOString().replace(/[:.]/g, "-");
        const backup = await prior.getFileHandle(String(current.revision).padStart(5, "0") + "-" + stamp + "-browser.json", {create: true});
        await writeHandle(backup, JSON.stringify(current, null, 2) + "\n");
        // Retain the whole previous workspace as well, so other review edits are recoverable.
        const full = await history.getDirectoryHandle("workspace", {create: true});
        await writeHandle(await full.getFileHandle(stamp + "-" + (data.confidential ? "confidential-" : "") + "browser.json", {create: true}), repository.text);
        const stream = await repository.handle.createWritable();
        try {
          if (await (await repository.handle.getFile()).text() !== repository.text) {
            throw new Error("Repository files changed while saving. Your browser edits are retained; click Save to repo again.");
          }
          repository.workspace.reviews[data.stem] = updated;
          await stream.write(JSON.stringify(repository.workspace, null, 2) + "\n");
          await stream.close();
        } catch (error) { try { await stream.abort(); } catch (_) {} throw error; }
        committed = true;
        baseRevision = state.revision = updated.revision;
        if (changes === sentVersion) dirty = false;
        cache();
        result.written = false;
        if (result.can_export) {
          const rebuttals = await directFolder.getDirectoryHandle("rebuttals", {create: true});
          const target = await rebuttals.getFileHandle(current.target, {create: true});
          await writeHandle(target, result.body);
          result.written = true;
        }
        status.textContent = dirty ? "Earlier edits saved · save again for newer edits" : result.written ? "Saved to repo · Markdown updated" : "Fragments saved to repo · Markdown needs attention";
        receipt(result, directFolder.name);
        refresh();
      };
      if (navigator.locks && navigator.locks.request) await navigator.locks.request("rebuttal-save-" + data.workspaceId, transaction);
      else await transaction();
    } catch (error) {
      if (error.name === "AbortError") status.textContent = "Repository save canceled · browser copy retained";
      else status.textContent = (committed ? "Fragments saved to repo; Markdown save failed. " : "Repository save failed. ") + error.message;
      if (!committed) directFolder = null;
    } finally {
      fileSaveActive = false;
      saveButton.disabled = !ready;
      cache();
    }
  }

  saveButton.addEventListener("click", () => {
    clearTimeout(timer);
    if (!ready || conflicted) return;
    if (serverOutdated) { status.textContent = "Restart start_workspace.cmd and reload to save the new headings."; return; }
    if (!fileMode && connected) save(true);
    else directSave();
  });

  function changed() {
    Draft.upgrade(state, data.response);
    Draft.ensure(state);
    changes += 1;
    dirty = true;
    cache();
    document.getElementById("repo-save-receipt").textContent = storageAvailable
      ? "New edits are saved in this browser; the repository copy has not been updated with these edits yet."
      : "Browser storage is unavailable. Use Save to repo or download a fragment backup to retain these edits.";
    refresh();
    if (storageAvailable) status.textContent = serverOutdated ? "Browser copy saved · restart the local server" : fileMode ? "Saved in browser · repo save pending" : "Saved in browser · saving files…";
    clearTimeout(timer);
    timer = window.setTimeout(save, 550);
  }

  document.getElementById("assembled-text").addEventListener("beforeinput", event => {
    draftSelection = {start: event.target.selectionStart, end: event.target.selectionEnd, body: Draft.body(state)};
  });

  document.getElementById("responses").addEventListener("input", event => {
    if (!ready) return;
    const input = event.target;
    if (input.id === "assembled-text") { Draft.edit(state, input.value, draftSelection); draftSelection = null; }
    else if (input.dataset.point && input.dataset.field) {
      if (input.dataset.field === "text") Draft.setAnswer(state, input.dataset.point, input.value);
      else fragment(input.dataset.point)[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
    } else if (input.dataset.reviewField) state[input.dataset.reviewField] = input.value;
    else if (input.dataset.topic) Draft.setHeading(state, input.dataset.topic, input.value);
    else if (input.dataset.variantSelector) { syncFields(); return; }
    else if (input.dataset.variantField) {
      const field = input.dataset.variantField;
      const selected = document.querySelector(`[data-variant-selector="${field}"]`).value;
      state.frame_variants[field].find(option => option.id === selected).text = input.value;
    }
    else return;
    changed();
  });

  document.getElementById("responses").addEventListener("click", event => {
    if (!ready) return;
    const applyVariant = event.target.closest("[data-apply-variant]");
    if (applyVariant) {
      const field = applyVariant.dataset.applyVariant;
      const selected = document.querySelector(`[data-variant-selector="${field}"]`).value;
      try { localStorage.setItem(key + ":before-wording-option", JSON.stringify({base_revision: baseRevision, dirty, review: state})); } catch (_) {}
      state[field] = state.frame_variants[field].find(option => option.id === selected).text;
      changed(); return;
    }
    const separate = event.target.closest("[data-separate-answer]");
    if (separate) { Draft.separate(state, separate.dataset.separateAnswer); changed(); return; }
    const edit = event.target.closest("[data-edit-draft]");
    if (!edit) return;
    const span = Draft.spans(state)[Number(edit.dataset.editDraft)];
    const editor = document.getElementById("assembled-text");
    editor.focus(); editor.setSelectionRange(span.contentStart, span.contentEnd);
    editor.scrollIntoView({block: "center", behavior: "auto"});
  });

  document.getElementById("arrange-feedback-topics").addEventListener("click", () => {
    if (!ready) return;
    try { localStorage.setItem(key + ":before-topic-arrangement", JSON.stringify({base_revision: baseRevision, dirty, review: state})); } catch (_) {}
    Draft.arrange(state); changed();
    document.getElementById("assembled-draft").scrollIntoView({block: "start", behavior: "auto"});
  });

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], {type}));
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = name;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  document.getElementById("download-markdown").addEventListener("click", () => {
    const result = assemble();
    if (result.can_export) download(state.target, result.body, "text/markdown;charset=utf-8");
  });
  document.getElementById("download-fragments").addEventListener("click", () => {
    download(data.stem + "_fragments.json", JSON.stringify({version: 1, stem: data.stem, review: state}, null, 2) + "\n", "application/json");
  });
  document.getElementById("import-fragments").addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 1_000_000) throw new Error("Backup is too large.");
      const backup = JSON.parse(await file.text());
      if (backup.version !== 1 || backup.stem !== data.stem || !sameShape(backup.review)) throw new Error("This backup does not match this review.");
      const revision = state.revision;
      state = clone(backup.review); state.revision = revision;
      hydrate(); changed();
    } catch (error) { status.textContent = error.message; }
    event.target.value = "";
  });
  document.getElementById("load-saved-version").addEventListener("click", () => {
    if (!remoteConflict || !sameShape(remoteConflict)) return;
    // Keep a separate recoverable copy before replacing this browser's edits.
    try { localStorage.setItem(key + ":before-conflict", JSON.stringify({base_revision: baseRevision, dirty, review: state})); } catch (_) {}
    state = clone(remoteConflict); baseRevision = state.revision; dirty = false;
    conflicted = false; remoteConflict = null;
    document.getElementById("save-conflict").hidden = true;
    cache(); hydrate(); status.textContent = "Loaded from files";
  });
  document.getElementById("keep-browser-version").addEventListener("click", () => {
    if (!remoteConflict) return;
    baseRevision = remoteConflict.revision; state.revision = baseRevision;
    conflicted = false; remoteConflict = null;
    document.getElementById("save-conflict").hidden = true;
    changed();
    if (fileMode || !connected) directSave();
  });
  window.addEventListener("beforeunload", event => {
    cache();
    if (dirty && !fileMode || dirty && !storageAvailable) { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("online", () => { if (dirty) save(); });
  document.addEventListener("visibilitychange", () => { if (document.hidden && dirty) save(); });

  async function initialize() {
    // Inputs are read-only until disk or browser drafts have been restored.
    const fields = document.querySelectorAll("#responses textarea:not([readonly]), #responses input, #responses select");
    fields.forEach(input => { input.disabled = true; });
    const stored = readCache();
    if (fileMode) {
      if (stored && (stored.dirty || stored.review.revision >= state.revision)) {
        state = clone(stored.review); baseRevision = stored.base_revision; dirty = stored.dirty;
      }
      document.getElementById("file-mode-note").hidden = false;
      status.textContent = "Browser copy loaded · use Save to repo";
    } else {
      try {
        const response = await fetch("/api/reviews/" + data.stem, {cache: "no-store"});
        if (!response.ok) throw new Error("Workspace unavailable");
        const result = await response.json();
        if (!sameShape(result.review)) throw new Error("Workspace does not match review");
        serverOutdated = result.editor_format !== 3;
        connected = !serverOutdated; state = result.review; baseRevision = state.revision;
        status.textContent = serverOutdated ? "Restart the local server to save the new editor fields" : "Loaded from files · edits save automatically";
        document.getElementById("server-update-note").hidden = !serverOutdated;
        if (stored && stored.dirty) {
          const server = clone(state);
          state = clone(stored.review); baseRevision = stored.base_revision; dirty = true;
          if (baseRevision !== server.revision) conflict(server);
        }
      } catch (_) {
        if (stored) { state = clone(stored.review); baseRevision = stored.base_revision; dirty = stored.dirty; }
        status.textContent = "Browser save only · local file server unavailable";
        document.getElementById("file-mode-note").hidden = false;
      }
    }
    fields.forEach(input => { input.disabled = false; });
    ready = true;
    saveButton.disabled = false;
    hydrate(); cache();
    document.dispatchEvent(new CustomEvent("responses-ready"));
    if (dirty && connected && !conflicted) save();
  }
  initialize();
})();
