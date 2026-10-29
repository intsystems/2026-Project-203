/* Exact draft text and its links to reviewer concerns; no browser storage here. */
(() => {
  "use strict";
  const clone = value => JSON.parse(JSON.stringify(value));
  const answer = (review, id) => review.fragments.find(item => item.id === id);
  const topic = (review, id) => (review.reply_topics || []).find(item => item.id === id);
  const histories = new WeakMap();

  function valid(review) {
    if (review.frame_variants !== undefined) {
      if (!review.frame_variants || Array.isArray(review.frame_variants) ||
          Object.keys(review.frame_variants).some(key => !["intro", "closing"].includes(key))) return false;
      if (!Object.values(review.frame_variants).every(options => {
        const ids = new Set();
        return Array.isArray(options) && options.length >= 1 && options.length <= 12 && options.every(option => {
          if (!option || typeof option.id !== "string" || ids.has(option.id) ||
              typeof option.label !== "string" || option.label.length > 200 ||
              typeof option.text !== "string" || option.text.length > 10000) return false;
          ids.add(option.id); return true;
        });
      })) return false;
    }
    if (review.reply_topics !== undefined) {
      if (!Array.isArray(review.reply_topics) || review.reply_topics.length > 300) return false;
      const ids = new Set(), points = [];
      if (!review.reply_topics.every(item => {
        if (!item || typeof item.id !== "string" || ids.has(item.id) ||
            !["heading", "question", "source"].every(field => typeof item[field] === "string") ||
            item.heading.length > 5000 || item.question.length > 50000 || item.source.length > 100 ||
            !Array.isArray(item.points) || !item.points.length) return false;
        ids.add(item.id); points.push(...item.points); return true;
      })) return false;
      if (review.reply_topics.length && (points.length !== review.fragments.length || new Set(points).size !== points.length ||
          points.some(id => !answer(review, id)))) return false;
    }
    if (!("draft" in review)) return true; // Earlier browser copies and backups.
    if (!Array.isArray(review.draft) || review.draft.length > review.fragments.length + (review.reply_topics || []).length + 3) return false;
    const seen = new Set(), frames = new Set();
    return review.draft.every(block => {
      if (!block || !["intro", "answer", "closing", "text", "heading"].includes(block.kind) ||
          !["before", "after"].every(field => typeof block[field] === "string" &&
            block[field].length <= 50000 && !block[field].trim())) return false;
      if (block.kind === "answer") {
        if (!Array.isArray(block.points) || !block.points.length) return false;
        let text;
        return block.points.every(id => {
          const item = answer(review, id);
          if (!item || seen.has(id) || text !== undefined && text !== item.text) return false;
          seen.add(id); text = item.text; return true;
        });
      }
      if (block.kind === "text") return typeof block.text === "string" && block.text.length <= 50000;
      if (block.kind === "heading") {
        if (!topic(review, block.topic) || frames.has(block.topic)) return false;
        frames.add(block.topic); return true;
      }
      if (frames.has(block.kind)) return false;
      frames.add(block.kind); return true;
    });
  }

  function insert(blocks, index, block) {
    if (index && !blocks[index - 1].after.endsWith("\n\n")) {
      blocks[index - 1].after += blocks[index - 1].after.endsWith("\n") ? "\n" : "\n\n";
    }
    blocks.splice(index, 0, {...block, before: "", after: index < blocks.length ? "\n\n" : "\n"});
  }

  function defaults(review) {
    const selected = review.fragments.filter(item => item.include);
    const result = review.intro.trim() ? [{kind: "intro"}] : [];
    const topics = review.reply_topics || [];
    if (topics.length) {
      topics.forEach(item => {
        const points = item.points.filter(id => answer(review, id).include);
        if (!points.length) return;
        if (item.heading.trim()) result.push({kind: "heading", topic: item.id});
        points.forEach(id => result.push({kind: "answer", points: [id]}));
      });
    } else selected.forEach(item => result.push({kind: "answer", points: [item.id]}));
    if (review.closing.trim()) result.push({kind: "closing"});
    return result.map((block, index) => {
      const following = result[index + 1]?.kind;
      let after = following ? "\n\n" : "\n";
      if (topics.length && following && ["heading", "answer"].includes(block.kind)) {
        after = following === "closing" ? "\n\n" : following === "heading" ? "\n" : " ";
      }
      return {...block, before: "", after};
    });
  }

  function blocks(review) {
    const selected = review.fragments.filter(item => item.include);
    if (!Array.isArray(review.draft)) return defaults(review);
    const result = clone(review.draft).filter(block => {
      if (block.kind === "heading") return topic(review, block.topic).points.some(id => answer(review, id).include);
      if (block.kind !== "answer") return true;
      block.points = block.points.filter(id => answer(review, id).include);
      return block.points.length;
    });
    const mapped = new Set(result.flatMap(block => block.points || []));
    const pointOrder = (review.reply_topics || []).flatMap(item => item.points);
    const order = id => pointOrder.length ? pointOrder.indexOf(id) : review.fragments.findIndex(item => item.id === id);
    selected.filter(item => !mapped.has(item.id)).forEach(item => {
      let index = result.findIndex(block => block.kind === "closing" || block.kind === "answer" &&
        Math.min(...block.points.map(order)) > order(item.id));
      if (index < 0) index = result.length;
      insert(result, index, {kind: "answer", points: [item.id]});
    });
    if (review.intro.trim() && !result.some(block => block.kind === "intro")) insert(result, 0, {kind: "intro"});
    if (review.closing.trim() && !result.some(block => block.kind === "closing")) insert(result, result.length, {kind: "closing"});
    return result;
  }

  function upgrade(review, source) {
    if (!review.reply_topics && source.reply_topics) review.reply_topics = clone(source.reply_topics);
    if (!review.frame_variants && source.frame_variants) review.frame_variants = clone(source.frame_variants);
  }

  function setHeading(review, id, value) {
    ensure(review);
    topic(review, id).heading = value;
    if (!value.trim() || review.draft.some(block => block.kind === "heading" && block.topic === id)) return;
    const index = review.draft.findIndex(block => block.kind === "answer" &&
      block.points.some(point => topic(review, id).points.includes(point)));
    if (index >= 0) review.draft.splice(index, 0, {kind: "heading", topic: id, before: "", after: " "});
  }

  function arrange(review) {
    ensure(review);
    const old = clone(review.draft), remaining = old.filter(block => block.kind === "answer");
    const result = review.intro.trim() ? [{kind: "intro"}] : [];
    for (const item of review.reply_topics || []) {
      const matches = remaining.filter(block => block.points.some(id => item.points.includes(id)));
      if (!matches.length) continue;
      if (item.heading.trim()) result.push({kind: "heading", topic: item.id});
      matches.forEach(block => { result.push(block); remaining.splice(remaining.indexOf(block), 1); });
    }
    result.push(...remaining, ...old.filter(block => block.kind === "text"));
    if (review.closing.trim()) result.push({kind: "closing"});
    review.draft = result.map((block, index) => {
      const following = result[index + 1]?.kind;
      let after = following ? "\n\n" : "\n";
      if (following && ["heading", "answer"].includes(block.kind)) after = following === "closing" ? "\n\n" : following === "heading" ? "\n" : " ";
      return {...block, before: "", after};
    });
  }

  function ensure(review) { review.draft = blocks(review); }
  function text(review, block) {
    if (block.kind === "heading") return topic(review, block.topic).heading;
    return block.kind === "answer" ? answer(review, block.points[0]).text : block.kind === "text" ? block.text : review[block.kind];
  }
  function body(review) { return blocks(review).map(block => block.before + text(review, block) + block.after).join(""); }
  function setAnswer(review, id, value) {
    ensure(review);
    const linked = review.draft.find(block => block.kind === "answer" && block.points.includes(id));
    (linked ? linked.points : [id]).forEach(point => { answer(review, point).text = value; });
  }

  function spans(review) {
    let offset = 0;
    return blocks(review).map(block => {
      const raw = block.before + text(review, block) + block.after;
      const span = {block, start: offset, end: offset + raw.length, contentStart: offset + block.before.length,
        contentEnd: offset + block.before.length + text(review, block).length};
      offset = span.end; return span;
    });
  }

  function edit(review, value, selection) {
    ensure(review);
    const previous = body(review);
    if (value === previous) return;
    // Restore the links too when native textarea undo brings back an earlier text.
    let history = histories.get(review);
    if (!history) { history = new Map(); histories.set(review, history); }
    const included = review.fragments.filter(item => item.include);
    const signature = included.map(item => item.id).join("|");
    history.set(previous, {signature, draft: clone(review.draft), intro: review.intro, closing: review.closing,
      answers: Object.fromEntries(included.map(item => [item.id, item.text])),
      headings: Object.fromEntries((review.reply_topics || []).map(item => [item.id, item.heading]))});
    if (history.size > 50) history.delete(history.keys().next().value);
    const earlier = history.get(value);
    if (earlier && earlier.signature === signature) {
      review.draft = clone(earlier.draft); review.intro = earlier.intro; review.closing = earlier.closing;
      included.forEach(item => { item.text = earlier.answers[item.id]; });
      (review.reply_topics || []).forEach(item => { item.heading = earlier.headings[item.id]; });
      return;
    }
    // A textarea input is one replacement, including paste, deletion and undo.
    let start = 0, end = previous.length, nextEnd = value.length;
    const hintedEnd = selection ? value.length - (previous.length - selection.end) : -1;
    if (selection && selection.body === previous && hintedEnd >= selection.start &&
        value.slice(0, selection.start) === previous.slice(0, selection.start) &&
        value.slice(hintedEnd) === previous.slice(selection.end)) {
      start = selection.start; end = selection.end; nextEnd = hintedEnd;
    } else {
      while (start < end && start < nextEnd && previous[start] === value[start]) start++;
      while (end > start && nextEnd > start && previous[end - 1] === value[nextEnd - 1]) { end--; nextEnd--; }
    }
    const ranges = spans(review);
    if (!ranges.length) {
      review.draft = [{kind: "text", text: value, before: "", after: ""}];
      return;
    }
    let first = ranges.findIndex(span => span.end > start);
    if (first < 0) first = ranges.length - 1;
    let last = end > start ? ranges.findIndex(span => span.end >= end && span.end > span.start) : first;
    if (last < first) last = first;
    const affected = review.draft.slice(first, last + 1);
    const raw = previous.slice(ranges[first].start, start) + value.slice(start, nextEnd) + previous.slice(end, ranges[last].end);
    const points = affected.flatMap(block => block.points || []);
    const kind = points.length ? "answer" : affected.length === 1 ? affected[0].kind : "text";
    // Frame text absorbed into an answer stays there, rather than being duplicated.
    affected.filter(block => ["intro", "closing"].includes(block.kind)).forEach(block => { review[block.kind] = ""; });
    affected.filter(block => block.kind === "heading").forEach(block => { topic(review, block.topic).heading = ""; });
    const before = raw.match(/^\s*/)[0];
    const rest = raw.slice(before.length);
    const after = rest.match(/\s*$/)[0];
    const content = rest.slice(0, rest.length - after.length);
    const replacement = {kind, before, after};
    if (kind === "answer") { replacement.points = points; points.forEach(id => { answer(review, id).text = content; }); }
    else if (kind === "text") replacement.text = content;
    else if (kind === "heading") { replacement.topic = affected[0].topic; topic(review, replacement.topic).heading = content; }
    else review[kind] = content;
    review.draft.splice(first, last - first + 1, replacement);
  }

  function separate(review, id) {
    ensure(review);
    const index = review.draft.findIndex(block => block.kind === "answer" && block.points.includes(id));
    if (index < 0 || review.draft[index].points.length < 2) return;
    review.draft[index].points = review.draft[index].points.filter(point => point !== id);
    insert(review.draft, index + 1, {kind: "answer", points: [id]});
  }

  window.RebuttalDraft = {valid, blocks, defaults, ensure, upgrade, arrange, setHeading, text, body, setAnswer, spans, edit, separate};
})();
