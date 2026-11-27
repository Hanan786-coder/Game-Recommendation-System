// Next Game front end: pick games, call the recommender API, render the ranking.

const MAX_PICKS = 10;
const picked = new Map(); // app_id -> game

const $ = (id) => document.getElementById(id);
const els = {
  search: $("search"), suggestions: $("suggestions"), picked: $("picked"), pickedEmpty: $("picked-empty"),
  recommend: $("recommend"), error: $("form-error"), tray: $("tray"), samples: $("samples"),
  playerForm: $("player-form"), playerId: $("player-id"),
  results: $("results"), resultsTitle: $("results-title"), resultsSub: $("results-sub"),
  history: $("history"), historyStrip: $("history-strip"), ranking: $("ranking"),
  drawer: $("drawer"), backdrop: $("drawer-backdrop"), drawerTitle: $("drawer-title"),
  drawerSub: $("drawer-sub"), drawerList: $("drawer-list"), drawerClose: $("drawer-close"),
};

// ---- helpers ------------------------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || "The recommender did not respond. Check that the server is running.");
  return body;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Steam cover, or a lettered tile when the image is missing
function cover(game, className = "cover") {
  const img = el("img", className);
  img.src = game.cover;
  img.alt = "";
  img.loading = "lazy";
  img.addEventListener("error", () => {
    const tile = el("div", `${className} cover-fallback`, game.title);
    img.replaceWith(tile);
  }, { once: true });
  return img;
}

function price(value) {
  return value === 0 ? "Free to play" : `$${value.toFixed(2)}`;
}

function meta(game) {
  const list = el("ul", "pick-meta");
  [game.year, `${game.positive_ratio}% positive`, price(game.price)].forEach((t) => list.append(el("li", "", t)));
  return list;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = !message;
}

function joinTitles(titles) {
  if (titles.length <= 2) return titles.join(" and ");
  return `${titles.slice(0, -1).join(", ")} and ${titles.at(-1)}`;
}

// ---- picking games ------------------------------------------------------------------------

function addGame(game) {
  if (picked.has(game.app_id)) return;
  if (picked.size >= MAX_PICKS) {
    showError(`You can pick up to ${MAX_PICKS} games. Remove one to add another.`);
    return;
  }
  showError("");
  picked.set(game.app_id, game);
  renderPicked();
}

function removeGame(appId) {
  picked.delete(appId);
  renderPicked();
}

function renderPicked() {
  els.picked.replaceChildren(...[...picked.values()].map((game) => {
    const chip = el("li", "chip");
    const remove = el("button", "chip-remove", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${game.title}`);
    remove.addEventListener("click", () => removeGame(game.app_id));
    chip.append(cover(game, ""), el("span", "", game.title), remove);
    return chip;
  }));
  els.pickedEmpty.hidden = picked.size > 0;
  els.recommend.disabled = picked.size === 0;
  els.tray.querySelectorAll(".tray-item").forEach((button) => {
    button.setAttribute("aria-pressed", String(picked.has(Number(button.dataset.appId))));
  });
  document.querySelectorAll(".add-from-drawer").forEach(syncDrawerButton);
}

// ---- search combobox ----------------------------------------------------------------------

let searchTimer;
let options = [];
let activeOption = -1;

function closeSuggestions() {
  els.suggestions.hidden = true;
  els.search.setAttribute("aria-expanded", "false");
  els.search.removeAttribute("aria-activedescendant");
  activeOption = -1;
}

function setActive(index) {
  const items = els.suggestions.querySelectorAll(".option");
  items.forEach((item, i) => item.setAttribute("aria-selected", String(i === index)));
  activeOption = index;
  if (items[index]) {
    els.search.setAttribute("aria-activedescendant", items[index].id);
    items[index].scrollIntoView({ block: "nearest" });
  }
}

function choose(index) {
  const game = options[index];
  if (!game) return;
  addGame(game);
  els.search.value = "";
  closeSuggestions();
  els.search.focus();
}

function renderSuggestions(games, query) {
  options = games;
  const template = $("game-option");
  if (!games.length) {
    els.suggestions.replaceChildren(el("li", "option-empty", `No games match “${query}”.`));
  } else {
    els.suggestions.replaceChildren(...games.map((game, i) => {
      const item = template.content.firstElementChild.cloneNode(true);
      item.id = `option-${i}`;
      item.querySelector("img").replaceWith(cover(game, ""));
      item.querySelector(".option-title").textContent = game.title;
      item.querySelector(".option-year").textContent = picked.has(game.app_id) ? "Added" : game.year;
      item.addEventListener("mousedown", (event) => { event.preventDefault(); choose(i); });
      return item;
    }));
  }
  els.suggestions.hidden = false;
  els.search.setAttribute("aria-expanded", "true");
  activeOption = -1;
}

els.search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const query = els.search.value.trim();
  if (query.length < 2) { closeSuggestions(); return; }
  searchTimer = setTimeout(async () => {
    try {
      renderSuggestions(await api(`/api/search?q=${encodeURIComponent(query)}`), query);
    } catch (error) {
      showError(error.message);
    }
  }, 150);
});

els.search.addEventListener("keydown", (event) => {
  if (els.suggestions.hidden) return;
  if (event.key === "ArrowDown") { event.preventDefault(); setActive(Math.min(activeOption + 1, options.length - 1)); }
  if (event.key === "ArrowUp") { event.preventDefault(); setActive(Math.max(activeOption - 1, 0)); }
  if (event.key === "Enter") { event.preventDefault(); choose(activeOption >= 0 ? activeOption : 0); }
  if (event.key === "Escape") closeSuggestions();
});

els.search.addEventListener("blur", closeSuggestions);

// ---- popular tray -------------------------------------------------------------------------

async function loadTray() {
  try {
    const games = await api("/api/popular");
    els.tray.replaceChildren(...games.map((game) => {
      const button = el("button", "tray-item");
      button.type = "button";
      button.dataset.appId = game.app_id;
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-label", game.title);
      button.title = game.title;
      button.append(cover(game, ""));
      button.addEventListener("click", () => (picked.has(game.app_id) ? removeGame(game.app_id) : addGame(game)));
      return button;
    }));
  } catch (error) {
    showError(error.message);
  }
}

// ---- results ------------------------------------------------------------------------------

function renderPick(game, index) {
  const item = el("li", "pick");
  item.style.setProperty("--i", index);

  const art = el("div", "pick-art");
  const rank = el("span", "pick-rank", String(game.rank));
  rank.setAttribute("aria-hidden", "true");
  art.append(rank, cover(game));

  const body = el("div", "pick-body");
  const title = el("h3", "", game.title);
  body.append(title, meta(game));
  title.setAttribute("aria-label", `Number ${game.rank}: ${game.title}`);
  if (game.because) {
    const because = el("p", "pick-because", "Because you liked ");
    because.append(el("strong", "", game.because));
    body.append(because);
  }
  if (game.tags.length) {
    const tags = el("ul", "tags");
    game.tags.forEach((tag) => tags.append(el("li", "", tag)));
    body.append(tags);
  }
  const more = el("button", "pick-more", "More like this");
  more.type = "button";
  more.addEventListener("click", () => openDrawer(game, more));
  body.append(more);

  item.append(art, body);
  return item;
}

function renderResults(results, heading, subtitle, liked = null) {
  els.resultsTitle.textContent = heading;
  els.resultsSub.textContent = subtitle;

  els.history.hidden = !liked;
  if (liked) {
    els.historyStrip.replaceChildren(...liked.map((game) => {
      const li = el("li");
      li.append(cover(game, ""), el("span", "", game.title));
      return li;
    }));
  }

  els.ranking.classList.remove("is-entering");
  els.ranking.replaceChildren(...results.map(renderPick));
  void els.ranking.offsetWidth; // restart the entrance animation
  els.ranking.classList.add("is-entering");

  els.results.hidden = false;
  els.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  els.resultsTitle.focus({ preventScroll: true });
}

async function withLoading(task) {
  els.results.classList.add("is-loading");
  showError("");
  try {
    await task();
  } catch (error) {
    showError(error.message);
  } finally {
    els.results.classList.remove("is-loading");
  }
}

els.recommend.addEventListener("click", () => withLoading(async () => {
  const games = [...picked.values()];
  const { results } = await api("/api/recommendations", {
    method: "POST", body: JSON.stringify({ app_ids: games.map((g) => g.app_id) }),
  });
  renderResults(results, "Your next ten games", `Based on ${joinTitles(games.map((g) => g.title))}.`);
}));

async function showPlayer(userId) {
  await withLoading(async () => {
    const data = await api(`/api/users/${userId}/recommendations`);
    renderResults(data.results, `Ten picks for player ${data.user_id}`,
      `Based on the ${data.liked.length} games this player recommended on Steam.`, data.liked);
  });
}

els.playerForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = els.playerId.value.trim();
  if (!/^\d+$/.test(value)) { showError("Player IDs are numbers, for example 51580."); return; }
  showPlayer(value);
});

async function loadSamples() {
  try {
    const ids = await api("/api/sample-users");
    ids.forEach((id) => {
      const button = el("button", "sample", String(id));
      button.type = "button";
      button.addEventListener("click", () => { els.playerId.value = id; showPlayer(id); });
      els.samples.append(button);
    });
  } catch {
    els.samples.hidden = true;
  }
}

// ---- mode tabs ----------------------------------------------------------------------------

const tabs = [...document.querySelectorAll(".mode-tab")];

function selectTab(tab) {
  tabs.forEach((t) => {
    const selected = t === tab;
    t.setAttribute("aria-selected", String(selected));
    t.tabIndex = selected ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !selected;
  });
  showError("");
}

tabs.forEach((tab, i) => {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    const next = tabs[(i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    selectTab(next);
    next.focus();
  });
});

// ---- similar games drawer -----------------------------------------------------------------

let drawerOpener = null;

function syncDrawerButton(button) {
  const added = picked.has(Number(button.dataset.appId));
  button.disabled = added;
  button.textContent = added ? "In your picks" : "Add to my picks";
}

async function openDrawer(game, opener) {
  drawerOpener = opener;
  els.drawerTitle.textContent = `More like ${game.title}`;
  els.drawerSub.textContent = "Closest matches by tags and description, or by who played them when a game has no tags.";
  els.drawerList.replaceChildren(el("li", "", "Finding similar games…"));
  els.drawer.hidden = false;
  els.backdrop.hidden = false;
  document.body.style.overflow = "hidden";
  els.drawerClose.focus();

  try {
    const { results } = await api(`/api/games/${game.app_id}/similar`);
    if (!results.length) {
      els.drawerList.replaceChildren(el("li", "", "No close matches for this game yet."));
      return;
    }
    els.drawerList.replaceChildren(...results.map((similar) => {
      const li = el("li");
      const body = el("div");
      const add = el("button", "add-from-drawer");
      add.type = "button";
      add.dataset.appId = similar.app_id;
      add.addEventListener("click", () => addGame(similar));
      syncDrawerButton(add);
      body.append(el("h3", "", similar.title), meta(similar), add);
      li.append(cover(similar), body);
      return li;
    }));
  } catch (error) {
    els.drawerList.replaceChildren(el("li", "form-error", error.message));
  }
}

function closeDrawer() {
  els.drawer.hidden = true;
  els.backdrop.hidden = true;
  document.body.style.overflow = "";
  drawerOpener?.focus();
}

els.drawerClose.addEventListener("click", closeDrawer);
els.backdrop.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.drawer.hidden) closeDrawer();
});

// Keep keyboard focus inside the open drawer
els.drawer.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  const focusable = [...els.drawer.querySelectorAll("button:not(:disabled)")];
  const first = focusable[0];
  const last = focusable.at(-1);
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
});

loadTray();
loadSamples();
renderPicked();
