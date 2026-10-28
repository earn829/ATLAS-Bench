'use strict';

const $ = (selector, context = document) => context.querySelector(selector);
const $$ = (selector, context = document) => [...context.querySelectorAll(selector)];
const explanations = {
  'hard-objective': ['EVIDENCE / STRUCTURED ATTRIBUTES', 'Check the recorded opening hours against the query timestamp. A closed place—or an unsupported opening-hours claim—cannot satisfy this hard requirement.'],
  'hard-subjective': ['EVIDENCE / PLACE-SPECIFIC REVIEWS', 'Look for review evidence supporting a quiet environment at this exact POI. A high overall rating does not establish quietness. Unsupported hard requirements make a recommendation unacceptable.'],
  'soft-objective': ['EVIDENCE / PRICE ATTRIBUTES', 'Check spending information. A ¥80 option can remain acceptable if every hard constraint is supported and the ¥60 preference gap is explicitly disclosed. Disclosure is not counted as satisfaction.'],
  'soft-subjective': ['EVIDENCE / EXPERIENCE & DISCLOSURE', 'Inspect reviews for the requested atmosphere. If the preference is unmet or cannot be verified, disclose the gap; it need not invalidate a recommendation that satisfies all hard requirements.']
};
$$('[data-constraint]').forEach(button => button.addEventListener('click', () => {
  $$('[data-constraint]').forEach(item => { item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', String(item === button)); });
  const [source, description] = explanations[button.dataset.constraint];
  $('#constraint-source').textContent = source;
  $('#constraint-description').textContent = description;
}));

const metricGroups = {
  answer: { columns: ['csr', 'cr', 'success', 'recs', 'factual'], default: 'csr', help: 'Select a column to sort. Percentages except Recs/q. All values are reported in the paper.' },
  trajectory: { columns: ['intent', 'appropriateness', 'evidence', 'strategy'], default: 'evidence', help: 'Select a column to sort. Higher scores indicate better judged trajectory quality; the reported scale is up to 5.' },
  cost: { columns: ['tokens', 'calls', 'rounds'], default: 'tokens', help: 'Select a column to sort. Average resource use per query; lower cost is not necessarily better answer quality.' }
};
const metrics = {
  csr: { label: 'CSR', precision: 1, description: 'Constraint satisfaction rate credits satisfied constraints only in qualified or acceptable recommendations. Empty answers add nothing to its numerator or denominator.' },
  cr: { label: 'CR', precision: 1, description: 'Constraint recall also credits supported constraints within otherwise unacceptable recommendations. It gives partial credit where CSR does not.' },
  success: { label: 'Success@k', precision: 1, description: 'The share of satisfiable queries with at least one qualified or acceptable recommendation in the returned list. Empty answers count as failures.' },
  recs: { label: 'Recs/q', precision: 2, description: 'The average number of returned recommendations over satisfiable queries, including zero for empty answers. More recommendations do not by themselves imply better quality.' },
  factual: { label: 'Factual', precision: 2, description: 'The percentage of all queries passing both deterministic evidence-grounding checks and the LLM factuality check.' },
  intent: { label: 'Intent', precision: 2, description: 'Judged understanding of the user request, constraints, and spatiotemporal context.' },
  appropriateness: { label: 'Tool use', precision: 2, description: 'Judged appropriateness of tool selection and use in resolving the search request.' },
  evidence: { label: 'Evidence', precision: 2, description: 'Judged sufficiency of the retrieved evidence for the requested constraints and final recommendation.' },
  strategy: { label: 'Strategy', precision: 2, description: 'Judged adaptation of the search strategy to intermediate observations.' },
  tokens: { label: 'Tokens/q', precision: 1, description: 'Mean token consumption per query, displayed in thousands. Compare this resource cost with answer quality and query success.' },
  calls: { label: 'Calls/q', precision: 2, description: 'Mean number of tool invocations per query.' },
  rounds: { label: 'Rounds/q', precision: 2, description: 'Mean number of execution rounds per query.' }
};
let group = 'answer', sortKey = 'csr', ascending = false;
const formatMetric = (value, key) => key === 'tokens' ? `${(value / 1000).toFixed(1)}K` : value.toFixed(metrics[key].precision);
function renderResults() {
  const config = metricGroups[group];
  const rows = [...(window.ATLAS_RESULTS || [])].sort((a, b) => (a[sortKey] - b[sortKey]) * (ascending ? 1 : -1));
  const maximum = Math.max(...rows.map(row => row[sortKey]));
  $('#results-table thead').innerHTML = `<tr><th scope="col">#</th><th scope="col">Model / configuration</th>${config.columns.map(key => `<th scope="col" aria-sort="${sortKey === key ? (ascending ? 'ascending' : 'descending') : 'none'}"><button data-sort="${key}" aria-label="Sort by ${metrics[key].label}">${metrics[key].label}${sortKey === key ? (ascending ? ' ↑' : ' ↓') : ''}</button></th>`).join('')}</tr>`;
  $('#results-table tbody').innerHTML = rows.map((row, i) => `<tr><td>${String(i + 1).padStart(2, '0')}</td><td>${row.name}<span class="model-setting">${row.setting}</span></td>${config.columns.map(key => `<td>${key === sortKey ? `<span class="score-primary" style="--score:${(row[key] / maximum * 100).toFixed(2)}%"><span>${formatMetric(row[key], key)}</span></span>` : formatMetric(row[key], key)}</td>`).join('')}</tr>`).join('');
  $('#table-help').textContent = config.help;
  $('#metric-explainer').innerHTML = `<strong>${metrics[sortKey].label} / </strong>${metrics[sortKey].description}`;
  $$('[data-sort]').forEach(button => button.addEventListener('click', () => {
    if (sortKey === button.dataset.sort) ascending = !ascending;
    else { sortKey = button.dataset.sort; ascending = group === 'cost'; }
    renderResults();
    $(`[data-sort="${sortKey}"]`).focus({ preventScroll: true });
  }));
}
$$('[data-group]').forEach(button => button.addEventListener('click', () => {
  group = button.dataset.group; sortKey = metricGroups[group].default; ascending = group === 'cost';
  $$('[data-group]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  renderResults();
}));
renderResults();
$('#download-results').addEventListener('click', () => {
  const keys = ['name', 'setting', ...Object.keys(metrics)];
  const csv = [keys.join(','), ...window.ATLAS_RESULTS.map(row => keys.map(key => `"${String(row[key]).replaceAll('"', '""')}"`).join(','))].join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url; a.download = 'atlas-bench-paper-results.csv'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

let toastTimeout;
function toast(message) { const box = $('#toast'); box.textContent = message; box.classList.add('visible'); clearTimeout(toastTimeout); toastTimeout = setTimeout(() => box.classList.remove('visible'), 2600); }
$$('.copy-button').forEach(button => button.addEventListener('click', async () => {
  const code = $('pre code', button.closest('.code-block'));
  try {
    if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(code.textContent);
    else {
      const area = document.createElement('textarea'); area.value = code.textContent; area.style.position = 'fixed'; area.style.opacity = '0'; document.body.append(area); area.select();
      const success = document.execCommand('copy'); area.remove(); if (!success) throw new Error('copy unavailable');
    }
    button.textContent = 'Copied ✓'; toast('Command copied to clipboard');
    setTimeout(() => { button.textContent = 'Copy'; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents(code); const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range); toast('Command selected. Press Ctrl+C or ⌘C to copy.');
  }
}));
$$('[data-provider]').forEach(button => button.addEventListener('click', () => {
  $$('[data-provider]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  $$('.provider-panel').forEach(panel => { panel.hidden = panel.id !== `provider-${button.dataset.provider}`; });
}));
$$('pre code').forEach(code => {
  const lines = code.textContent.split('\n');
  code.replaceChildren(...lines.flatMap((line, i) => {
    const node = document.createElement('span'); node.textContent = line; if (line.trim().startsWith('#')) node.className = 'code-comment';
    return i < lines.length - 1 ? [node, document.createTextNode('\n')] : [node];
  }));
});

const dialog = $('#figure-dialog');
$$('[data-image]').forEach(button => button.addEventListener('click', () => {
  $('#dialog-image').src = button.dataset.image; $('#dialog-image').alt = button.dataset.caption;
  $('#dialog-caption').textContent = button.dataset.caption; $('#dialog-pdf').href = button.dataset.image.replace(/\.png$/, '.pdf');
  dialog.showModal();
}));
$('#close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } });
const menu = $('.menu-toggle');
const closeMenu = () => { menu.setAttribute('aria-expanded', 'false'); $('#main-nav').classList.remove('open'); };
menu.addEventListener('click', () => { const open = menu.getAttribute('aria-expanded') !== 'true'; menu.setAttribute('aria-expanded', String(open)); $('#main-nav').classList.toggle('open', open); });
$$('#main-nav a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });

const mascotToggle = $('[data-mascot-toggle]');
const mascotReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
let mascotPaused = false;
const syncMascotMotion = () => {
  const paused = mascotPaused || mascotReducedMotion.matches;
  document.documentElement.classList.toggle('mascot-motion-paused', paused);
  mascotToggle.setAttribute('aria-pressed', String(paused));
  mascotToggle.setAttribute('aria-disabled', String(mascotReducedMotion.matches));
  mascotToggle.setAttribute('aria-label', mascotReducedMotion.matches
    ? 'ATLAS scout: a robot inspecting a map. Animation disabled by your motion preference.'
    : `${paused ? 'Play' : 'Pause'} ATLAS scout animation`);
  mascotToggle.title = mascotReducedMotion.matches ? 'Reduced motion enabled' : `${paused ? 'Play' : 'Pause'} animation`;
};
mascotToggle.addEventListener('click', () => {
  if (mascotReducedMotion.matches) return;
  mascotPaused = !mascotPaused;
  syncMascotMotion();
});
mascotReducedMotion.addEventListener('change', syncMascotMotion);
syncMascotMotion();
const syncMascotVisibility = () => document.documentElement.classList.toggle('mascot-page-hidden', document.hidden);
document.addEventListener('visibilitychange', syncMascotVisibility);
syncMascotVisibility();
if ('IntersectionObserver' in window) {
  const mascotObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => entry.target.classList.toggle('mascot-offscreen', !entry.isIntersecting));
  });
  $$('.mascot').forEach(mascot => mascotObserver.observe(mascot));
}
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) $$('.docs-nav a').forEach(link => link.classList.toggle('active', link.hash === `#${entry.target.id}`)); });
  }, { rootMargin: '-110px 0px -55% 0px', threshold: 0 });
  $$('.doc-step').forEach(section => observer.observe(section));
}
