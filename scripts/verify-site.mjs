import { readFile, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { load } from 'cheerio';
import { createMarkdownProcessor } from '@astrojs/markdown-remark';
import { visit } from 'unist-util-visit';

const manifest = JSON.parse(
  await readFile('reports/content-manifest.json', 'utf8'),
);
const errors = [];
const sourceCorrections = [];
const counts = {
  courses: 0,
  chapters: 0,
  lessons: 0,
  projects: 0,
  plots: 0,
  codeBlocks: 0,
  formulas: 0,
  tables: 0,
  links: 0,
};
const base = (process.env.SITE_BASE || '/').replace(/\/$/, '');
const courseScope = process.argv
  .find((arg) => arg.startsWith('--course='))
  ?.split('=')[1];
if (manifest.partial && !courseScope)
  throw new Error(
    'Partial content manifests require --course=<slug>; regenerate all content before full-site verification.',
  );
const routeSet = new Set();
const decodeMeta = (text) =>
  Object.fromEntries(
    text
      .split('---\n')[1]
      .trim()
      .split('\n')
      .map((line) => {
        const separator = line.indexOf(':');
        return [
          line.slice(0, separator),
          JSON.parse(line.slice(separator + 1)),
        ];
      }),
  );
const key = (value) => value.trim().replace(/\r\n/g, '\n');
const frequency = (values) => {
  const count = new Map();
  for (const value of values)
    count.set(key(value), (count.get(key(value)) || 0) + 1);
  return count;
};
const containsCounts = (expected, actual) =>
  [...expected].every(([value, count]) => (actual.get(value) || 0) >= count);

async function projectPlots(body) {
  const definitions = [];
  const processor = await createMarkdownProcessor({
    syntaxHighlight: false,
    remarkPlugins: [
      () => (tree) => {
        visit(tree, 'code', (node) => {
          if (node.lang === 'plotly') definitions.push(JSON.parse(node.value));
        });
      },
    ],
  });
  await processor.render(body);
  return definitions;
}

async function verifyPlots(plots, metadata, content, $, project = false) {
  const actualPlots = content.find('[data-plot-url]').toArray();
  if (
    plots.length !== metadata.plots.length ||
    plots.length !== actualPlots.length
  )
    errors.push(
      `Plot coverage mismatch: ${metadata.course}:${metadata.sourceId}`,
    );
  for (const [index, reference] of metadata.plots.entries()) {
    const pattern = project
      ? /^plots\/project-\d+-\d+\.json$/
      : /^plots\/\d+-\d+\.json$/;
    if (!pattern.test(reference)) {
      errors.push(`Invalid plot ownership: ${metadata.sourceId}`);
      continue;
    }
    const contentPath = `src/content/courses/${metadata.course}/${reference}`;
    const targetPath = `dist/courses/${metadata.course}/${reference}`;
    try {
      const data = JSON.parse(await readFile(contentPath, 'utf8'));
      const generated = JSON.parse(await readFile(targetPath, 'utf8'));
      if (
        JSON.stringify(data) !== JSON.stringify(plots[index]) ||
        JSON.stringify(generated) !== JSON.stringify(data)
      )
        errors.push(`Plot fidelity mismatch: ${metadata.sourceId}:${index}`);
      if (
        $(actualPlots[index]).attr('data-plot-url') !==
        `${base}/courses/${metadata.course}/${reference}`
      )
        errors.push(`Plot URL mismatch: ${metadata.sourceId}:${index}`);
    } catch {
      errors.push(`Missing plot: ${contentPath}`);
    }
    counts.plots++;
  }
}

const sources = [];
for (const entry of manifest.entries) {
  const metadata = decodeMeta(await readFile(entry.path, 'utf8'));
  const relative =
    entry.path
      .replace('src/content/', '')
      .replace(/\.md$/, '')
      .replace(/\/index$/, '') + '/';
  const route = '/' + relative;
  routeSet.add(base + route);
  sources.push({ entry, metadata, route });
}
for (const { entry, metadata, route } of sources) {
  if (courseScope && metadata.course !== courseScope) continue;
  const destination = join('dist', route.slice(1), 'index.html');
  let html;
  try {
    html = await readFile(destination, 'utf8');
  } catch {
    errors.push(`Missing page: ${route}`);
    continue;
  }
  const $ = load(html);
  if ($('main').length !== 1 || $('h1').first().text() !== metadata.title)
    errors.push(`Page structure mismatch: ${route}`);
  counts[entry.kind + 's']++;
  $('a[href]').each((_, element) => {
    const target = $(element).attr('href');
    if (
      target.startsWith(base + '/courses/') &&
      !target.includes('/plots/') &&
      !routeSet.has(target.split('#')[0])
    )
      errors.push(`Broken internal link: ${route} -> ${target}`);
    counts.links++;
  });
  if (entry.kind === 'course') {
    const expected = JSON.parse(
      await readFile(`.crawl/cache/courses/${metadata.course}.json`, 'utf8'),
    );
    const headings = $('.chapter-list summary strong')
      .toArray()
      .map((el) => $(el).text());
    const expectedChapters = [...expected.chapters].sort(
      (a, b) => a.number - b.number,
    );
    if (
      JSON.stringify(headings) !==
      JSON.stringify(expectedChapters.map((ch) => ch.title))
    )
      errors.push(`Chapter directory order mismatch: ${metadata.course}`);
    const links = $('.chapter-list li a')
      .toArray()
      .map((el) => $(el).attr('href'));
    const expectedLinks = expectedChapters.flatMap((ch) =>
      [...ch.sections]
        .sort((a, b) => a.order - b.order)
        .map(
          (lesson) =>
            `${base}/courses/${metadata.course}/${ch.slug}/${lesson.slug}/`,
        ),
    );
    if (JSON.stringify(links) !== JSON.stringify(expectedLinks))
      errors.push(`Lesson directory order mismatch: ${metadata.course}`);
    if (
      expected.learning_outcomes.some(
        (outcome) =>
          !$('.course-facts').text().includes(outcome.topic) ||
          !$('.course-facts').text().includes(outcome.description),
      )
    )
      errors.push(`Learning outcome mismatch: ${metadata.course}`);
  }
  if (entry.kind === 'project') {
    const source = JSON.parse(
      await readFile(`.crawl/cache/courses/${metadata.course}.json`, 'utf8'),
    );
    await verifyPlots(
      await projectPlots(source.project.content),
      metadata,
      $('#lesson-content'),
      $,
      true,
    );
  }
  if (entry.kind !== 'lesson') continue;
  const sameBook = sources.filter(
    (item) =>
      item.entry.kind === 'lesson' && item.metadata.course === metadata.course,
  );
  const lessonIndex = sameBook.findIndex(
    (item) => item.entry.sourceId === entry.sourceId,
  );
  const nav = $('.lesson-pagination a')
    .toArray()
    .map((el) => $(el).attr('href'));
  const previous = sameBook[lessonIndex - 1];
  const next = sameBook[lessonIndex + 1];
  if (
    nav[0] !==
    (previous ? base + previous.route : `${base}/courses/${metadata.course}/`)
  )
    errors.push(`Previous lesson mismatch: ${entry.sourceId}`);
  if (next && nav[1] !== base + next.route)
    errors.push(`Next lesson mismatch: ${entry.sourceId}`);
  const content = $('#lesson-content');
  if (content.find('.katex-error').length)
    errors.push(`Formula rendering error: ${entry.sourceId}`);
  const source = JSON.parse(
    await readFile(`.crawl/cache/sections/${entry.sourceId}.json`, 'utf8'),
  );
  const raw = load(source.content);
  const expectedCode = frequency(
    raw('pre')
      .toArray()
      .map((el) => raw(el).text())
      .filter((value) => value.trim()),
  );
  const actualCode = frequency(
    content
      .find('pre code')
      .toArray()
      .map((el) => $(el).text()),
  );
  if (!containsCounts(expectedCode, actualCode))
    errors.push(`Code fidelity mismatch: ${entry.sourceId}`);
  counts.codeBlocks += [...expectedCode.values()].reduce((a, b) => a + b, 0);
  const mathKey = (expression) =>
    expression
      .replaceAll('\\text{\\textdollar}', '\\$')
      .replace(/\\vert\s*/g, '|')
      .replace(/\s/g, '');
  const sourceMath = raw('annotation[encoding="application/x-tex"]')
    .toArray()
    .map((el) => raw(el).text());
  const expectedMath = frequency(
    sourceMath
      .filter((expression) => {
        if (
          metadata.sourceCorrections?.length &&
          /^\d[\d,.]*[kKMB]?\s+\|\s*$/.test(expression)
        ) {
          sourceCorrections.push({
            lesson: entry.sourceId,
            kind: 'currency-table-boundary',
            expression,
          });
          return false;
        }
        return true;
      })
      .map(mathKey),
  );
  const actualMath = frequency(
    content
      .find('annotation[encoding="application/x-tex"]')
      .toArray()
      .map((el) => mathKey($(el).text())),
  );
  if (!containsCounts(expectedMath, actualMath))
    errors.push(`Formula fidelity mismatch: ${entry.sourceId}`);
  counts.formulas += [...expectedMath.values()].reduce((a, b) => a + b, 0);
  counts.tables += raw('table').length;
  const tableCells = (parser, tables) =>
    tables.toArray().map((table) => {
      const clone = parser(table).clone();
      clone.find('.katex, annotation').remove();
      return clone
        .find('th,td')
        .toArray()
        .map((cell) =>
          parser(cell)
            .text()
            .replace(/[^\p{L}\p{N}]/gu, ''),
        )
        .join('|');
    });
  const expectedTables = frequency(tableCells(raw, raw('table')));
  const actualTables = frequency(tableCells($, content.find('table')));
  if (!containsCounts(expectedTables, actualTables))
    errors.push(`Table fidelity mismatch: ${entry.sourceId}`);
  const plots = raw('[data-plot-definition]')
    .toArray()
    .map((el) => JSON.parse(raw(el).attr('data-plot-definition')));
  await verifyPlots(plots, metadata, content, $);
}
const catalog = JSON.parse(await readFile('.crawl/cache/catalog.json', 'utf8'));
if (!manifest.partial && !courseScope) {
  const home = load(await readFile('dist/index.html', 'utf8'));
  if (home('.book-card').length !== catalog.length)
    errors.push('Homepage coverage mismatch');
  const expected = {
    courses: catalog.length,
    chapters: 0,
    lessons: 0,
    projects: 0,
  };
  for (const item of catalog) {
    const source = JSON.parse(
      await readFile(`.crawl/cache/courses/${item.slug}.json`, 'utf8'),
    );
    expected.chapters += source.chapters.length;
    expected.lessons += source.chapters.reduce(
      (n, ch) => n + ch.sections.length,
      0,
    );
    if (source.project?.content) expected.projects++;
  }
  for (const key in expected)
    if (expected[key] !== counts[key])
      errors.push(
        `Coverage mismatch: ${key}: ${counts[key]} != ${expected[key]}`,
      );
}
try {
  await stat('dist/pagefind/pagefind.js');
} catch {
  errors.push('Missing search index');
}
if (!counts.courses) errors.push('No courses verified');
const report = {
  verifiedAt: new Date().toISOString(),
  partial: Boolean(manifest.partial || courseScope),
  courseScope,
  counts,
  sourceCorrections,
  errors,
};
await writeFile(
  courseScope
    ? 'reports/prototype-verification.json'
    : 'reports/site-verification.json',
  JSON.stringify(report, null, 2) + '\n',
);
console.log(
  JSON.stringify({ ...report, errors: errors.slice(0, 20) }, null, 2),
);
if (errors.length) process.exitCode = 1;
