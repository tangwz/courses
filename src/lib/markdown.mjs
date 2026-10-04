import { visit } from 'unist-util-visit';

const aliases = {
  prompt: 'text',
  pseudocode: 'text',
  plotly: 'json',
  pkg: 'text',
  'julia>': 'julia',
  '.dockerignore': 'text',
  txt: 'text',
  batch: 'bat',
  cmd: 'bat',
  'c++': 'cpp',
  graphviz: 'dot',
  cuda: 'cpp',
  mlir: 'text',
  ptx: 'text',
  gitignore: 'text',
  promql: 'text',
};

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

export function courseMarkdown() {
  return (tree, file) => {
    const course = file.data.astro?.frontmatter?.course;
    const base = (process.env.SITE_BASE || '/').replace(/\/$/, '');
    visit(tree, 'code', (node) => {
      if (aliases[node.lang]) node.lang = aliases[node.lang];
    });
    visit(tree, 'image', (node, index, parent) => {
      if (!node.url.startsWith('plots/') || !course || !parent) return;
      const url = `${base}/courses/${course}/${node.url}`;
      const title = escapeHtml(node.alt || 'Interactive chart');
      parent.children[index] = {
        type: 'html',
        value: `<figure class="course-plot" data-plot-url="${escapeHtml(url)}" data-pagefind-ignore><figcaption>${title}</figcaption><div class="plot-canvas" role="img" aria-label="${title}"></div><p class="plot-status">Loading chart...</p><a class="plot-download" href="${escapeHtml(url)}" download>JSON</a></figure>`,
      };
    });
    // Content links are portable when serving beneath a deployment prefix.
    visit(tree, 'link', (node) => {
      if (base && node.url.startsWith('/courses/')) node.url = base + node.url;
    });
  };
}
