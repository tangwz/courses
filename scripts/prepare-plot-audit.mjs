import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { normalizePlot } from '../src/lib/plots.ts';

const manifest = JSON.parse(
  await readFile('reports/content-manifest.json', 'utf8'),
);
const plots = [];
for (const entry of manifest.entries.filter(
  (entry) => entry.kind === 'lesson',
)) {
  const course = entry.path.split('/')[3];
  for (const reference of entry.plots) {
    const source = JSON.parse(
      await readFile(`src/content/courses/${course}/${reference}`, 'utf8'),
    );
    plots.push({
      id: `${course}/${reference}`,
      definition: normalizePlot(source, false),
    });
  }
}
await mkdir('output/playwright', { recursive: true });
await writeFile('output/playwright/plot-audit.json', JSON.stringify(plots));
await copyFile('output/playwright/plot-audit.json', 'dist/plot-audit.json');
console.log(`Prepared ${plots.length} plots for browser rendering audit`);
