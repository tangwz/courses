import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  existsSync,
  rmSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

function fixture(partial: boolean) {
  const root = mkdtempSync(join(tmpdir(), 'course-verification-'));
  const write = (path: string, value: unknown) => {
    const target = join(root, path);
    mkdirSync(join(target, '..'), { recursive: true });
    writeFileSync(
      target,
      typeof value === 'string' ? value : JSON.stringify(value),
    );
  };
  write('reports/content-manifest.json', {
    partial,
    entries: [
      {
        kind: 'course',
        sourceId: 1,
        path: 'src/content/courses/first/index.md',
      },
    ],
  });
  write(
    'src/content/courses/first/index.md',
    '---\nsourceId: 1\ncourse: "first"\ntitle: "First"\n---\n',
  );
  write('.crawl/cache/catalog.json', [{ slug: 'first' }, { slug: 'second' }]);
  for (const course of ['first', 'second'])
    write(`.crawl/cache/courses/${course}.json`, {
      chapters: [],
      learning_outcomes: [],
    });
  write('dist/courses/first/index.html', '<main><h1>First</h1></main>');
  write(
    'dist/index.html',
    '<div class="book-card"></div><div class="book-card"></div>',
  );
  write('dist/pagefind/pagefind.js', '');
  return root;
}

const script = fileURLToPath(
  new URL('../scripts/verify-site.mjs', import.meta.url),
);

test('unscoped verification rejects a partial manifest before checking pages', () => {
  const root = fixture(true);
  try {
    const result = spawnSync(process.execPath, [script], {
      cwd: root,
      encoding: 'utf8',
    });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /Partial content manifests require --course/);
    assert.equal(
      existsSync(join(root, 'reports/site-verification.json')),
      false,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('explicit course verification accepts a partial manifest and labels its report', () => {
  const root = fixture(true);
  try {
    const result = spawnSync(process.execPath, [script, '--course=first'], {
      cwd: root,
      encoding: 'utf8',
    });
    assert.equal(result.status, 0, result.stderr || result.stdout);
    const report = JSON.parse(
      readFileSync(join(root, 'reports/prototype-verification.json'), 'utf8'),
    );
    assert.equal(report.partial, true);
    assert.equal(report.courseScope, 'first');
    assert.equal(report.counts.courses, 1);
    assert.deepEqual(report.errors, []);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('full verification rejects a manifest missing a catalog course', () => {
  const root = fixture(false);
  try {
    const result = spawnSync(process.execPath, [script], {
      cwd: root,
      encoding: 'utf8',
    });
    assert.equal(result.status, 1);
    assert.match(result.stdout, /Coverage mismatch: courses: 1 != 2/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
