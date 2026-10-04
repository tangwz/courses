import test from 'node:test';
import assert from 'node:assert/strict';
import { createRetryableLoader } from '../src/lib/resources';
import { createPagefindLoader, type Pagefind } from '../src/lib/search';

test('concurrent resource loads share failures and retry only once', async () => {
  const attempts: number[] = [];
  const load = createRetryableLoader(async (attempt) => {
    attempts.push(attempt);
    if (!attempt) throw new Error('Offline');
    return 'ready';
  });
  const first = load();
  assert.equal(load(), first);
  await assert.rejects(first, /Offline/);
  const retry = load();
  assert.equal(load(), retry);
  assert.equal(await retry, 'ready');
  assert.equal(load(), retry);
  assert.deepEqual(attempts, [0, 1]);
});

test('Pagefind retries a rejected import with a fresh URL and preserves the site base', async () => {
  const paths: string[] = [];
  const options: unknown[] = [];
  const index = {
    options: async (value: unknown) => {
      options.push(value);
    },
    search: async () => ({ results: [] }),
  } satisfies Pagefind;
  const load = createPagefindLoader('/library/', async (path) => {
    paths.push(path);
    if (paths.length === 1) throw new Error('Offline');
    return index;
  });
  await assert.rejects(load(), /Offline/);
  assert.equal(await load(), index);
  assert.equal(await load(), index);
  assert.deepEqual(paths, [
    '/library/pagefind/pagefind.js',
    '/library/pagefind/pagefind.js?retry=1',
  ]);
  assert.deepEqual(options, [{ baseUrl: '/library/' }]);
});

test('Pagefind initialization failures can recover on the next load', async () => {
  let calls = 0;
  const load = createPagefindLoader('/', async () => ({
    options: async () => {
      if (!calls++) throw new Error('Initialization failed');
    },
    search: async () => ({ results: [] }),
  }));
  await assert.rejects(load(), /Initialization failed/);
  await load();
  assert.equal(calls, 2);
});
