import test from 'node:test';
import assert from 'node:assert/strict';
import { searchCourses, type Pagefind } from '../src/lib/search';

const unavailable = async (): Promise<Pagefind> => {
  throw new Error('Offline');
};

test('fallback search respects the course scope and query whitespace', async () => {
  const result = await searchCourses(
    ' TOKEN ',
    { base: '/library/', course: 'a' },
    {
      loadIndex: unavailable,
      fetchIndex: async (path) => {
        assert.equal(path, '/library/search-index.json');
        return Response.json([
          { url: '/a/', course: 'a', title: 'Tokenizer', excerpt: 'Text' },
          { url: '/b/', course: 'b', title: 'Tokenizer', excerpt: 'Text' },
        ]);
      },
    },
  );
  assert.equal(result.fallback, true);
  assert.deepEqual(
    result.results.map((item) => item.url),
    ['/a/'],
  );
});

test('a failed fallback request rejects even when its response contains valid JSON', async () => {
  await assert.rejects(
    searchCourses(
      'token',
      { base: '/' },
      {
        loadIndex: unavailable,
        fetchIndex: async () => Response.json([], { status: 503 }),
      },
    ),
    /Search index is unavailable/,
  );
});

test('full-text success reports recovery and returns plain excerpts', async () => {
  const result = await searchCourses(
    'token',
    { base: '/', course: 'a' },
    {
      loadIndex: async () => ({
        options: async () => {},
        search: async (query, options) => {
          assert.equal(query, 'token');
          assert.deepEqual(options.filters, { course: 'a' });
          return {
            results: [
              {
                data: async () => ({
                  url: '/a/',
                  meta: { title: 'Tokenizer' },
                  excerpt: '<mark>Token</mark> ID',
                }),
              },
            ],
          };
        },
      }),
      fetchIndex: async () => {
        throw new Error('Fallback must not run');
      },
    },
  );
  assert.equal(result.fallback, false);
  assert.deepEqual(result.results, [
    { url: '/a/', title: 'Tokenizer', excerpt: 'Token ID' },
  ]);
});
