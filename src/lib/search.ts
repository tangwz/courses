import { createRetryableLoader } from './resources';

export type SearchResult = { url: string; title: string; excerpt: string };
export type Pagefind = {
  options: (options: { baseUrl: string }) => Promise<void>;
  search: (
    query: string,
    options: { filters?: Record<string, string> },
  ) => Promise<{
    results: {
      data: () => Promise<{
        url: string;
        meta: { title: string };
        excerpt: string;
      }>;
    }[];
  }>;
};

export function createPagefindLoader(
  base: string,
  importer: (path: string) => Promise<Pagefind> = (path) =>
    import(/* @vite-ignore */ path),
) {
  return createRetryableLoader(async (attempt) => {
    const path = base.replace(/\/$/, '') + '/pagefind/pagefind.js';
    // Browsers also cache rejected module imports, so retries need a fresh URL.
    const index = await importer(path + (attempt ? `?retry=${attempt}` : ''));
    await index.options({ baseUrl: base });
    return index;
  });
}

const loaders = new Map<string, ReturnType<typeof createPagefindLoader>>();
export function loadPagefind(base: string) {
  let loader = loaders.get(base);
  if (!loader) {
    loader = createPagefindLoader(base);
    loaders.set(base, loader);
  }
  return loader();
}

export async function searchCourses(
  query: string,
  { base, course }: { base: string; course?: string },
  dependencies = {
    loadIndex: loadPagefind,
    fetchIndex: (...args: Parameters<typeof fetch>) => fetch(...args),
  },
): Promise<{ results: SearchResult[]; fallback: boolean }> {
  try {
    const index = await dependencies.loadIndex(base);
    const response = await index.search(query, {
      filters: course ? { course } : undefined,
    });
    const data = await Promise.all(
      response.results.slice(0, 30).map((result) => result.data()),
    );
    return {
      fallback: false,
      results: data.map((item) => ({
        url: item.url,
        title: item.meta.title,
        excerpt: item.excerpt.replace(/<[^>]*>/g, ''),
      })),
    };
  } catch {
    const response = await dependencies.fetchIndex(base + 'search-index.json');
    if (!response.ok) throw new Error('Search index is unavailable');
    const index: (SearchResult & { course: string })[] = await response.json();
    const term = query.trim().toLowerCase();
    return {
      fallback: true,
      results: index
        .filter(
          (item) =>
            (!course || item.course === course) &&
            (item.title + item.excerpt).toLowerCase().includes(term),
        )
        .slice(0, 30),
    };
  }
}
