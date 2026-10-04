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
