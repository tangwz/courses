import { useEffect, useRef, useState } from 'react';
import { ui } from '../lib/ui';
import { loadPagefind, type SearchResult } from '../lib/search';

export default function SearchPanel({
  course,
  base,
}: {
  course?: string;
  base: string;
}) {
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState(Boolean(course));
  const [results, setResults] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [error, setError] = useState('');
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    function open() {
      dialog.current?.showModal();
      input.current?.focus();
    }
    function key(event: KeyboardEvent) {
      if (
        event.key === '/' &&
        !(
          event.target instanceof HTMLElement &&
          event.target.closest('input,textarea,[contenteditable]')
        )
      ) {
        event.preventDefault();
        open();
      }
    }
    document
      .querySelectorAll('[data-open-search]')
      .forEach((button) => button.addEventListener('click', open));
    document.addEventListener('keydown', key);
    return () => {
      document
        .querySelectorAll('[data-open-search]')
        .forEach((button) => button.removeEventListener('click', open));
      document.removeEventListener('keydown', key);
    };
  }, []);
  useEffect(() => {
    let active = true;
    if (!query.trim()) {
      setResults([]);
      setBusy(false);
      return;
    }
    setBusy(true);
    const timer = setTimeout(async () => {
      try {
        const index = await loadPagefind(base);
        const response = await index.search(query, {
          filters: scope && course ? { course } : undefined,
        });
        const data = await Promise.all(
          response.results.slice(0, 30).map((result) => result.data()),
        );
        if (active) {
          setResults(
            data.map((item) => ({
              url: item.url,
              title: item.meta.title,
              excerpt: item.excerpt.replace(/<[^>]*>/g, ''),
            })),
          );
          setError('');
        }
      } catch {
        try {
          const index: (SearchResult & { course: string })[] = await (
            await fetch(base + 'search-index.json')
          ).json();
          if (active) {
            setFallback(true);
            setResults(
              index
                .filter(
                  (item) =>
                    (!scope || !course || item.course === course) &&
                    (item.title + item.excerpt)
                      .toLowerCase()
                      .includes(query.toLowerCase()),
                )
                .slice(0, 30),
            );
          }
        } catch {
          if (active) setError(ui.noResults);
        }
      } finally {
        if (active) setBusy(false);
      }
    }, 200);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [query, course, scope, base]);
  return (
    <dialog className="search-dialog" ref={dialog}>
      <div className="dialog-heading">
        <h2>{ui.search}</h2>
        <button
          className="icon-button"
          aria-label={ui.close}
          onClick={() => dialog.current?.close()}
        >
          ×
        </button>
      </div>
      <input
        ref={input}
        className="search-input"
        type="search"
        value={query}
        placeholder={ui.searchPlaceholder}
        aria-label={ui.search}
        onChange={(event) => setQuery(event.target.value)}
      />
      {course && (
        <label className="search-scope">
          <input
            type="checkbox"
            checked={scope}
            onChange={(event) => setScope(event.target.checked)}
          />
          {ui.scopeCourse}
        </label>
      )}
      <div className="search-results" aria-live="polite">
        {busy ? (
          <p>{ui.searching}</p>
        ) : !query ? (
          <p>{ui.searchHint}</p>
        ) : !results.length ? (
          <p>{error || ui.noResults}</p>
        ) : (
          results.map((result) => (
            <a key={result.url} href={result.url}>
              <strong>{result.title}</strong>
              <p>{result.excerpt}</p>
            </a>
          ))
        )}
      </div>
      {fallback && <small>{ui.devSearch}</small>}
    </dialog>
  );
}
