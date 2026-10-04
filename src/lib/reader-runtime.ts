import { ui } from './ui';

export function initReader() {
  const sidebar = document.querySelector<HTMLElement>('#reader-sidebar');
  const sidebarScroll = sidebar?.querySelector<HTMLElement>(
    '.reader-sidebar-scroll',
  );
  const toc = document.querySelector<HTMLElement>('#reader-toc');
  const mobile = matchMedia('(max-width: 800px)');
  const compact = matchMedia('(max-width: 1180px)');
  const lessonHeader = document.querySelector<HTMLElement>('.lesson-header');
  const columns = document.querySelector<HTMLElement>('.reader-columns');
  if (lessonHeader && columns) {
    new ResizeObserver(() => {
      columns.style.setProperty(
        '--reader-header-height',
        lessonHeader.offsetHeight + 'px',
      );
    }).observe(lessonHeader);
  }
  let collapsed = false;
  let tocCollapsed = false;
  try {
    collapsed =
      localStorage.getItem('course-reader-sidebar-collapsed') === 'true';
  } catch {}
  function syncPanels() {
    const sidebarVisible = mobile.matches
      ? document.body.classList.contains('sidebar-open')
      : !collapsed;
    const tocVisible = compact.matches
      ? document.body.classList.contains('toc-open')
      : !tocCollapsed;
    document.body.classList.toggle(
      'sidebar-collapsed',
      !mobile.matches && collapsed,
    );
    document.body.classList.toggle(
      'toc-collapsed',
      !compact.matches && tocCollapsed,
    );
    if (sidebar) sidebar.inert = !sidebarVisible;
    if (toc) toc.inert = !tocVisible;
    document
      .querySelector('[data-open-sidebar]')
      ?.setAttribute('aria-expanded', String(sidebarVisible));
    document
      .querySelector('[data-toggle-toc]')
      ?.setAttribute('aria-expanded', String(tocVisible));
  }
  const toggle = (open: boolean) => {
    if (mobile.matches) {
      document.body.classList.toggle('sidebar-open', open);
      if (open) document.body.classList.remove('toc-open');
    } else {
      collapsed = !open;
      try {
        localStorage.setItem(
          'course-reader-sidebar-collapsed',
          String(collapsed),
        );
      } catch {}
    }
    syncPanels();
  };
  document
    .querySelector('[data-open-sidebar]')
    ?.addEventListener('click', () =>
      toggle(
        mobile.matches
          ? !document.body.classList.contains('sidebar-open')
          : collapsed,
      ),
    );
  document
    .querySelector('[data-collapse-sidebar]')
    ?.addEventListener('click', () => {
      toggle(false);
      (document.querySelector('[data-open-sidebar]') as HTMLElement)?.focus();
    });
  document.querySelector('[data-toggle-toc]')?.addEventListener('click', () => {
    if (compact.matches) {
      document.body.classList.toggle('toc-open');
      document.body.classList.remove('sidebar-open');
    } else tocCollapsed = !tocCollapsed;
    syncPanels();
  });
  const closePanels = () => {
    document.body.classList.remove('sidebar-open', 'toc-open');
    syncPanels();
  };
  document
    .querySelector('[data-close-sidebar]')
    ?.addEventListener('click', closePanels);
  mobile.addEventListener('change', closePanels);
  compact.addEventListener('change', closePanels);
  toc
    ?.querySelectorAll('a')
    .forEach((link) => link.addEventListener('click', closePanels));
  syncPanels();
  document.querySelectorAll('[data-reader-settings]').forEach((button) => {
    if (button === sidebar?.querySelector('[data-reader-settings]')) return;
    button.addEventListener('click', () =>
      (
        sidebar?.querySelector('[data-reader-settings]') as HTMLElement
      )?.click(),
    );
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      const control = document.body.classList.contains('sidebar-open')
        ? '[data-open-sidebar]'
        : document.body.classList.contains('toc-open')
          ? '[data-toggle-toc]'
          : null;
      closePanels();
      if (control) document.querySelector<HTMLElement>(control)?.focus();
    }
  });
  document
    .querySelector('[data-back-top]')
    ?.addEventListener('click', () =>
      window.scrollTo({ top: 0, behavior: 'smooth' }),
    );
  const article = document.querySelector<HTMLElement>('#lesson-content');
  const course = article
    ?.getAttribute('data-pagefind-filter')
    ?.replace(/^course:/, '');
  const current = sidebar?.querySelector<HTMLElement>('[aria-current=page]');
  const completionKey = 'course-reader-completed-' + course;
  const applyCompletion = (ids: number[]) =>
    sidebar
      ?.querySelectorAll<HTMLElement>('[data-lesson-id]')
      .forEach((link) =>
        link.classList.toggle(
          'is-complete',
          ids.includes(Number(link.dataset.lessonId)),
        ),
      );
  try {
    applyCompletion(JSON.parse(localStorage.getItem(completionKey) || '[]'));
  } catch {}
  document.addEventListener('reader-completion', (event) =>
    applyCompletion((event as CustomEvent).detail),
  );
  if (current && sidebarScroll)
    sidebarScroll.scrollTop = Math.max(
      0,
      current.offsetTop -
        sidebarScroll.offsetTop -
        sidebarScroll.clientHeight / 2,
    );
  const positionKey = 'course-reader-position-' + current?.dataset.lessonId;
  try {
    if (current && !location.hash) {
      const position = Number(localStorage.getItem(positionKey) || 0);
      if (position) requestAnimationFrame(() => window.scrollTo(0, position));
    }
    if (current && course)
      localStorage.setItem(
        'course-reader-last-' + course,
        JSON.stringify({ url: location.pathname, title: document.title }),
      );
  } catch {}
  let pending = false;
  function updateProgress() {
    pending = false;
    const distance = document.documentElement.scrollHeight - innerHeight;
    const progress =
      distance > 0
        ? Math.min(100, Math.max(0, (scrollY / distance) * 100))
        : 100;
    document
      .querySelector<HTMLElement>('#reading-line')
      ?.style.setProperty('width', progress + '%');
    const percent = document.querySelector('#reading-percent');
    if (percent) percent.textContent = Math.round(progress) + '%';
  }
  document.addEventListener(
    'scroll',
    () => {
      if (!pending) {
        pending = true;
        requestAnimationFrame(updateProgress);
      }
    },
    { passive: true },
  );
  window.addEventListener('resize', updateProgress);
  window.addEventListener('pagehide', () => {
    if (current)
      try {
        localStorage.setItem(positionKey, String(scrollY));
      } catch {}
  });
  updateProgress();
  const headingLinks = new Map(
    [...document.querySelectorAll<HTMLAnchorElement>('.reader-toc a')].map(
      (link) => [decodeURIComponent(link.hash.slice(1)), link],
    ),
  );
  const headings = [
    ...(article?.querySelectorAll<HTMLElement>('h2,h3,h4') || []),
  ];
  const initialHeading = decodeURIComponent(location.hash.slice(1));
  (
    headingLinks.get(initialHeading) || headingLinks.values().next().value
  )?.classList.add('active');
  const observer = new IntersectionObserver(
    (entries) => {
      const first = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
      if (first) {
        headingLinks.forEach((link) => link.classList.remove('active'));
        headingLinks.get(first.target.id)?.classList.add('active');
      }
    },
    { rootMargin: '-70px 0px -65% 0px' },
  );
  headings.forEach((heading) => observer.observe(heading));
  article?.querySelectorAll<HTMLElement>('pre').forEach((pre) => {
    const code = pre.querySelector('code');
    if (!code) return;
    const container = document.createElement('div');
    container.className = 'code-block';
    pre.before(container);
    container.append(pre);
    const toolbar = document.createElement('div');
    toolbar.className = 'code-toolbar';
    const label = document.createElement('span');
    label.textContent =
      code.textContent!.trimEnd().split('\n').length + ' lines';
    const copy = document.createElement('button');
    copy.textContent = ui.copy;
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(code.textContent || '');
        copy.textContent = ui.copied;
        setTimeout(() => {
          copy.textContent = ui.copy;
        }, 1500);
      } catch {
        copy.textContent = 'Copy unavailable';
      }
    });
    toolbar.append(label, copy);
    container.prepend(toolbar);
    if (code.textContent!.split('\n').length > 20) {
      container.classList.add('code-collapsed');
      const expand = document.createElement('button');
      expand.className = 'code-expand';
      expand.textContent = ui.expand;
      expand.addEventListener('click', () => {
        container.classList.toggle('code-collapsed');
        expand.textContent = container.classList.contains('code-collapsed')
          ? ui.expand
          : ui.collapse;
      });
      container.append(expand);
    }
  });
  article?.querySelectorAll('table').forEach((table) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-scroll';
    table.before(wrapper);
    wrapper.append(table);
  });
}
