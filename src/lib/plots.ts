import { ui } from './ui';
import { createRetryableLoader, loadScript } from './resources';
import type * as Plotly from 'plotly.js';
type PlotDefinition = {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  frames?: Plotly.Frame[];
  config?: Partial<Plotly.Config>;
};

function normalizeTitles(value: unknown): void {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    value.forEach(normalizeTitles);
    return;
  }
  const object = value as Record<string, unknown>;
  if (typeof object.title === 'string') object.title = { text: object.title };
  for (const [key, child] of Object.entries(object)) {
    if (key !== 'meta' && key !== 'customdata') normalizeTitles(child);
  }
}

export function normalizePlot(
  definition: PlotDefinition,
  dark: boolean,
): PlotDefinition {
  const color = dark ? '#c9c5d3' : '#45404f';
  const layout = structuredClone(definition.layout || {});
  const source = definition as PlotDefinition & Record<string, unknown>;
  if (source.layout_yaxis2)
    layout.yaxis2 = structuredClone(
      source.layout_yaxis2,
    ) as Partial<Plotly.LayoutAxis>;
  if (source.shapes)
    layout.shapes = structuredClone(source.shapes) as Partial<Plotly.Shape>[];
  if (source.grid)
    layout.grid = structuredClone(source.grid) as Plotly.Layout['grid'];
  if (typeof source.showlegend === 'boolean')
    layout.showlegend = source.showlegend;
  normalizeTitles(layout);
  delete layout.width;
  layout.height = Math.min(650, Math.max(340, layout.height || 400));
  let data = definition.data.map((trace) => {
    const t = structuredClone(trace) as unknown as {
      type?: string;
      mode?: string;
      orientation?: string;
      z?: unknown[][];
      colorscale?: Plotly.ColorScale;
      showscale?: boolean;
      zmin?: number | number[];
      zmax?: number | number[];
      zauto?: boolean;
      yaxis?: string;
      colorbar?: unknown;
      marker?: { colorbar?: unknown };
    };
    normalizeTitles(t.colorbar);
    normalizeTitles(t.marker?.colorbar);
    if (t.type === 'line' || t.type === 'markers') {
      t.mode ??= t.type === 'line' ? 'lines' : 'markers';
      t.type = 'scatter';
    }
    if (t.type === 'timeline') {
      t.type = 'bar';
      t.orientation = 'h';
    }
    if (t.type === 'image' && typeof t.z?.[0]?.[0] === 'number') {
      t.type = 'heatmap';
      t.colorscale = [
        [0, 'rgb(0, 0, 0)'],
        [1, 'rgb(255, 255, 255)'],
      ];
      t.showscale = false;
      t.zmin = typeof t.zmin === 'number' ? t.zmin : 0;
      t.zmax = typeof t.zmax === 'number' ? t.zmax : 255;
      t.zauto = false;
      const axisKey = (t.yaxis || 'y').replace(/^y/, 'yaxis');
      const axes = layout as Record<string, Partial<Plotly.LayoutAxis>>;
      const axis = (axes[axisKey] ||= {});
      axis.autorange = 'reversed';
    }
    return t as Plotly.Data;
  });
  if (
    definition.data.some((trace) => (trace as unknown as { ph?: string }).ph)
  ) {
    const events = definition.data as unknown as {
      ph: string;
      name: string;
      pid: number;
      tid: number;
      ts: number;
      dur: number;
      args?: { name?: string };
    }[];
    const tracks = new Map(
      events
        .filter((event) => event.ph === 'M')
        .map((event) => [
          `${event.pid}:${event.tid}`,
          event.args?.name || event.name,
        ]),
    );
    data = events
      .filter((event) => event.ph === 'X')
      .map((event) => ({
        type: 'bar',
        orientation: 'h',
        name: event.name,
        x: [event.dur],
        base: [event.ts],
        y: [
          tracks.get(`${event.pid}:${event.tid}`) ||
            `${event.pid}:${event.tid}`,
        ],
        hovertemplate:
          '%{fullData.name}<br>start: %{base}<br>duration: %{x}<extra></extra>',
      }));
    layout.barmode = 'overlay';
    layout.xaxis = { title: { text: 'Timestamp (source units)' } };
  }
  return {
    ...definition,
    data,
    layout: {
      ...layout,
      autosize: true,
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { ...layout.font, color, family: 'system-ui, sans-serif' },
      margin: { l: 60, r: 30, t: 65, b: 60, ...layout.margin },
    },
  };
}
export function initPlots(runtimeUrl: string) {
  const figures = [
    ...document.querySelectorAll<HTMLElement>('[data-plot-url]'),
  ];
  if (!figures.length) return;
  const loadRuntime = createRetryableLoader(() =>
    loadScript(
      runtimeUrl,
      () => (window as Window & { Plotly?: typeof Plotly }).Plotly,
    ),
  );
  const loaded = new Map<HTMLElement, PlotDefinition>();
  const widths = new WeakMap<HTMLElement, number>();
  const resizeObserver = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const figure = entry.target as HTMLElement;
      const width = entry.contentRect.width;
      const previous = widths.get(figure);
      widths.set(figure, width);
      if (
        width <= 0 ||
        (previous !== undefined && Math.abs(width - previous) < 1) ||
        figure.dataset.plotLoaded !== 'true'
      )
        continue;
      const canvas = figure.querySelector<HTMLElement>('.plot-canvas')!;
      loadRuntime()
        .then((plotly) => plotly.Plots.resize(canvas))
        .catch(() => {});
    }
  });
  async function render(figure: HTMLElement) {
    const canvas = figure.querySelector<HTMLElement>('.plot-canvas')!;
    const status = figure.querySelector<HTMLElement>('.plot-status')!;
    status.hidden = false;
    status.textContent = ui.loadingChart;
    try {
      const definition =
        loaded.get(figure) ||
        (await (async () => {
          const response = await fetch(figure.dataset.plotUrl!);
          if (!response.ok) throw new Error('Missing plot');
          return response.json();
        })());
      loaded.set(figure, definition);
      const plotly = await loadRuntime();
      const normalized = normalizePlot(
        definition,
        document.documentElement.dataset.theme === 'dark',
      );
      await plotly.react(canvas, normalized.data, normalized.layout, {
        ...definition.config,
        responsive: true,
        displaylogo: false,
        scrollZoom: false,
      });
      if (definition.frames?.length)
        await plotly.addFrames(canvas, definition.frames);
      status.hidden = true;
      figure.dataset.plotLoaded = 'true';
    } catch (error) {
      status.textContent = ui.chartError + ': ' + String(error);
      if (!status.querySelector('button')) {
        const retry = document.createElement('button');
        retry.textContent = ui.retry;
        retry.addEventListener('click', () => render(figure));
        status.append(retry);
      }
    }
  }
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries)
        if (entry.isIntersecting) {
          observer.unobserve(entry.target);
          render(entry.target as HTMLElement);
        }
    },
    { rootMargin: '400px' },
  );
  figures.forEach((figure) => {
    figure.querySelector<HTMLElement>('.plot-status')!.textContent =
      ui.loadingChart;
    observer.observe(figure);
    resizeObserver.observe(figure);
  });
  new MutationObserver(() =>
    loaded.forEach((_, figure) => render(figure)),
  ).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme'],
  });
}
