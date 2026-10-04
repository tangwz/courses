import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizePlot } from '../src/lib/plots';

test('plot adaptation preserves source data and makes titles compatible with Plotly 3', () => {
  const definition = {
    data: [{ type: 'line', x: [1, 2], y: [3, 4] }],
    layout: { title: 'Chart', width: 1200, xaxis: { title: 'X' } },
  };
  const normalized = normalizePlot(definition as never, true);
  assert.equal(normalized.data[0].type, 'scatter');
  assert.deepEqual(normalized.layout?.title, { text: 'Chart' });
  assert.deepEqual(normalized.layout?.xaxis?.title, { text: 'X' });
  assert.equal(normalized.layout?.width, undefined);
  assert.equal(definition.data[0].type, 'line');
  assert.equal(definition.layout.width, 1200);
});
test('source timeline and profiling events render as duration bars', () => {
  const timeline = normalizePlot(
    { data: [{ type: 'timeline', x: [2], y: ['lane'], base: [3] }] } as never,
    false,
  );
  assert.equal(timeline.data[0].type, 'bar');
  const profile = normalizePlot(
    {
      data: [{ ph: 'X', pid: 1, tid: 1, name: 'operation', ts: 100, dur: 50 }],
    } as never,
    false,
  );
  assert.equal(profile.data[0].type, 'bar');
  assert.deepEqual((profile.data[0] as { base: number[] }).base, [100]);
  assert.deepEqual((profile.data[0] as { x: number[] }).x, [50]);
});
test('all subplot axes and nested title containers retain their labels', () => {
  const definition = {
    data: [
      { type: 'heatmap', colorbar: { title: 'Intensity' } },
      {
        type: 'scatter',
        marker: { colorbar: { title: 'Weight' } },
        meta: { title: 'Metadata' },
      },
    ],
    layout: {
      xaxis3: { title: 'Third axis' },
      yaxis9: { title: 'Ninth axis' },
      legend: { title: 'Legend' },
      scene: {
        xaxis: { title: 'Scene X' },
        zaxis: { title: { text: 'Scene Z', font: { size: 18 } } },
      },
      coloraxis: { colorbar: { title: 'Color axis' } },
      updatemenus: [{ buttons: [{ args: [{ title: 'Updated title' }] }] }],
      meta: { title: 'Layout metadata' },
    },
    layout_yaxis2: { title: 'Source extension' },
  };
  const source = structuredClone(definition);
  const normalized = normalizePlot(
    definition as never,
    false,
  ) as unknown as typeof definition;
  assert.deepEqual(normalized.layout.xaxis3.title, { text: 'Third axis' });
  assert.deepEqual(normalized.layout.yaxis9.title, { text: 'Ninth axis' });
  assert.deepEqual(normalized.layout.scene.xaxis.title, { text: 'Scene X' });
  assert.deepEqual(
    normalized.layout.scene.zaxis.title,
    definition.layout.scene.zaxis.title,
  );
  assert.deepEqual(normalized.layout.legend.title, { text: 'Legend' });
  assert.deepEqual(normalized.layout.coloraxis.colorbar.title, {
    text: 'Color axis',
  });
  assert.deepEqual(normalized.layout.updatemenus[0].buttons[0].args[0].title, {
    text: 'Updated title',
  });
  assert.deepEqual(normalized.data[0].colorbar?.title, { text: 'Intensity' });
  assert.deepEqual(normalized.data[1].marker?.colorbar.title, {
    text: 'Weight',
  });
  assert.deepEqual((normalized.layout as Record<string, unknown>).yaxis2, {
    title: { text: 'Source extension' },
  });
  assert.deepEqual(normalized.layout.meta, definition.layout.meta);
  assert.deepEqual(normalized.data[1].meta, definition.data[1].meta);
  assert.deepEqual(definition, source);
});
test('grayscale pixel matrices use heatmaps instead of RGB image traces', () => {
  const definition = {
    data: [
      {
        type: 'image',
        z: [
          [0, 255],
          [128, 64],
        ],
        colorscale: 'gray',
      },
    ],
  };
  const normalized = normalizePlot(definition as never, false);
  assert.equal(normalized.data[0].type, 'heatmap');
  assert.deepEqual((normalized.data[0] as { z: number[][] }).z, [
    [0, 255],
    [128, 64],
  ]);
  assert.equal(definition.data[0].type, 'image');
});
test('grayscale comparisons share an absolute intensity scale without reversing unrelated axes', () => {
  const definition = {
    data: [
      { type: 'image', z: [[50, 200]], yaxis: 'y' },
      { type: 'image', z: [[150, 150]], yaxis: 'y2' },
      { type: 'image', z: [[130, 160]], yaxis: 'y3' },
      { type: 'image', z: [[10, 20]], zmin: 10, zmax: 20, yaxis: 'y5' },
    ],
    layout: { yaxis4: { autorange: true } },
  };
  const source = structuredClone(definition);
  const normalized = normalizePlot(definition as never, false);
  const traces = normalized.data as {
    zmin: number;
    zmax: number;
    zauto: boolean;
  }[];
  assert.deepEqual(
    traces.map(({ zmin, zmax, zauto }) => [zmin, zmax, zauto]),
    [
      [0, 255, false],
      [0, 255, false],
      [0, 255, false],
      [10, 20, false],
    ],
  );
  for (const key of ['yaxis', 'yaxis2', 'yaxis3', 'yaxis5']) {
    assert.equal(
      (normalized.layout as Record<string, { autorange: string }>)[key]
        .autorange,
      'reversed',
    );
  }
  assert.equal(normalized.layout?.yaxis4?.autorange, true);
  assert.deepEqual(definition, source);
});

test('RGB image traces and existing heatmap scales are preserved', () => {
  const definition = {
    data: [
      { type: 'image', z: [[[128, 64, 32]]] },
      { type: 'heatmap', z: [[0.2, 0.8]], zmin: 0, zmax: 1 },
    ],
  };
  assert.deepEqual(
    normalizePlot(definition as never, false).data,
    definition.data,
  );
});
