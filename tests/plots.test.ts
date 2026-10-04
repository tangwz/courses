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
