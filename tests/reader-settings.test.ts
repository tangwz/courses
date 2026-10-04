import test from 'node:test';
import assert from 'node:assert/strict';
import {
  defaultReaderSettings,
  normalizeReaderSettings,
  readerSettingLimits,
  readReaderSettings,
} from '../src/lib/reader-settings';

test('invalid stored shapes fall back to reader defaults', () => {
  for (const value of [null, [], 'dark', 123, true])
    assert.deepEqual(normalizeReaderSettings(value), defaultReaderSettings);
});

test('invalid fields fall back independently without discarding valid preferences', () => {
  assert.deepEqual(
    normalizeReaderSettings({
      theme: 'dark',
      fontSize: 20,
      lineHeight: null,
      width: '780',
    }),
    { ...defaultReaderSettings, theme: 'dark', fontSize: 20 },
  );
  assert.equal(normalizeReaderSettings({ theme: 'invalid' }).theme, 'system');
  for (const theme of ['system', 'light', 'dark'])
    assert.equal(normalizeReaderSettings({ theme }).theme, theme);
});

test('numeric settings require finite values within their inclusive UI limits', () => {
  for (const [key, { min, max }] of Object.entries(readerSettingLimits)) {
    const field = key as keyof typeof readerSettingLimits;
    for (const value of [min, max])
      assert.equal(normalizeReaderSettings({ [key]: value })[field], value);
    for (const value of [min - 1, max + 1, NaN, Infinity, null, '18'])
      assert.equal(
        normalizeReaderSettings({ [key]: value })[field],
        defaultReaderSettings[field],
      );
  }
});

test('malformed JSON and unavailable storage do not prevent reader recovery', () => {
  for (const value of [null, '{broken', '{"lineHeight":null}', 'null', '[]'])
    assert.deepEqual(
      readReaderSettings({ getItem: () => value }),
      defaultReaderSettings,
    );
  assert.deepEqual(
    readReaderSettings({
      getItem: () => {
        throw new Error('Storage unavailable');
      },
    }),
    defaultReaderSettings,
  );
});
