import test from 'node:test';
import assert from 'node:assert/strict';
import {
  completionStorageKey,
  readCompletion,
  toggleCompletion,
} from '../src/lib/completion';

function createStorage() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => {
      values.set(key, value);
    },
  };
}

test('sequential writes from stale tabs preserve other lessons and removals', () => {
  const storage = createStorage();
  const staleTab = readCompletion(storage, 'a');
  assert.deepEqual(staleTab, []);
  toggleCompletion(storage, 'a', 1);
  assert.deepEqual(toggleCompletion(storage, 'a', 2), [1, 2]);
  toggleCompletion(storage, 'a', 1);
  assert.deepEqual(toggleCompletion(storage, 'a', 3), [2, 3]);
  assert.deepEqual(readCompletion(storage, 'a'), [2, 3]);
  assert.deepEqual(readCompletion(storage, 'b'), []);
});

test('completion reads deduplicate IDs and reject invalid stored values', () => {
  const storage = createStorage();
  storage.setItem(completionStorageKey('a'), '[1,1,"2",null,-1,2.5,3]');
  assert.deepEqual(readCompletion(storage, 'a'), [1, 3]);
  storage.setItem(completionStorageKey('a'), '{}');
  assert.throws(
    () => toggleCompletion(storage, 'a', 4),
    /Invalid completion data/,
  );
  assert.equal(storage.getItem(completionStorageKey('a')), '{}');
});
