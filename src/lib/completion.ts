type CompletionStorage = Pick<Storage, 'getItem' | 'setItem'>;

export function completionStorageKey(course: string) {
  return 'course-reader-completed-' + course;
}

export function readCompletion(
  storage: Pick<Storage, 'getItem'>,
  course: string,
): number[] {
  const value: unknown = JSON.parse(
    storage.getItem(completionStorageKey(course)) || '[]',
  );
  if (!Array.isArray(value)) throw new Error('Invalid completion data');
  return [
    ...new Set(
      value.filter((id): id is number => Number.isInteger(id) && id > 0),
    ),
  ];
}

export function toggleCompletion(
  storage: CompletionStorage,
  course: string,
  lessonId: number,
) {
  const latest = readCompletion(storage, course);
  const next = latest.includes(lessonId)
    ? latest.filter((id) => id !== lessonId)
    : [...latest, lessonId];
  storage.setItem(completionStorageKey(course), JSON.stringify(next));
  return next;
}
