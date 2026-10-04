export type ReaderSettings = {
  theme: 'system' | 'light' | 'dark';
  fontSize: number;
  lineHeight: number;
  width: number;
};

export const defaultReaderSettings: ReaderSettings = {
  theme: 'system',
  fontSize: 17,
  lineHeight: 1.85,
  width: 780,
};

export const readerSettingLimits = {
  fontSize: { min: 14, max: 23 },
  lineHeight: { min: 1.5, max: 2.2 },
  width: { min: 620, max: 1000 },
};

export function normalizeReaderSettings(value: unknown): ReaderSettings {
  const saved =
    value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  const bounded = (key: keyof typeof readerSettingLimits) => {
    const candidate = saved[key];
    const { min, max } = readerSettingLimits[key];
    return typeof candidate === 'number' &&
      Number.isFinite(candidate) &&
      candidate >= min &&
      candidate <= max
      ? candidate
      : defaultReaderSettings[key];
  };
  return {
    theme:
      saved.theme === 'system' ||
      saved.theme === 'light' ||
      saved.theme === 'dark'
        ? saved.theme
        : defaultReaderSettings.theme,
    fontSize: bounded('fontSize'),
    lineHeight: bounded('lineHeight'),
    width: bounded('width'),
  };
}

export function readReaderSettings(storage: Pick<Storage, 'getItem'>) {
  try {
    return normalizeReaderSettings(
      JSON.parse(storage.getItem('course-reader-settings') || 'null'),
    );
  } catch {
    return { ...defaultReaderSettings };
  }
}
