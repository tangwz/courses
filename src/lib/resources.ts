export function createRetryableLoader<T>(
  load: (attempt: number) => Promise<T>,
) {
  let cached: Promise<T> | undefined;
  let attempt = 0;
  return () => {
    cached ??= Promise.resolve()
      .then(() => load(attempt++))
      .catch((error) => {
        cached = undefined;
        throw error;
      });
    return cached;
  };
}

export function loadScript<T>(url: string, getValue: () => T | undefined) {
  const existing = getValue();
  if (existing) return Promise.resolve(existing);
  return new Promise<T>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.async = true;
    script.onload = () => {
      script.remove();
      const value = getValue();
      if (value) resolve(value);
      else reject(new Error('Script did not initialize its runtime'));
    };
    script.onerror = () => {
      script.remove();
      reject(new Error('Unable to load script'));
    };
    document.head.append(script);
  });
}
