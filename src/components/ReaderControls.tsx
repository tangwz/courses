import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ui } from '../lib/ui';
import {
  completionStorageKey,
  readCompletion,
  toggleCompletion,
} from '../lib/completion';
import ReaderIcon from './ReaderIcon';
import {
  defaultReaderSettings as defaults,
  readerSettingLimits,
  readReaderSettings,
  type ReaderSettings,
} from '../lib/reader-settings';

type Props = {
  course: string;
  lessonId: number;
  total: number;
};

export default function ReaderControls(props: Props) {
  const [settings, setSettings] = useState(defaults);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [error, setError] = useState('');
  const [completed, setCompleted] = useState<number[]>([]);
  const [ready, setReady] = useState(false);
  const modalRef = useRef<HTMLDialogElement>(null);
  const isComplete = completed.includes(props.lessonId);

  useEffect(() => {
    setReady(true);
    try {
      setSettings(readReaderSettings(localStorage));
      setCompleted(readCompletion(localStorage, props.course));
    } catch {}
    const synchronizeCompletion = (event: StorageEvent) => {
      if (
        event.storageArea !== localStorage ||
        (event.key !== null && event.key !== completionStorageKey(props.course))
      )
        return;
      try {
        const latest = readCompletion(localStorage, props.course);
        setCompleted(latest);
        document.dispatchEvent(
          new CustomEvent('reader-completion', { detail: latest }),
        );
      } catch {
        setError(ui.storageError);
      }
    };
    window.addEventListener('storage', synchronizeCompletion);
    const openSettings = () => setSettingsOpen(true);
    const toggleTheme = () =>
      setSettings((current) => ({
        ...current,
        theme:
          document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark',
      }));
    const themeButtons = document.querySelectorAll('[data-reader-theme]');
    const settingsButtons = document.querySelectorAll('[data-reader-settings]');
    themeButtons.forEach((button) =>
      button.addEventListener('click', toggleTheme),
    );
    settingsButtons.forEach((button) =>
      button.addEventListener('click', openSettings),
    );
    return () => {
      window.removeEventListener('storage', synchronizeCompletion);
      themeButtons.forEach((button) =>
        button.removeEventListener('click', toggleTheme),
      );
      settingsButtons.forEach((button) =>
        button.removeEventListener('click', openSettings),
      );
    };
  }, [props.course]);
  useEffect(() => {
    if (!ready) return;
    const root = document.documentElement;
    const media = matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      root.dataset.theme =
        settings.theme === 'system'
          ? media.matches
            ? 'dark'
            : 'light'
          : settings.theme;
    };
    apply();
    media.addEventListener('change', apply);
    root.style.setProperty('--reader-font', settings.fontSize + 'px');
    root.style.setProperty('--reader-leading', String(settings.lineHeight));
    root.style.setProperty('--reader-width', settings.width + 'px');
    try {
      localStorage.setItem('course-reader-settings', JSON.stringify(settings));
    } catch {
      setError(ui.storageError);
    }
    return () => media.removeEventListener('change', apply);
  }, [settings, ready]);
  useEffect(() => {
    const dialog = modalRef.current;
    if (settingsOpen) {
      if (!dialog?.open) dialog?.showModal();
    } else dialog?.close();
  }, [settingsOpen, ready]);

  function toggleCompleted() {
    try {
      const next = toggleCompletion(localStorage, props.course, props.lessonId);
      setCompleted(next);
      document.dispatchEvent(
        new CustomEvent('reader-completion', { detail: next }),
      );
    } catch {
      setError(ui.storageError);
    }
  }

  return (
    <>
      {ready &&
        createPortal(
          <div className="reader-progress">
            <div>
              <span>{ui.completed}</span>
              <span>
                {completed.length} / {props.total}
              </span>
            </div>
            <progress
              aria-label={ui.completed}
              max={props.total}
              value={completed.length}
            />
            {error && (
              <p className="reader-error" role="status">
                {error}
              </p>
            )}
          </div>,
          document.querySelector('#reader-course-progress')!,
        )}
      {ready &&
        props.lessonId > 0 &&
        createPortal(
          <div className="reader-actions">
            <button
              className={isComplete ? 'active' : ''}
              aria-pressed={isComplete}
              onClick={toggleCompleted}
            >
              <ReaderIcon name="check" />
              <span>{isComplete ? ui.completed : ui.complete}</span>
            </button>
          </div>,
          document.querySelector('#reader-actions')!,
        )}
      {ready &&
        createPortal(
          <dialog
            ref={modalRef}
            className="reader-dialog"
            aria-labelledby="reader-settings-title"
            onCancel={() => setSettingsOpen(false)}
          >
            <div className="dialog-heading">
              <h2 id="reader-settings-title">{ui.settings}</h2>
              <button
                className="icon-button"
                aria-label={ui.close}
                onClick={() => setSettingsOpen(false)}
              >
                {'\u00d7'}
              </button>
            </div>
            <div className="settings-form">
              <label>
                {ui.theme}
                <select
                  value={settings.theme}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      theme: event.target.value as ReaderSettings['theme'],
                    })
                  }
                >
                  <option value="system">{ui.system}</option>
                  <option value="light">{ui.light}</option>
                  <option value="dark">{ui.dark}</option>
                </select>
              </label>
              <label>
                {ui.fontSize}
                <input
                  aria-label={ui.fontSize}
                  type="range"
                  min={readerSettingLimits.fontSize.min}
                  max={readerSettingLimits.fontSize.max}
                  value={settings.fontSize}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      fontSize: +event.target.value,
                    })
                  }
                />
                <span>{settings.fontSize}px</span>
              </label>
              <label>
                {ui.lineHeight}
                <input
                  aria-label={ui.lineHeight}
                  type="range"
                  min={readerSettingLimits.lineHeight.min}
                  max={readerSettingLimits.lineHeight.max}
                  step="0.05"
                  value={settings.lineHeight}
                  onChange={(event) =>
                    setSettings({
                      ...settings,
                      lineHeight: +event.target.value,
                    })
                  }
                />
                <span>{settings.lineHeight.toFixed(2)}</span>
              </label>
              <label>
                {ui.width}
                <input
                  aria-label={ui.width}
                  type="range"
                  min={readerSettingLimits.width.min}
                  max={readerSettingLimits.width.max}
                  step="20"
                  value={settings.width}
                  onChange={(event) =>
                    setSettings({ ...settings, width: +event.target.value })
                  }
                />
                <span>{settings.width}px</span>
              </label>
              <button onClick={() => setSettings(defaults)}>{ui.reset}</button>
            </div>
          </dialog>,
          document.body,
        )}
    </>
  );
}
