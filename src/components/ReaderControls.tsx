import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ui } from '../lib/ui';
import ReaderIcon from './ReaderIcon';

type Props = {
  course: string;
  lessonId: number;
  total: number;
};
type Settings = {
  theme: 'system' | 'light' | 'dark';
  fontSize: number;
  lineHeight: number;
  width: number;
};
const defaults: Settings = {
  theme: 'system',
  fontSize: 17,
  lineHeight: 1.85,
  width: 780,
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
      setSettings({
        ...defaults,
        ...JSON.parse(localStorage.getItem('course-reader-settings') || '{}'),
      });
      setCompleted(
        JSON.parse(
          localStorage.getItem('course-reader-completed-' + props.course) ||
            '[]',
        ),
      );
    } catch {}
    const openSettings = () => setSettingsOpen(true);
    const toggleTheme = () =>
      setSettings((current) => ({
        ...current,
        theme:
          document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark',
      }));
    const themeButtons = document.querySelectorAll('[data-reader-theme]');
    const settingsButton = document.querySelector('[data-reader-settings]');
    themeButtons.forEach((button) =>
      button.addEventListener('click', toggleTheme),
    );
    settingsButton?.addEventListener('click', openSettings);
    return () => {
      themeButtons.forEach((button) =>
        button.removeEventListener('click', toggleTheme),
      );
      settingsButton?.removeEventListener('click', openSettings);
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
    const next = isComplete
      ? completed.filter((id) => id !== props.lessonId)
      : [...completed, props.lessonId];
    try {
      localStorage.setItem(
        'course-reader-completed-' + props.course,
        JSON.stringify(next),
      );
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
                      theme: event.target.value as Settings['theme'],
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
                  min="14"
                  max="23"
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
                  min="1.5"
                  max="2.2"
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
                  min="620"
                  max="1000"
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
