type IconName =
  | 'book'
  | 'sidebar'
  | 'search'
  | 'outline'
  | 'settings'
  | 'theme'
  | 'arrow'
  | 'external'
  | 'check';

const paths: Record<IconName, string[]> = {
  book: [
    'M12 6.5C9 4.5 5 4.5 3 5v14c3-.5 6-.5 9 1 3-1.5 6-1.5 9-1V5c-2-.5-6-.5-9 1.5Z',
    'M12 6.5V20',
  ],
  sidebar: ['M4 4h16v16H4Z', 'M9 4v16', 'm16 9-3 3 3 3'],
  search: ['M17 10a7 7 0 1 1-14 0 7 7 0 0 1 14 0Z', 'm15 15 6 6'],
  outline: ['M4 5h16v14H4Z', 'M14 5v14', 'M7 9h4', 'M7 13h4'],
  settings: [
    'M4 6h16',
    'M4 12h16',
    'M4 18h16',
    'M8 3v6',
    'M16 9v6',
    'M10 15v6',
  ],
  theme: ['M20 14a8 8 0 0 1-10-10 8 8 0 1 0 10 10Z'],
  arrow: ['M20 12H4', 'm10 6-6 6 6 6'],
  external: ['M14 3h7v7', 'm10 14 11-11', 'M10 3H3v18h18v-7'],
  check: ['m5 12 4 4L19 6'],
};

export default function ReaderIcon({ name }: { name: IconName }) {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {paths[name].map((path, index) => (
        <path key={index} d={path} />
      ))}
    </svg>
  );
}
