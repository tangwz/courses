import type { APIRoute } from 'astro';
const definitions = import.meta.glob('/src/content/courses/*/plots/*.json', {
  eager: true,
  import: 'default',
});
export function getStaticPaths() {
  return Object.entries(definitions).map(([path, definition]) => {
    const parts = path.split('/');
    return {
      params: { course: parts[4], plot: parts[6].replace(/\.json$/, '') },
      props: { definition },
    };
  });
}
export const GET: APIRoute = ({ props }) =>
  new Response(JSON.stringify(props.definition), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
