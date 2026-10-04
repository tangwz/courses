import type { APIRoute } from 'astro';
import { getBooks, lessonUrl } from '../lib/catalog';
export const GET: APIRoute = async () => {
  const results = (await getBooks()).flatMap((book) =>
    book.lessons.map((lesson) => ({
      course: lesson.data.course,
      title: lesson.data.title,
      excerpt: lesson.data.description,
      url: lessonUrl(lesson),
    })),
  );
  return new Response(JSON.stringify(results), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
};
