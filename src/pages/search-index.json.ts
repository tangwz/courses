import type { APIRoute } from 'astro';
import { getBooks, chapterUrl, courseUrl, lessonUrl } from '../lib/catalog';
export const GET: APIRoute = async () => {
  const results = (await getBooks()).flatMap((book) => [
    ...book.chapters.map((chapter) => ({
      course: chapter.data.course,
      title: chapter.data.title,
      excerpt: chapter.data.description,
      url: chapterUrl(chapter.data.course, chapter.data.chapter),
    })),
    ...(book.project
      ? [
          {
            course: book.project.data.course,
            title: book.project.data.title,
            excerpt: book.project.data.description,
            url: courseUrl(book.project.data.course) + 'project/',
          },
        ]
      : []),
    ...book.lessons.map((lesson) => ({
      course: lesson.data.course,
      title: lesson.data.title,
      excerpt: lesson.data.description,
      url: lessonUrl(lesson),
    })),
  ]);
  return new Response(JSON.stringify(results), {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
};
