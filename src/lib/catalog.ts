import { getCollection } from 'astro:content';
import type { CollectionEntry } from 'astro:content';

export type Course = CollectionEntry<'courses'>;
export type Chapter = CollectionEntry<'chapters'>;
export type Lesson = CollectionEntry<'lessons'>;
export type Project = CollectionEntry<'projects'>;
export type Book = {
  course: Course;
  chapters: Chapter[];
  lessons: Lesson[];
  project?: Project;
};

export function url(path = '') {
  return (
    import.meta.env.BASE_URL.replace(/\/$/, '') + '/' + path.replace(/^\//, '')
  );
}
export const courseUrl = (course: string) => url(`courses/${course}/`);
export const chapterUrl = (course: string, chapter: string) =>
  url(`courses/${course}/${chapter}/`);
export const lessonUrl = (lesson: Lesson) =>
  chapterUrl(lesson.data.course, lesson.data.chapter) +
  lesson.data.lesson +
  '/';

let books: Promise<Book[]>;
export function getBooks() {
  books ??= Promise.all([
    getCollection('courses'),
    getCollection('chapters'),
    getCollection('lessons'),
    getCollection('projects'),
  ]).then(([courses, chapters, lessons, projects]) =>
    courses
      .sort((a, b) => a.data.order - b.data.order)
      .map((course) => {
        const ownChapters = chapters
          .filter((ch) => ch.data.course === course.data.course)
          .sort((a, b) => a.data.order - b.data.order);
        const rank = new Map(
          ownChapters.map((ch) => [ch.data.chapter, ch.data.order]),
        );
        return {
          course,
          chapters: ownChapters,
          lessons: lessons
            .filter((l) => l.data.course === course.data.course)
            .sort(
              (a, b) =>
                rank.get(a.data.chapter)! - rank.get(b.data.chapter)! ||
                a.data.order - b.data.order,
            ),
          project: projects.find((p) => p.data.course === course.data.course),
        };
      }),
  );
  return books;
}

export const categoryNames: Record<string, string> = {
  Mathematics: '\u6570\u5b66',
  Programming: '\u7f16\u7a0b',
  'Machine Learning': '\u673a\u5668\u5b66\u4e60',
  'Data Science': '\u6570\u636e\u79d1\u5b66',
  'Data Engineering': '\u6570\u636e\u5de5\u7a0b',
  Databases: '\u6570\u636e\u5e93',
  Database: '\u6570\u636e\u5e93',
  'Large Language Models': '\u5927\u578b\u8bed\u8a00\u6a21\u578b',
};
export const levelNames = [
  '',
  '\u5165\u95e8\u7ea7',
  '\u4e13\u4e1a\u7ea7',
  '\u4e13\u5458\u7ea7',
  '\u4e13\u5bb6\u7ea7',
];
