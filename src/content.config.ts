import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const common = z.object({
  sourceId: z.number().int(),
  title: z.string(),
  description: z.string().default(''),
  course: z.string(),
  sourceUrl: z.url(),
  order: z.number().int(),
});
const loader = (pattern: string | string[]) =>
  glob({
    pattern,
    base: './src/content/courses',
    generateId: ({ entry }) => entry.replace(/\.md$/, ''),
  });
export const collections = {
  courses: defineCollection({
    loader: loader('*/index.md'),
    schema: common.extend({
      category: z.string(),
      level: z.number().int().min(1).max(4),
      duration: z.number(),
      prerequisites: z.string(),
      color: z.string(),
      chapterCount: z.number().int(),
      lessonCount: z.number().int(),
      outcomes: z.array(
        z.object({ topic: z.string(), description: z.string() }),
      ),
      hasProject: z.boolean(),
    }),
  }),
  chapters: defineCollection({
    loader: loader('*/*/index.md'),
    schema: common.extend({ chapter: z.string(), hasQuiz: z.boolean() }),
  }),
  lessons: defineCollection({
    loader: loader(['*/*/*.md', '!*/*/index.md']),
    schema: common.extend({
      chapter: z.string(),
      lesson: z.string(),
      plots: z.array(z.string()),
      sourceHash: z.string(),
      sourceCorrections: z.array(
        z.object({ kind: z.string(), line: z.number().int() }),
      ),
    }),
  }),
  projects: defineCollection({
    loader: loader('*/project.md'),
    schema: common.extend({ plots: z.array(z.string()) }),
  }),
};
