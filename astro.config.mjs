import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { courseMarkdown } from './src/lib/markdown.mjs';
import { unified } from '@astrojs/markdown-remark';

export default defineConfig({
  output: 'static',
  base: process.env.SITE_BASE || '/',
  trailingSlash: 'always',
  integrations: [react()],
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, courseMarkdown],
      rehypePlugins: [[rehypeKatex, { strict: false, throwOnError: false }]],
      smartypants: false,
    }),
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark' },
      defaultColor: false,
      wrap: false,
    },
  },
  vite: { optimizeDeps: { exclude: ['plotly.js-dist-min'] } },
  build: { inlineStylesheets: 'never' },
});
