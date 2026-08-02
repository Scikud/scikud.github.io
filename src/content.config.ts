import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    /** The aside shown under the title. Yours to be as flippant as you like. */
    summary: z.string().optional(),
    /** What search engines and link previews show. Falls back to `summary`. */
    description: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { posts };
