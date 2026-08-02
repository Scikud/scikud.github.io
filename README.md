# ForceMultiplied

Source for [scikud.github.io](https://scikud.github.io) — an [Astro](https://astro.build) static
site. No client-side JavaScript is shipped.

## Working on it

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in dist/
npm run preview  # serve the built output
```

## Writing a post

Add a markdown file to `src/content/posts/`. The filename is the URL slug, so
`src/content/posts/my-post.md` publishes at `/posts/my-post/`.

```markdown
---
title: "My post"
date: 2026-08-01
summary: "One line shown on the home page."
---

Body text. Standard markdown, plus:

- LaTeX — `$inline$` and `$$display$$`, rendered at build time by KaTeX.
- Footnotes — `text[^1]` with `[^1]: the note` at the bottom.
- Code blocks — fenced with a language, highlighted at build time by Shiki.
- Images — put files in `public/images/` and reference them as `/images/foo.png`.
```

Set `draft: true` in the frontmatter to keep a post out of the build.

## Layout

| Path                     | What it is                                             |
| ------------------------ | ------------------------------------------------------ |
| `src/content/posts/`     | The posts                                              |
| `src/content.config.ts`  | Frontmatter schema (a build fails on bad frontmatter)  |
| `src/consts.ts`          | Site title, description, author; date/reading-time     |
| `src/layouts/Base.astro` | The `<head>`, masthead, footer                         |
| `src/pages/`             | Home page and the post template                        |
| `src/styles/global.css`  | All the styling                                        |
| `astro.config.mjs`       | Markdown plugins and redirects from the old Jekyll URLs |
| `public/`                | Copied to the site root as-is (images, favicons)       |

## Deploys

Pushing to `master` triggers `.github/workflows/deploy.yml`, which builds the
site and publishes it to GitHub Pages. This requires **Settings → Pages →
Source → GitHub Actions** on the repository.
