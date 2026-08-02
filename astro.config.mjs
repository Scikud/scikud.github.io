import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// The old Jekyll site used `permalink: pretty` with the post's `categories`
// as a path prefix, which produced URLs like /openai/pixylls/jeykll/2019/11/18/…
// Posts now live at /posts/<slug>/; these keep the old links alive.
const legacyPostUrls = {
  '/openai/pixylls/jeykll/2019/11/18/coin-wishing-well': '/posts/coin-wishing-well/',
  '/openai/pixylls/jeykll/2020/10/22/new-blog-who-dis': '/posts/new-blog-who-dis/',
  '/openai/2020/11/05/ay-dios-mio': '/posts/ay-dios-mio/',
  '/openai/2020/11/22/reframe-reparametrize': '/posts/reframe-reparametrize/',
  '/openai/2020/12/04/compute': '/posts/compute/',
  '/openai/2020/12/20/troubles': '/posts/troubles/',
  '/openai/2021/01/15/road-so-far': '/posts/road-so-far/',
  '/openai/2021/01/29/anderson-acceleration': '/posts/anderson-acceleration/',
  '/openai/2021/02/12/ml-thoughts': '/posts/ml-thoughts/',
  '/openai/2021/02/25/feedback-transformer': '/posts/feedback-transformer/',
  '/openai/2021/04/09/wrapping-up': '/posts/wrapping-up/',
  // Pages that no longer exist.
  '/contact': '/',
  '/page2': '/',
  '/page3': '/',
};

export default defineConfig({
  site: 'https://scikud.github.io',
  redirects: legacyPostUrls,
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: { theme: 'github-light' },
  },
});
