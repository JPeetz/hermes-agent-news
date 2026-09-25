import adapterStatic from '@sveltejs/adapter-static';
import adapterVercel from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// Vercel sets VERCEL=1 during its builds -> use the Vercel adapter.
// Local / Docker (VPS) builds -> static adapter outputting to ../web (nginx serves it).
const isVercel = process.env.VERCEL === '1';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),

	kit: {
		adapter: isVercel
			? adapterVercel({
					runtime: 'nodejs22.x',
					regions: ['dub1'],
					split: false
				})
			: adapterStatic({
					pages: '../web',
					assets: '../web',
					fallback: 'index.html',
					precompress: false,
					strict: true
				}),
		paths: {
			base: ''
		},
		prerender: {
			handleHttpError: ({ path, referrer, message }) => {
				// Ignore 404s for /data/ paths - these are runtime files, not built
				if (path.startsWith('/data/')) {
					return;
				}
				// Throw for all other errors
				throw new Error(message);
			}
		},
		csp: {
			mode: 'hash',
			directives: {
				'default-src': ['self'],
				'script-src': ['self'],
				'style-src': ['self', 'unsafe-inline'],
				'img-src': ['self', 'data:'],
				'font-src': ['self'],
				'connect-src': ['self'],
				'worker-src': ['self'],
				'object-src': ['none'],
				'base-uri': ['self'],
				'frame-ancestors': ['self']
			}
		}
	}
};

export default config;