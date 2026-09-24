/** @type {import('tailwindcss').Config} */
export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	darkMode: 'class',
	theme: {
		extend: {
			colors: {
				// Agent N's Hermes News Brand Colors
				'hermes-gold': '#F59E0B',
				'hermes-gold-dark': '#D97706',
				'agent-cyan': '#06B6D4',
				'agent-cyan-dark': '#0891B2',
				'bg-dark': '#121212',
				'bg-card': '#1E1E1E',
				'agent-gray': {
					100: '#f5f5f5',
					200: '#e5e5e5',
					300: '#d4d4d4',
					400: '#a3a3a3',
					500: '#737373',
					600: '#525252',
					700: '#404040',
					800: '#262626',
					900: '#171717'
				},
				// Legacy AATF aliases (kept for compatibility during migration)
				'trend-red': '#E63946',
				'trend-gray': {
					100: '#f5f5f5',
					200: '#e5e5e5',
					300: '#d4d4d4',
					400: '#a3a3a3',
					500: '#737373',
					600: '#525252',
					700: '#404040',
					800: '#262626',
					900: '#171717'
				},
				// Category accent colors (simplified for newsletter)
				'category-releases': '#F59E0B',
				'category-merged-prs': '#06B6D4',
				'category-community': '#10b981',
				'category-tips': '#8b5cf6'
			},
			fontFamily: {
				sans: [
					'-apple-system',
					'BlinkMacSystemFont',
					'Segoe UI',
					'Roboto',
					'Oxygen',
					'Ubuntu',
					'Cantarell',
					'sans-serif'
				],
				mono: ['JetBrains Mono', 'Fira Code', 'monospace']
			},
			boxShadow: {
				'card': '0 2px 8px rgba(0, 0, 0, 0.08)',
				'card-hover': '0 4px 12px rgba(0, 0, 0, 0.12)',
				'glow-gold': '0 0 12px rgba(245, 158, 11, 0.3)',
				'glow-cyan': '0 0 12px rgba(6, 182, 212, 0.3)'
			},
			typography: {
				DEFAULT: {
					css: {
						'code::before': { content: 'none' },
						'code::after': { content: 'none' },
						code: {
							fontWeight: '400',
							backgroundColor: 'rgb(0 0 0 / 0.06)',
							padding: '0.1em 0.35em',
							borderRadius: '0.25rem',
							fontSize: '0.9em'
						}
					}
				},
				invert: {
					css: {
						code: {
							backgroundColor: 'rgb(255 255 255 / 0.1)',
							color: 'inherit'
						}
					}
				}
			}
		}
	},
	plugins: [
		require('@tailwindcss/typography')
	]
};