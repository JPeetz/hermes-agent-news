import { decodeHTMLStrict } from 'entities';

/** Decode source title entities once, without parsing or rendering any HTML.
 * Shared with the search worker, where DOM-based decoders are unavailable.
 * The returned title must continue to use ordinary Svelte text interpolation.
 */
export function decodeItemTitle<T extends { title: string }>(item: T): T {
	if (!item.title?.includes('&')) return item;
	return { ...item, title: decodeHTMLStrict(item.title) };
}
