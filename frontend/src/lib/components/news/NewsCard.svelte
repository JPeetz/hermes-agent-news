<script lang="ts">
	import type { NewsItem, Category } from '$lib/types';
	import { CATEGORY_CONFIG } from '$lib/types';
	import { formatRelativeTime } from '$lib/services/dateUtils';
	import { markdownToHtml } from '$lib/services/markdown';
	import { safeHtml } from '$lib/services/safeHtml';
	import { isSafeUrl } from '$lib/services/sanitize';
	import CategoryBadge from './CategoryBadge.svelte';

	export let item: NewsItem;
	export let category: Category;
	export let date: string;
	export let showCategory: boolean = false;
	// The `item-{id}` anchor must exist exactly once per page: it is what
	// scrollToHashTarget() in +page.svelte resolves with getElementById. A preview
	// copy of a card already on the page would shadow the real one, so it opts out.
	export let anchor: boolean = true;
	// The link preview pins its own read/share actions outside the scroll area, so
	// the card's footer would be a second copy the reader has to scroll to reach.
	export let showActions: boolean = true;

	let expanded = false;
	let copied = false;

	function copyShareLink() {
		const url = `${window.location.origin}/?date=${date}&category=${category}#item-${item.id}`;
		navigator.clipboard.writeText(url);
		copied = true;
		setTimeout(() => (copied = false), 2000);
	}

	$: config = CATEGORY_CONFIG[category];
	$: safeUrl = isSafeUrl(item.url) ? item.url : undefined;
	$: hasContent = item.content && item.content.length > 0;
	$: truncatedContent = item.content?.slice(0, 300);
	$: needsTruncation = item.content?.length > 300;
	$: freshness = item.freshness;

	// Use pre-rendered HTML if available, otherwise convert client-side
	$: summaryHtml = item.summary_html || markdownToHtml(item.summary || '');
	$: contentHtml = item.content_html || markdownToHtml(item.content || '');

	// Determine importance tier class
	$: importanceTierClass =
		item.importance_score >= 80
			? 'card-importance-high'
			: item.importance_score >= 60
				? 'card-importance-medium'
				: item.importance_score >= 40
					? 'card-importance-standard'
					: 'card-importance-low';
</script>

<article
	id={anchor ? `item-${item.id}` : undefined}
	class="card {importanceTierClass}"
	style="scroll-margin-top: 5rem;"
>
	<div class="flex items-start justify-between gap-4 mb-3">
		<div class="flex-1 min-w-0">
			{#if showCategory}
				<CategoryBadge {category} class="mb-2" />
			{/if}

			<h3 class="font-semibold text-text-light leading-snug">
				<a
					href={safeUrl}
					target="_blank"
					rel="noopener noreferrer"
					class="hover:text-hermes-gold transition-colors"
				>
					{item.title}
				</a>
			</h3>
		</div>

		<!-- Importance score -->
		<div
			class="flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center text-sm font-bold
			       {item.importance_score >= 80
				? 'bg-hermes-gold/20 text-hermes-gold'
				: item.importance_score >= 60
					? 'bg-agent-cyan/20 text-agent-cyan'
					: 'bg-bg-surface text-text-muted'}"
			title="Importance score: {item.importance_score}"
		>
			{Math.round(item.importance_score)}
		</div>
	</div>

	<!-- Metadata -->
	<div class="flex flex-wrap items-center gap-2 text-sm text-text-muted mb-3">
		<span>{item.source}</span>
		{#if freshness?.label}
			<span
				class="text-[11px] leading-none px-1.5 py-1 rounded border border-teal-border text-text-muted bg-bg-surface"
				title={freshness.reason || freshness.label}
			>
				{freshness.label}
			</span>
		{/if}
		{#if item.author}
			<span>&middot;</span>
			<span>{item.author}</span>
		{/if}
		{#if item.published}
			<span>&middot;</span>
			<span>{formatRelativeTime(item.published)}</span>
		{/if}
	</div>

	<!-- AI Analysis -->
	{#if item.summary}
		<div class="mb-3 pl-3 border-l-2 border-hermes-gold/30">
			<div class="flex items-center gap-1.5 text-xs font-bold text-text-muted mb-1">
				<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
				</svg>
				<span>AI Analysis</span>
			</div>
			<div class="text-text-muted leading-relaxed font-bold prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-a:text-hermes-gold prose-a:no-underline hover:prose-a:underline">
				{@html safeHtml(summaryHtml)}
			</div>
		</div>
	{/if}

	<!-- Content (expandable) -->
	{#if hasContent}
		<div class="text-text-muted mb-3">
			<div
				class="prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-a:text-hermes-gold prose-a:no-underline hover:prose-a:underline"
				class:line-clamp-3={!expanded && needsTruncation}
			>
				{@html safeHtml(contentHtml)}
			</div>

			{#if needsTruncation}
				<button
					on:click={() => (expanded = !expanded)}
					class="text-hermes-gold hover:text-hermes-gold-dark mt-2 font-medium"
				>
					{expanded ? 'Show less' : 'Read more'}
				</button>
			{/if}
		</div>
	{/if}

	<!-- Themes -->
	{#if item.themes && item.themes.length > 0}
		<div class="flex flex-wrap gap-2 mb-3">
			{#each item.themes as theme}
				<span class="text-xs px-2 py-1 rounded-full bg-bg-surface text-text-muted">
					{theme}
				</span>
			{/each}
		</div>
	{/if}

	<!-- Actions -->
	{#if showActions}
		<div
			class="flex items-center justify-between pt-3 border-t border-teal-border"
		>
			<a
				href={safeUrl}
				target="_blank"
				rel="noopener noreferrer"
				class="text-sm font-medium text-hermes-gold hover:text-hermes-gold-dark transition-colors"
			>
				{category === 'research'
					? 'View Research'
					: category === 'reddit'
						? 'View Discussion'
						: 'Read More'} &rarr;
			</a>
			<button
				on:click={copyShareLink}
				class="text-sm font-medium text-hermes-gold hover:text-hermes-gold-dark transition-colors"
			>
				{copied ? 'Copied!' : 'Share'}
			</button>
		</div>
	{/if}
</article>
