<script lang="ts">
	import { page } from '$app/stores';
	import { browser } from '$app/environment';
	import { currentDate, navigation, goToPreviousDate, goToNextDate, goToLatestDate } from '$lib/stores/dateStore';
	import { formatDate } from '$lib/services/dateUtils';

	// Optional coverage date prop from parent
	export let coverageDate: string | undefined = undefined;

	$: dateDisplay = $currentDate ? formatDate($currentDate, 'EEEE, MMMM d, yyyy') : '';
	$: coverageDisplay = coverageDate ? formatDate(coverageDate, 'EEEE, MMMM d, yyyy') : '';
	$: isLatest = !$navigation.hasNext;

	// Read current category from URL to preserve it during navigation
	$: currentCategory = browser ? $page.url.searchParams.get('category') : null;
</script>

<div class="flex items-center justify-between gap-4">
	<!-- Previous button -->
	<button
		on:click={() => goToPreviousDate(currentCategory || undefined)}
		disabled={!$navigation.hasPrevious}
		class="p-2 rounded-lg transition-colors
		       {$navigation.hasPrevious
			? 'hover:bg-bg-surface text-text-light'
			: 'text-text-muted cursor-not-allowed'}"
		aria-label="Previous day"
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="w-5 h-5"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
			stroke-width="2"
		>
			<path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7" />
		</svg>
	</button>

	<!-- Date display -->
	<div class="text-center flex-1">
		<div>
			<span class="font-medium text-text-light">
				{dateDisplay}
			</span>
			{#if isLatest}
				<span class="ml-2 text-xs bg-hermes-gold text-bg-dark px-2 py-0.5 rounded-full">
					Latest
				</span>
			{/if}
		</div>
		{#if coverageDisplay}
			<div class="text-sm text-text-muted mt-1">
				Coverage: {coverageDisplay}, 00:00–23:59 ET
			</div>
		{/if}
	</div>

	<!-- Next button -->
	<button
		on:click={() => goToNextDate(currentCategory || undefined)}
		disabled={!$navigation.hasNext}
		class="p-2 rounded-lg transition-colors
		       {$navigation.hasNext
			? 'hover:bg-bg-surface text-text-light'
			: 'text-text-muted cursor-not-allowed'}"
		aria-label="Next day"
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="w-5 h-5"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
			stroke-width="2"
		>
			<path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7" />
		</svg>
	</button>
</div>

{#if !isLatest}
	<div class="text-center mt-2">
		<button
			on:click={() => goToLatestDate(currentCategory || undefined)}
			class="text-sm text-hermes-gold hover:text-hermes-gold-dark transition-colors"
		>
			Jump to latest &rarr;
		</button>
	</div>
{/if}
