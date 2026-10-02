<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let day: Date;
	export let inMonth: boolean;
	export let today: boolean;
	export let selected: boolean;
	export let available: boolean;

	const dispatch = createEventDispatcher();

	$: dayNumber = day.getDate();

	function handleClick() {
		if (available) {
			dispatch('click');
		}
	}
</script>

<button
	on:click={handleClick}
	disabled={!available}
	aria-label={day.toDateString()}
	aria-pressed={selected}
	title={available ? `${day.toDateString()} – has news` : day.toDateString()}
	class="
		relative aspect-square rounded-xl text-sm font-medium transition-all duration-150
		{inMonth
			? available
				? 'text-agent-gray-900 dark:text-agent-gray-100'
				: 'text-agent-gray-400 dark:text-agent-gray-500'
			: 'text-agent-gray-300 dark:text-agent-gray-700'}
		{available
			? 'cursor-pointer hover:bg-hermes-gold/15 hover:scale-105'
			: 'cursor-default'}
		{selected
			? 'bg-hermes-gold text-bg-dark shadow-glow-gold hover:bg-hermes-gold-dark hover:text-bg-dark'
			: ''}
		{today && !selected ? 'ring-2 ring-hermes-gold/70 ring-inset' : ''}
	"
>
	<span class="relative z-10 flex items-center justify-center h-full">
		{dayNumber}
	</span>

	{#if available && !selected}
		<span
			class="absolute bottom-1.5 left-1/2 -translate-x-1/2 h-1.5 w-1.5 rounded-full"
			class:bg-agent-cyan={!today}
			class:bg-hermes-gold={today}
		></span>
	{/if}
</button>