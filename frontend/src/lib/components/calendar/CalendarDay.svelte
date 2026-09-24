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
	class="
		relative aspect-square p-1 rounded-lg text-sm transition-all
		{inMonth ? 'text-text-light' : 'text-text-muted'}
		{available
			? 'cursor-pointer hover:bg-hermes-gold/10'
			: 'cursor-default'}
		{selected
			? 'bg-hermes-gold text-bg-dark hover:bg-hermes-gold-dark'
			: ''}
		{today && !selected
			? 'ring-2 ring-hermes-gold ring-inset'
			: ''}
	"
>
	<span class="relative z-10">{dayNumber}</span>

	<!-- Data indicator dot -->
	{#if available && !selected}
		<span
			class="absolute bottom-1 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-hermes-gold"
		></span>
	{/if}
</button>
