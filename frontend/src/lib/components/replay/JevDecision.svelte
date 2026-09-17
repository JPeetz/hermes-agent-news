<script lang="ts">
	import type { ReplayCall } from '$lib/types/replay';
	export let call: ReplayCall;
	// Transcript supplies only the output that has arrived at the playback cursor.
	export let text = '';
	export let finished = false;
	export let unavailable = false;
	type Article = {
		id: string; title: string; source: string; snippet: string;
		relevance: string | null; probabilities: Record<string, number> | null;
		confidence: number | null; critical_probability: number | null;
		effective_keep: boolean; fallback_reason: string | null;
	};
	type Result = { articles: Article[]; raw_response: unknown; raw_response_redacted?: boolean };
	let view: 'results' | 'raw' = 'results';
	$: if (call.id) view = 'results';
	function parse(value: string): Result | null {
		try {
			const result = JSON.parse(value);
			return result.schema_version === 'jev-relevance-replay/v1' && Array.isArray(result.articles)
				? result : null;
		} catch { return null; }
	}
	function percent(value: number | null | undefined) {
		return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : '—';
	}
	function label(article: Article) {
		return article.fallback_reason ? 'Retained · fallback' : article.effective_keep ? 'Included' : 'Excluded';
	}
	$: result = parse(text);
	$: rows = result?.articles ?? [];
	$: included = rows.filter((a) => a.effective_keep && !a.fallback_reason).length;
	$: excluded = rows.filter((a) => !a.effective_keep).length;
	$: fallback = rows.filter((a) => !!a.fallback_reason).length;
</script>

<section class="jev" aria-label="Jev batch results">
	<div class="batch-head">
		<div>
			<p class="eyebrow">PARALLEL DECISIONS</p>
			<h4>{call.decision_item_count ?? rows.length} articles. One response.</h4>
			<p class="explain">{call.decision_question_count ?? rows.length * 2} questions evaluated together.</p>
		</div>
		{#if result}
			<div class="tabs" role="group" aria-label="Jev result view">
				<button class:chosen={view === 'results'} on:click={() => view = 'results'}>Results</button>
				<button class:chosen={view === 'raw'} on:click={() => view = 'raw'}>Raw JSON</button>
			</div>
		{/if}
	</div>
	{#if !text}
		<p class="empty">{!finished ? 'Waiting for the complete batch response…' : unavailable ? 'Response data was not retained for this call.' : 'This attempt returned no captured response.'}</p>
	{:else if !result}
		<p class="empty">The result could not be displayed. Captured output:</p>
		<pre>{text}</pre>
	{:else if view === 'raw'}
		<p class="explain raw-note">Original API response, including answers that the inclusion rule does not use.</p>
		{#if result.raw_response_redacted}
			<p class="empty">The raw response was withheld because it contained a credential.</p>
		{:else}
			<pre aria-label="Raw Jev response">{JSON.stringify(result.raw_response, null, 2)}</pre>
		{/if}
	{:else}
		<div class="counts" aria-label="Inclusion decisions">
			<span><strong>{included}</strong> included</span>
			<span><strong>{excluded}</strong> excluded</span>
			{#if fallback}<span class="fallback"><strong>{fallback}</strong> retained on fallback</span>{/if}
		</div>
		<div class="articles">
			{#each rows as article, i (`${call.id}-${article.id}-${i}`)}
				<details>
					<summary>
						<span class="number">{i + 1}</span>
						<span class="headline">{article.title}<small>{article.source}</small></span>
						<span class="decision" class:excluded={!article.effective_keep} class:fallback={!!article.fallback_reason}>{label(article)}</span>
					</summary>
					<div class="evidence">
						<p>{article.snippet}</p>
						<dl>
							<div><dt>Native Choice</dt><dd>{article.relevance?.replaceAll('_', ' ') ?? 'Unavailable'}</dd></div>
							<div><dt>Choice confidence</dt><dd>{percent(article.confidence)}</dd></div>
							{#if article.relevance === 'relevant'}
								<div><dt>Important story · P(yes)</dt><dd>{percent(article.critical_probability)}</dd></div>
							{/if}
						</dl>
						{#if article.probabilities}
							<div class="probabilities" aria-label="Choice probabilities">
								{#each ['relevant', 'irrelevant', 'insufficient_evidence'] as option}
									<div><span>{option.replaceAll('_', ' ')}</span><strong>{percent(article.probabilities[option])}</strong></div>
								{/each}
							</div>
						{/if}
						{#if article.fallback_reason}<p class="fallback">Fallback: {article.fallback_reason.replaceAll('_', ' ')}.</p>{/if}
						<small class="identifier">{article.id}</small>
					</div>
				</details>
			{/each}
		</div>
		<p class="explain rule">The native relevance choice controls inclusion. Confidence is shown for inspection. Importance is used only for relevant articles.</p>
	{/if}
</section>

<style>
	.jev { --ink: #134e4a; --soft: #f0fdfa; color: #262626; border: 1px solid #99f6e4; border-radius: 10px; overflow: hidden; }
	:global(.dark) .jev { --ink: #5eead4; --soft: #102824; color: #e5e5e5; border-color: #28524c; }
	.batch-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 18px; background: var(--soft); flex-wrap: wrap; }
	.eyebrow { color: var(--ink); font-size: 9px; font-weight: 700; letter-spacing: .12em; margin: 0 0 5px; }
	h4 { font-size: 17px; font-weight: 650; line-height: 1.3; margin: 0 0 5px; }
	.explain { font-size: 11px; line-height: 1.6; opacity: .72; margin: 0; }
	.tabs { display: flex; gap: 3px; border: 1px solid #0d948844; border-radius: 6px; padding: 3px; }
	.tabs button { font-size: 11px; padding: 6px 9px; border-radius: 4px; }
	.tabs button.chosen { background: #0f766e; color: white; }
	.tabs button:focus-visible, summary:focus-visible { outline: 2px solid #14b8a6; outline-offset: 2px; }
	.counts { display: flex; gap: 18px; padding: 12px 18px; flex-wrap: wrap; border-bottom: 1px solid #8882; font-size: 11px; }
	.counts strong { font-size: 16px; margin-right: 3px; font-variant-numeric: tabular-nums; }
	details + details { border-top: 1px solid #8882; }
	summary { cursor: pointer; display: flex; align-items: center; gap: 10px; padding: 12px 16px; list-style: none; }
	summary::-webkit-details-marker { display: none; }
	summary::after { content: '+'; opacity: .55; }
	details[open] summary::after { content: '−'; }
	.number { width: 20px; flex-shrink: 0; font-size: 10px; opacity: .5; font-variant-numeric: tabular-nums; }
	.headline { flex: 1; min-width: 0; font-size: 12px; line-height: 1.45; font-weight: 550; overflow-wrap: anywhere; }
	.headline small { display: block; font-size: 10px; opacity: .55; margin-top: 3px; font-weight: 400; }
	.decision { font-size: 9px; color: var(--ink); background: var(--soft); border-radius: 4px; padding: 4px 6px; flex-shrink: 0; }
	.decision.excluded { color: #a16207; background: #ca8a0418; }
	.fallback { color: #b45309; }
	:global(.dark) .fallback, :global(.dark) .decision.excluded { color: #fbbf24; }
	.evidence { padding: 0 18px 14px 46px; font-size: 11px; line-height: 1.6; }
	.evidence > p { margin: 0 0 12px; white-space: pre-wrap; overflow-wrap: anywhere; opacity: .85; }
	dl { display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 12px; }
	dt { font-size: 9px; opacity: .6; } dd { font-size: 12px; font-weight: 600; margin: 3px 0 0; }
	.probabilities { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 10px; border: 1px solid #8882; border-radius: 6px; margin: 0 0 10px; }
	.probabilities span { display: block; font-size: 9px; opacity: .65; }
	.probabilities strong { font-size: 12px; font-weight: 550; }
	.identifier { font-family: monospace; font-size: 9px; opacity: .45; }
	.rule, .raw-note { padding: 14px 18px; } .empty { padding: 20px 18px; font-size: 12px; opacity: .7; }
	pre { margin: 0; padding: 16px; max-height: 440px; overflow: auto; font: 11px/1.6 ui-monospace, monospace; background: #8881; }
	@media (max-width: 480px) { summary { gap: 7px; padding: 12px; } .number { display: none; } .evidence { padding-left: 12px; } .decision { font-size: 8px; } .probabilities { gap: 5px; } }
</style>
