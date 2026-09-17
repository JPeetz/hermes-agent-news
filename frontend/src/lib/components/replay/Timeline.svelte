<script lang="ts">
	import type { ReplayCall, ReplayIndex, ReplayRole } from '$lib/types/replay';
	import {
		agentColor,
		providerColor,
		formatClock,
		formatDuration,
		formatTokens,
		isImageCall,
		ROLE_LABELS
	} from '$lib/services/replayViz';

	export let index: ReplayIndex;
	export let t: number;
	export let selectedCallId: string | null = null;
	export let onSelectCall: (callId: string) => void = () => {};
	export let onSeek: (ms: number) => void = () => {};

	// Two colour systems live here, separated by *region* rather than by a toggle:
	//   · label gutter — agent identity, matching the Newsroom's cast colours
	//   · plot area    — the provider that served each request
	// They never touch, so neither needs a caption to stay unambiguous. An earlier
	// version let the reader switch the bars between provider and effort, which meant
	// the same swatches silently changed meaning.
	//
	// Effort needs no encoding of its own: it is a property of the role, and the
	// taxonomy gives each role its own lane, so a lane's effort is constant and can
	// simply be written next to its name.

	// Offline-reconstructed runs carry start/end only: `wait_ms` is 0 and
	// `first_token_ms` is null on every call. Segmenting a bar into wait / TTFT /
	// streaming would then render one flat slab of "time to first token", which reads
	// as a bug. Fall back to undivided bars and say so instead of faking the split.
	$: measured = index.run.timings_measured !== false;

	// The key must only name segments that are actually on screen. Queue wait is
	// drawn from `queued_ms → start_ms`, and in practice this pipeline never queues:
	// the LLM semaphore is per route at 8, but ANALYZER_MAX_CONCURRENT_BATCHES caps
	// the fan-out upstream, so peak in-flight per route runs 4-5. Across published
	// days the wait is 0 on every call bar a handful of 1ms scheduling jitters.
	//
	// So the test matches the bar's own render threshold rather than `> 0`. A 1ms
	// wait is ~0.00008% of a 20-minute run: the segment is never painted, and the
	// old `> 0` test still listed it — describing a stripe that is not on screen.
	$: hasQueueWait =
		measured && (index.calls ?? []).some((c) => pct(c.start_ms - c.queued_ms) > 0.05);

	// Row height also has to clear the two-line gutter label (name + effort tier).
	const LANE_H = 22;
	const LANE_GAP = 2;

	$: duration = Math.max(1, index.duration_ms || 1);
	$: agentById = new Map((index.agents ?? []).map((a) => [a.id, a]));

	/** Greedy row packing: a call goes in the first row whose last bar already ended. */
	function packRows(calls: ReplayCall[]): ReplayCall[][] {
		const sorted = [...calls].sort((a, b) => a.queued_ms - b.queued_ms);
		const rows: ReplayCall[][] = [];
		for (const call of sorted) {
			let placed = false;
			for (const row of rows) {
				if (row[row.length - 1].end_ms <= call.queued_ms) {
					row.push(call);
					placed = true;
					break;
				}
			}
			if (!placed) rows.push([call]);
		}
		return rows;
	}

	// Swimlanes: one per agent *per role*, in stage order.
	//
	// The Newsroom deliberately keeps a category's reader and ranker as one station —
	// they are one job, and splitting them there made a category read as two peers.
	// Here the opposite is true. Effort is a property of the role, so an agent that
	// mixes roles mixes tiers: the News Analyst spends `high` pre-filtering, `xhigh`
	// on its batch, `max` ranking. Collapsed into one lane there is no honest tier to
	// print beside the name. Split by role, every lane runs at exactly one effort and
	// can state it — which is the whole reason this view exists.
	//
	// This is a *view* decision, not a taxonomy one: `role` and `effort` are already
	// per call in the index, so nothing regenerates and the Newsroom is untouched.
	$: lanes = (() => {
		const order = (index.agents ?? []).map((a) => a.id);
		const rank = new Map(order.map((id, i) => [id, i]));
		// Group by agent *and* role, preserving first-seen role order within an agent.
		const groups = new Map<string, { agent_id: string; role: string; calls: ReplayCall[] }>();
		for (const call of index.calls ?? []) {
			const key = `${call.agent_id}:${call.role}`;
			let g = groups.get(key);
			if (!g) {
				g = { agent_id: call.agent_id, role: call.role, calls: [] };
				groups.set(key, g);
			}
			g.calls.push(call);
		}

		const result: {
			key: string;
			agent_id: string;
			label: string;
			sublabel: string | null;
			color: string;
			first: boolean;
			rows: ReplayCall[][];
		}[] = [];

		// Agents in cast order; unknown callers (rank undefined) sort to the end.
		const sorted = [...groups.values()].sort((a, b) => {
			const ai = rank.get(a.agent_id) ?? 999;
			const bi = rank.get(b.agent_id) ?? 999;
			if (ai !== bi) return ai - bi;
			// Within an agent, order roles by when their first call was queued, so the
			// lanes read top-to-bottom in the order the work actually happened.
			return (
				Math.min(...a.calls.map((c) => c.queued_ms)) -
				Math.min(...b.calls.map((c) => c.queued_ms))
			);
		});

		let prevAgent: string | null = null;
		for (const g of sorted) {
			const agent = agentById.get(g.agent_id);
			const known = agent != null;
			// Only name the role when the agent actually has more than one. A lane that
			// says "Copy Editor / Enrich" when enriching is all it ever does is noise.
			const roleCount = new Set(
				[...groups.values()].filter((x) => x.agent_id === g.agent_id).map((x) => x.role)
			).size;
			result.push({
				key: `${g.agent_id}:${g.role}`,
				agent_id: g.agent_id,
				label: agent?.label ?? g.agent_id,
				sublabel: roleCount > 1 ? (ROLE_LABELS[g.role as ReplayRole] ?? g.role) : null,
				color: known ? agentColor(agent) : '#737373',
				first: g.agent_id !== prevAgent,
				rows: packRows(g.calls)
			});
			prevAgent = g.agent_id;
		}
		return result;
	})();

	// Collector lanes, above the LLM lanes.
	//
	// The scouts do the first ~2 minutes of every run and made no LLM calls, so the
	// timeline used to open on dead air with no explanation of what the run was
	// waiting for. These are non-LLM work, drawn from `sources` rather than `calls`.
	$: collectorLanes = (() => {
		const order = (index.agents ?? []).map((a) => a.id);
		const rank = new Map(order.map((id, i) => [id, i]));
		const rows = [...(index.sources ?? [])].sort((a, b) => {
			const ai = rank.get(a.agent_id) ?? 999;
			const bi = rank.get(b.agent_id) ?? 999;
			if (ai !== bi) return ai - bi;
			return a.start_ms - b.start_ms;
		});
		let prevAgent: string | null = null;
		return rows.map((s) => {
			const agent = agentById.get(s.agent_id);
			const first = s.agent_id !== prevAgent;
			prevAgent = s.agent_id;
			return {
				key: `${s.agent_id} ${s.name}`,
				label: agent?.label ?? s.agent_id,
				sublabel: s.name,
				color: agentColor(agent),
				first,
				source: s,
				// Absent means UNmeasured — the opposite of the run-level flag's default.
				// Every day published before per-source timing existed has no field here
				// *and* no real span, so defaulting to true would draw six identical
				// phase-wide slabs as though each had been individually clocked.
				measured: s.timing_measured === true
			};
		});
	})();

	// Any collector row still stretched across the whole gathering phase, which is
	// what the pipeline used to record for all of them.
	$: hasEstimatedCollectors = collectorLanes.some((l) => !l.measured);

	$: totalRows = lanes.reduce((n, l) => n + l.rows.length, 0) + collectorLanes.length;
	$: chartHeight = Math.max(80, totalRows * (LANE_H + LANE_GAP));

	function pct(ms: number): number {
		return (ms / duration) * 100;
	}

	function barColor(call: ReplayCall): string {
		return providerColor(call.provider_id);
	}

	// Concurrency sparkline, drawn as an SVG polygon over the same x scale.
	//
	// Two series share one y scale: `concActiveMax` is in-flight requests (and matches
	// run.peak_concurrency in the header), while `concMax` adds the queued backlog on
	// top and is therefore the taller of the two. They are labelled separately —
	// reporting the stacked ceiling as "peak concurrency" would contradict the header.
	$: conc = index.concurrency?.samples ?? [];
	$: concMax = Math.max(1, ...conc.map((s) => s[1] + s[2]));
	$: concActiveMax = Math.max(1, ...conc.map((s) => s[1]));
	$: activePath = (() => {
		if (conc.length === 0) return '';
		const pts = conc.map((s) => `${(s[0] / duration) * 100},${100 - (s[1] / concMax) * 100}`);
		return `0,100 ${pts.join(' ')} 100,100`;
	})();
	$: totalPath = (() => {
		if (conc.length === 0) return '';
		const pts = conc.map(
			(s) => `${(s[0] / duration) * 100},${100 - ((s[1] + s[2]) / concMax) * 100}`
		);
		return `0,100 ${pts.join(' ')} 100,100`;
	})();

	// Tick marks every ~5 minutes of run time, snapped to something readable.
	$: ticks = (() => {
		const target = 8;
		const raw = duration / target;
		const steps = [30_000, 60_000, 120_000, 300_000, 600_000, 900_000, 1_800_000];
		const step = steps.find((s) => s >= raw) ?? steps[steps.length - 1];
		const out: number[] = [];
		for (let v = 0; v <= duration; v += step) out.push(v);
		return out;
	})();

	// Every provider that served a request, with its share of the run. The count is
	// what makes this a load-balancing readout rather than decoration.
	$: routeLegend = (() => {
		const counts = new Map<string, number>();
		for (const c of index.calls ?? []) counts.set(c.provider_id, (counts.get(c.provider_id) ?? 0) + 1);
		return [...counts.entries()]
			.sort((a, b) => b[1] - a[1])
			.map(([id, n]) => ({
				id,
				count: n,
				color: providerColor(id),
				note: id === 'image' ? 'image model, not an LLM route' : 'LLM route'
			}));
	})();

	/**
	 * The effort a lane runs at.
	 *
	 * Now that lanes are split by role this is single-valued for essentially every
	 * lane, which is the point of the split. It still returns null rather than picking
	 * a winner if a role ever did mix tiers — a label that is sometimes a lie is worse
	 * than an absent one, and those bars carry effort in their tooltip regardless.
	 */
	function laneEffort(calls: ReplayCall[][]): string | null {
		const efforts = new Set(calls.flat().map((c) => c.effort));
		return efforts.size === 1 ? [...efforts][0] : null;
	}

	function handleTrackClick(event: MouseEvent) {
		const el = event.currentTarget as HTMLElement;
		const rect = el.getBoundingClientRect();
		onSeek(((event.clientX - rect.left) / rect.width) * duration);
	}

	// Playhead offset in whole pixels rather than a percentage.
	//
	// A percentage resolves to a fractional pixel offset, and the playhead sits on its
	// own compositor layer (`will-change: transform`), so a fractional translate makes
	// the compositor resample the layer's bitmap every frame — a hard-edged 2px line
	// re-spread across a different pair of device columns each time, which is the
	// shimmer. Rounding to integers means the layer composites at the same subpixel
	// phase every frame and simply steps.
	//
	// The step is invisible at every speed the player offers: at 1x a 20-minute run
	// advances ~0.7 px/s, so the line ticks one pixel about once a second; at 120x it
	// covers ~1.5 px/frame and rounding is below the noise of the motion itself.
	let trackW = 0;
	$: playheadPx = Math.round((Math.min(t, duration) / duration) * trackW) || 0;
</script>

<div class="timeline card !p-4">
	<div class="tl-head">
		<div>
			<h3 class="tl-title">Call timeline</h3>
			<p class="tl-sub">
				{index.calls.length} requests · peak concurrency {index.run.peak_concurrency} · click a bar to
				open it{measured ? '' : ' · reconstructed timings'}
			</p>
		</div>

		<!-- No toggle, no caption: a swatch beside a provider name and a count reads
		     as "this color is this provider, and it took N calls" on its own. -->
		<ul class="legend">
			{#each routeLegend as r (r.id)}
				<li title="{r.count} of {index.calls.length} requests · {r.note}">
					<span class="swatch" style="background: {r.color}"></span>{r.id}
					<span class="legend-n">{r.count}</span>
				</li>
			{/each}
		</ul>
	</div>

	<!-- What the shading inside a bar means. Swatches use the same base-plus-overlay
	     recipe as the bars, so a shade in the key is findable in the chart. -->
	<div class="key">
		{#if collectorLanes.length > 0}
			<span><span class="key-swatch key-collect"></span>collecting (not an LLM call)</span>
		{/if}
		{#if hasEstimatedCollectors}
			<span
				><span class="key-swatch key-collect key-collect-est"></span>collection span not
				individually timed</span
			>
		{/if}
		{#if measured}
			{#if hasQueueWait}
				<span><span class="key-swatch key-wait"></span>queue wait</span>
			{/if}
			<span><span class="key-swatch key-ttft"></span>waiting for first token</span>
			<span><span class="key-swatch key-stream"></span>writing</span>
			{#if !hasQueueWait}
				<span class="key-note">nothing queued this run — every request started immediately</span>
			{/if}
		{:else}
			<span><span class="key-swatch key-stream"></span>request start → end</span>
			<span class="key-note"
				>timings reconstructed from run logs — no queue wait or first-token split recorded</span
			>
		{/if}
	</div>

	<div class="tl-body">
		<div class="gutter">
			<div class="gutter-rows" style="height: {chartHeight}px">
				{#each collectorLanes as lane (lane.key)}
					<div
						class="gutter-label"
						class:continued={!lane.first}
						style="height: {LANE_H + LANE_GAP}px; --accent: {lane.color}"
						title="{lane.label} — {lane.sublabel} — {lane.source.items.toLocaleString()} items{lane.measured
							? ''
							: ' — span not individually timed'}"
					>
						<span class="gutter-tick"></span>
						<span class="gutter-text">
							{#if lane.first}
								<span class="gutter-name">{lane.label}</span>
							{/if}
							<span class="gutter-sub">
								<span class="gutter-role">{lane.sublabel}</span>
							</span>
						</span>
					</div>
				{/each}
				{#each lanes as lane (lane.key)}
					{@const eff = laneEffort(lane.rows)}
					{@const name = lane.sublabel ? `${lane.label} — ${lane.sublabel}` : lane.label}
					<div
						class="gutter-label"
						class:continued={!lane.first}
						style="height: {lane.rows.length * (LANE_H + LANE_GAP)}px; --accent: {lane.color}"
						title={eff ? `${name} — every call at ${eff} effort` : name}
					>
						<span class="gutter-tick"></span>
						<span class="gutter-text">
							<!-- The agent is named once; its further roles indent under it, so a
							     split agent reads as one agent doing several jobs rather than
							     as several unrelated agents. -->
							{#if lane.first}
								<span class="gutter-name">{lane.label}</span>
							{/if}
							<span class="gutter-sub">
								{#if lane.sublabel}<span class="gutter-role">{lane.sublabel}</span>{/if}
								{#if eff}<span class="gutter-effort" data-effort={eff}>{eff}</span>{/if}
							</span>
						</span>
					</div>
				{/each}
			</div>
		</div>

		<div class="track-wrap">
			<!-- eslint-disable-next-line svelte/valid-compile -->
			<div
				class="track"
				style="height: {chartHeight}px"
				bind:clientWidth={trackW}
				on:click={handleTrackClick}
				on:keydown={(e) => {
					if (e.key === 'Enter') handleTrackClick(e as unknown as MouseEvent);
				}}
				role="slider"
				tabindex="0"
				aria-label="Seek within the run"
				aria-valuemin={0}
				aria-valuemax={Math.round(duration / 1000)}
				aria-valuenow={Math.round(t / 1000)}
				aria-valuetext="{formatClock(t)} elapsed"
			>
				{#each ticks as tick (tick)}
					<span class="grid-line" style="left: {pct(tick)}%"></span>
				{/each}

				<!-- Collectors: non-LLM work, so these carry items rather than tokens, and
				     are outlined-and-hatched rather than solid when the pipeline recorded
				     only the phase span for them. -->
				{#each collectorLanes as lane (lane.key)}
					{@const left = pct(lane.source.start_ms)}
					{@const width = Math.max(0.12, pct(lane.source.end_ms - lane.source.start_ms))}
					<div class="row" style="height: {LANE_H}px; margin-bottom: {LANE_GAP}px">
						<div
							class="collect-bar"
							class:estimated={!lane.measured}
							class:done={lane.source.end_ms <= t}
							class:live={lane.source.start_ms <= t && lane.source.end_ms > t}
							class:future={lane.source.start_ms > t}
							style="left: {left}%; width: {width}%; --c: {lane.color}"
						>
							<span class="collect-items">{lane.source.items.toLocaleString()}</span>
						</div>
					</div>
				{/each}

				{#each lanes as lane (lane.key)}
					{#each lane.rows as row, ri (ri)}
						<div class="row" style="height: {LANE_H}px; margin-bottom: {LANE_GAP}px">
							{#each row as call (call.id)}
								{@const left = pct(call.queued_ms)}
								{@const width = Math.max(0.12, pct(call.end_ms - call.queued_ms))}
								{@const waitW = pct(call.start_ms - call.queued_ms)}
								{@const ttftW = call.first_token_ms ? pct(call.first_token_ms - call.start_ms) : 0}
								<button
									type="button"
									class="bar"
									class:done={call.end_ms <= t}
									class:live={call.queued_ms <= t && call.end_ms > t}
									class:future={call.queued_ms > t}
									class:selected={call.id === selectedCallId}
									style="left: {left}%; width: {width}%; --c: {barColor(call)}"
									on:click|stopPropagation={() => onSelectCall(call.id)}
									title={isImageCall(call)
										? `${call.task} · ${call.model} · ${formatDuration(
												call.end_ms - call.start_ms
											)} · 1 image (no token metering)`
										: call.interaction_type === 'decision' ? `${call.task} · ${call.decision_item_count} articles · ${((call.end_ms - call.start_ms) / 1000).toFixed(3)}s · typed response`
										: `${call.task} · ${call.provider_id} · ${call.effort} effort · ${formatDuration(
												call.end_ms - call.start_ms
											)} · ${formatTokens(call.output_tokens)} tok${
												call.outcome !== 'ok' ? ` · ${call.outcome}` : ''
											}`}
								>
									{#if waitW > 0.05}
										<span class="seg-wait" style="width: {(waitW / width) * 100}%"></span>
									{/if}
									<span
										class="seg-ttft"
										style="left: {(waitW / width) * 100}%; width: {(ttftW / width) * 100}%"
									></span>
									<span
										class="seg-stream"
										style="left: {((waitW + ttftW) / width) * 100}%; right: 0"
									></span>
									{#if call.outcome === 'truncated' || call.outcome === 'failed' || call.outcome === 'refused'}
										<span class="bar-flag" data-outcome={call.outcome}></span>
									{/if}
									{#if call.fallback_from}
										<span class="bar-fallback" title="Failed over from {call.fallback_from}"></span>
									{/if}
								</button>
							{/each}
						</div>
					{/each}
				{/each}

				<!-- Moved with transform, not `left`: animating a layout property re-lays-out
				     the track every frame. Translated by whole pixels rather than a
				     percentage — see `playheadPx` for why the fraction was the shimmer. -->
				<!-- The clipping layer keeps the playhead wrapper from widening the track
				     once it is translated past the right edge. -->
				<span class="playhead-layer">
					<span class="playhead" style="transform: translateX({playheadPx}px)">
						<span class="playhead-line"></span>
					</span>
				</span>
			</div>

			<!-- Concurrency: the pipeline breathing -->
			<div class="conc">
				<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
					<polygon points={totalPath} class="conc-total" />
					<polygon points={activePath} class="conc-active" />
				</svg>
				<span class="conc-playhead" style="transform: translateX({playheadPx}px)">
					<span class="playhead-line"></span>
				</span>
				<span class="conc-label"
					>concurrency · peak {concActiveMax} in flight{concMax > concActiveMax
						? ` · ${concMax} incl. queue`
						: ''}</span
				>
			</div>

			<div class="axis">
				{#each ticks as tick (tick)}
					<span class="axis-tick" style="left: {pct(tick)}%">{formatClock(tick)}</span>
				{/each}
			</div>
		</div>
	</div>
</div>

<style>
	.tl-head {
		display: flex;
		flex-wrap: wrap;
		align-items: flex-start;
		justify-content: space-between;
		gap: 0.75rem;
		margin-bottom: 0.6rem;
	}
	.tl-title {
		font-size: 0.95rem;
		font-weight: 700;
		color: #262626;
	}
	:global(.dark) .tl-title {
		color: #f5f5f5;
	}
	.tl-sub {
		font-size: 0.7rem;
		color: #737373;
	}

	.legend-n {
		font-variant-numeric: tabular-nums;
		font-weight: 700;
		opacity: 0.55;
		margin-left: 0.1rem;
	}

	.legend {
		display: flex;
		gap: 0.6rem;
		flex-wrap: wrap;
		font-size: 0.62rem;
		color: #525252;
	}
	:global(.dark) .legend {
		color: #a3a3a3;
	}
	.legend li {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}
	.swatch {
		width: 8px;
		height: 8px;
		border-radius: 2px;
		display: inline-block;
	}

	.key {
		display: flex;
		gap: 0.85rem;
		font-size: 0.6rem;
		color: #737373;
		margin-bottom: 0.5rem;
	}
	.key span {
		display: flex;
		align-items: center;
		gap: 0.25rem;
	}
	.key-note {
		font-style: italic;
		opacity: 0.85;
	}
	/* Neutral stand-in for the per-call route color. The overlays below are
	   the exact ones the bars use, so the key reads as a slice of a real bar. */
	.key-swatch {
		width: 14px;
		height: 8px;
		border-radius: 2px;
		display: inline-block;
		position: static;
		background: #8ba3c7;
	}
	.key-wait {
		background-image: repeating-linear-gradient(
			45deg,
			rgb(255 255 255 / 0.55) 0 2px,
			transparent 2px 4px
		);
	}
	.key-ttft {
		background-image: linear-gradient(rgb(0 0 0 / 0.28), rgb(0 0 0 / 0.28));
	}
	/* Same outlined recipe the collector bars use, so the key reads as a slice of one. */
	.key-collect {
		background: rgb(139 163 199 / 0.3);
		border: 1px solid rgb(139 163 199 / 0.75);
		box-sizing: border-box;
	}
	.key-collect-est {
		border-style: dashed;
		background-image: repeating-linear-gradient(
			45deg,
			rgb(139 163 199 / 0.5) 0 3px,
			transparent 3px 6px
		);
	}

	.tl-body {
		display: flex;
		gap: 0.4rem;
	}

	.gutter {
		width: 7.5rem;
		flex: none;
		display: flex;
		flex-direction: column;
	}
	.gutter-rows {
		display: flex;
		flex-direction: column;
	}

	/* One tier per lane, stated rather than encoded. Matches the badge on the
	   Newsroom's stations so the same agent reads the same in both views. */
	.gutter-effort {
		display: inline;
		font-size: 0.48rem;
		font-weight: 700;
		letter-spacing: 0.05em;
		color: #a3a3a3;
		line-height: 1.2;
	}
	.gutter-effort[data-effort='xhigh'] {
		color: #7c3aed;
	}
	.gutter-effort[data-effort='max'] {
		color: #dc2626;
	}
	:global(.dark) .gutter-effort[data-effort='xhigh'] {
		color: #c4b5fd;
	}
	:global(.dark) .gutter-effort[data-effort='max'] {
		color: #fca5a5;
	}
	.gutter-label {
		display: flex;
		align-items: center;
		gap: 0.3rem;
		font-size: 0.62rem;
		color: #525252;
		border-top: 1px solid rgb(0 0 0 / 0.06);
		overflow: hidden;
	}
	:global(.dark) .gutter-label {
		color: #a3a3a3;
		border-color: rgb(255 255 255 / 0.07);
	}
	/* A second role of the same agent: no rule above it, so the agent's lanes read as
	   one banded block rather than as separate agents that happen to be adjacent. */
	.gutter-label.continued {
		border-top-color: transparent;
	}

	.gutter-name {
		display: block;
		font-weight: 600;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	/* Indented under the agent name, and lighter than it: the role is a subdivision of
	   the agent, not a peer of it. Inline so it shares a line with the effort badge —
	   a lane is only 22px, which is two lines of this size, not three. */
	.gutter-role {
		display: inline;
		font-size: 0.55rem;
		color: #8a8a8a;
	}
	:global(.dark) .gutter-role {
		color: #8f8f8f;
	}
	/* The role line as a whole is indented, via the wrapper's text-indent, so the
	   badge stays visually attached to the role it qualifies. */
	.gutter-sub {
		display: block;
		padding-left: 0.4rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	/* Agent identity, matching the Newsroom's cast colours so the same agent reads
	   the same in both views. It sits in the label gutter, not in the plot area, so
	   it never competes with the provider colour on the bars. */
	.gutter-tick {
		width: 3px;
		align-self: stretch;
		flex: none;
		border-radius: 2px;
		background: var(--accent);
		margin: 2px 0;
	}
	.gutter-text {
		min-width: 0;
		overflow: hidden;
		font-size: 0.6rem;
		line-height: 1.25;
		text-overflow: ellipsis;
	}

	.track-wrap {
		flex: 1;
		min-width: 0;
	}

	.track {
		position: relative;
		cursor: crosshair;
		background: rgb(0 0 0 / 0.025);
		border-radius: 4px;
	}
	:global(.dark) .track {
		background: rgb(255 255 255 / 0.03);
	}

	.grid-line {
		position: absolute;
		top: 0;
		bottom: 0;
		width: 1px;
		background: rgb(0 0 0 / 0.05);
	}
	:global(.dark) .grid-line {
		background: rgb(255 255 255 / 0.06);
	}

	.row {
		position: relative;
	}

	/* Collection is not an LLM request: outlined and tinted rather than a solid
	   provider-coloured slab, so a scan of the chart never mistakes a scout pulling
	   feeds for a model generating tokens. Carries the agent's own colour, since
	   there is no provider to attribute it to. */
	.collect-bar {
		position: absolute;
		top: 4px;
		bottom: 4px;
		border-radius: 3px;
		display: flex;
		align-items: center;
		justify-content: flex-end;
		padding-right: 4px;
		overflow: hidden;
		background: color-mix(in srgb, var(--c) 22%, transparent);
		border: 1px solid color-mix(in srgb, var(--c) 55%, transparent);
	}
	.collect-bar.future {
		opacity: 0.22;
	}
	.collect-bar.done {
		opacity: 0.8;
	}
	.collect-bar.live {
		opacity: 1;
		box-shadow: 0 0 8px -2px var(--c);
	}
	/* Hatched: real work happened in this window, but the pipeline recorded only the
	   phase span for it, not this source's own start and end. */
	.collect-bar.estimated {
		background-image: repeating-linear-gradient(
			45deg,
			color-mix(in srgb, var(--c) 30%, transparent) 0 3px,
			transparent 3px 6px
		);
		border-style: dashed;
	}
	.collect-items {
		font-size: 0.5rem;
		font-weight: 700;
		font-variant-numeric: tabular-nums;
		color: #525252;
		white-space: nowrap;
	}
	:global(.dark) .collect-items {
		color: #d4d4d4;
	}

	.bar {
		position: absolute;
		top: 2px;
		bottom: 2px;
		border-radius: 3px;
		overflow: hidden;
		background: var(--c);
		border: none;
		padding: 0;
		cursor: pointer;
		transition: opacity 150ms ease, filter 150ms ease;
	}
	.bar.future {
		opacity: 0.22;
	}
	.bar.done {
		opacity: 0.78;
	}
	.bar.live {
		opacity: 1;
		box-shadow: 0 0 0 1px var(--c), 0 0 10px -2px var(--c);
	}
	@media (hover: hover) and (pointer: fine) {
		.bar:hover {
			filter: brightness(1.15);
			opacity: 1;
		}
	}
	.bar.selected {
		box-shadow: 0 0 0 2px #fff, 0 0 0 3.5px #E63946;
		opacity: 1;
		z-index: 3;
	}
	:global(.dark) .bar.selected {
		box-shadow: 0 0 0 2px #171717, 0 0 0 3.5px #E63946;
	}

	.bar span {
		position: absolute;
		top: 0;
		bottom: 0;
	}
	.bar .seg-wait {
		left: 0;
		background: repeating-linear-gradient(
			45deg,
			rgb(255 255 255 / 0.55) 0 2px,
			transparent 2px 4px
		);
	}
	.bar .seg-ttft {
		background: rgb(0 0 0 / 0.28);
	}
	.bar .seg-stream {
		background: transparent;
	}

	.bar-flag {
		left: auto !important;
		right: 0;
		width: 3px;
		background: #f59e0b;
	}
	.bar-flag[data-outcome='failed'],
	.bar-flag[data-outcome='refused'] {
		background: #ef4444;
	}
	.bar-fallback {
		left: 0 !important;
		width: 3px;
		background: #a855f7;
	}

	.playhead-layer {
		position: absolute;
		inset: 0;
		overflow: hidden;
		pointer-events: none;
		z-index: 4;
	}

	/* Zero-width wrapper with the visible line hung off its leading edge, translated by
	   whole pixels. It was full-width so a percentage translate could resolve against
	   the track; now that the offset is computed in px there is nothing to resolve
	   against, and a 0-width box cannot overflow the track in the first place. */
	.playhead {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		width: 0;
		pointer-events: none;
		z-index: 4;
		will-change: transform;
	}
	/* Whole-pixel geometry, not 1.5px at -0.75px.
	   Transform already keeps this off the layout path, but the line still shimmered:
	   a 20-minute run advances the playhead ~0.012 css px per frame at 1x, far below
	   one device pixel. A 1.5px-wide line offset by a fractional -0.75px never lands
	   on a device-pixel boundary, so every frame the compositor re-spreads its
	   antialiasing coverage across a different pair of columns -- the "blurry thing
	   moving" shimmer. At 2px/-1px both edges are integral at dpr 1 and 2, so the
	   line resamples identically frame to frame and holds still. */
	.playhead-line {
		position: absolute;
		top: 0;
		bottom: 0;
		left: -1px;
		width: 2px;
		background: #E63946;
		box-shadow: 0 0 8px 0 rgb(230 57 70 / 0.8);
	}

	.conc {
		position: relative;
		height: 44px;
		margin-top: 4px;
		border-radius: 4px;
		overflow: hidden;
		background: rgb(0 0 0 / 0.03);
	}
	:global(.dark) .conc {
		background: rgb(255 255 255 / 0.04);
	}
	.conc svg {
		width: 100%;
		height: 100%;
		display: block;
	}
	.conc-total {
		fill: rgb(139 92 246 / 0.22);
	}
	.conc-active {
		fill: rgb(230 57 70 / 0.42);
	}
	.conc-playhead {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		width: 0;
		pointer-events: none;
		will-change: transform;
	}
	.conc-playhead .playhead-line {
		box-shadow: none;
	}
	.conc-label {
		position: absolute;
		top: 2px;
		left: 5px;
		font-size: 0.55rem;
		letter-spacing: 0.06em;
		text-transform: uppercase;
		color: #737373;
	}

	.axis {
		position: relative;
		height: 1.1rem;
		margin-top: 2px;
	}
	.axis-tick {
		position: absolute;
		transform: translateX(-50%);
		font-size: 0.58rem;
		font-variant-numeric: tabular-nums;
		color: #737373;
		white-space: nowrap;
	}

	@media (max-width: 700px) {
		.gutter {
			width: 4.5rem;
		}
		.gutter-text {
			font-size: 0.55rem;
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.bar {
			transition: none;
		}
	}
</style>
