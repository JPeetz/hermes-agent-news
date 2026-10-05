<script lang="ts">
	import { onMount } from 'svelte';
	import { browser } from '$app/environment';

	let showModal = $state(false);
	let email = $state('');
	let subscribeStatus: 'idle' | 'sending' | 'success' | 'error' = $state('idle');
	let errorMsg = $state('');

	const STORAGE_KEY = 'hermesnews-subscribe-dismissed';
	const DISMISS_DURATION_MS = 7 * 24 * 60 * 60 * 1000; // 7 days

	onMount(() => {
		if (!browser) return;
		// Check if user already dismissed
		const dismissed = localStorage.getItem(STORAGE_KEY);
		if (dismissed) {
			const dismissedAt = parseInt(dismissed, 10);
			if (Date.now() - dismissedAt < DISMISS_DURATION_MS) return;
		}
		// Show after 30 seconds
		const timer = setTimeout(() => {
			showModal = true;
		}, 30000);
		return () => clearTimeout(timer);
	});

	function dismiss() {
		showModal = false;
		if (browser) {
			localStorage.setItem(STORAGE_KEY, Date.now().toString());
		}
	}

	async function subscribe(e: Event) {
		e.preventDefault();
		if (!email || !email.includes('@')) return;
		subscribeStatus = 'sending';
		try {
			const resp = await fetch('/api/subscribe', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ email, website_url: '' })
			});
			if (!resp.ok) throw new Error('Subscription failed');
			subscribeStatus = 'success';
			setTimeout(() => {
				showModal = false;
				if (browser) localStorage.setItem(STORAGE_KEY, Date.now().toString());
			}, 2000);
		} catch (e) {
			subscribeStatus = 'error';
			errorMsg = 'Something went wrong. Try again or visit the subscribe page.';
		}
	}
</script>

{#if showModal}
	<!-- Overlay -->
	<div class="modal-overlay" onclick={dismiss} onkeydown={(e) => e.key === 'Escape' && dismiss()} role="dialog" aria-modal="true" aria-label="Subscribe to newsletter">
		<!-- Modal card -->
		<div class="modal-card" onclick={(e) => e.stopPropagation()} onkeydown={() => {}} role="document">
			<!-- Close button -->
			<button class="close-btn" onclick={dismiss} aria-label="Close subscription prompt">&times;</button>

			<!-- Agent N's silhouette/logo area -->
			<div class="modal-icon">
				<svg width="40" height="40" viewBox="0 0 40 40" fill="none" aria-hidden="true">
					<circle cx="20" cy="20" r="18" stroke="#FF8C00" stroke-width="2" fill="none"/>
					<path d="M12 26 C12 18, 18 12, 20 10 C22 12, 28 18, 28 26" stroke="#00E5FF" stroke-width="2" fill="none" stroke-linecap="round"/>
					<circle cx="16" cy="18" r="1.5" fill="#FF8C00"/>
					<circle cx="24" cy="18" r="1.5" fill="#FF8C00"/>
				</svg>
			</div>

			<h2 class="modal-title">Your AI Intelligence, Delivered</h2>
			<p class="modal-subtitle">
				Every morning, <strong>Agent N</strong> curates what matters: the model
				drops, the open-source breakthroughs, the community builds you'd miss.
				<strong>15 minutes</strong> to stay ahead of 10,000+ AI researchers, builders, and founders.
			</p>

			<div class="benefit-row">
				<div class="benefit">
					<span class="benefit-icon">🧠</span>
					<span>Model releases before they trend</span>
				</div>
				<div class="benefit">
					<span class="benefit-icon">🔬</span>
					<span>arXiv gems someone already read for you</span>
				</div>
				<div class="benefit">
					<span class="benefit-icon">⚡</span>
					<span>What the community is actually building</span>
				</div>
			</div>

			{#if subscribeStatus === 'success'}
				<div class="success-message">
					<svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
						<path d="M4 10 L8 14 L16 6" stroke="#FF8C00" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
					</svg>
					<span>You're in! Check your inbox for confirmation.</span>
				</div>
			{:else}
				<form class="modal-form" onsubmit={subscribe}>
					<!-- Honeypot: hidden field to trap bots -->
					<input type="text" name="website_url" class="honeypot" tabindex="-1" autocomplete="off" aria-hidden="true" />
					<div class="input-row">
						<input
							type="email"
							class="email-input"
							placeholder="your@email.com"
							bind:value={email}
							required
							disabled={subscribeStatus === 'sending'}
							aria-label="Email address"
						/>
						<button
							type="submit"
							class="subscribe-btn"
							disabled={subscribeStatus === 'sending'}
						>
							{subscribeStatus === 'sending' ? 'Sending…' : 'Subscribe'}
						</button>
					</div>
				</form>
				{#if subscribeStatus === 'error'}
					<p class="error-text">{errorMsg}</p>
				{/if}
			{/if}

			<p class="modal-footer">
				Subscribe &mdash; and tomorrow's briefing finds you.</p>
			<p class="modal-footer-meta">
				<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M6 1v5l3 2" stroke="#666" stroke-width="1.5" stroke-linecap="round"/></svg>
				Takes 8 seconds &middot; No spam &middot; <a href="/archive" onclick={dismiss} class="footer-link">See what you missed</a>
			</p>
		</div>
	</div>
{/if}

<style>
	.modal-overlay {
		position: fixed;
		inset: 0;
		z-index: 9999;
		background: rgba(0, 0, 0, 0.7);
		backdrop-filter: blur(4px);
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 16px;
		animation: fadeIn 0.3s ease;
	}

	.modal-card {
		background: linear-gradient(145deg, #111115 0%, #1A1A20 100%);
		border: 1px solid #2A2A35;
		border-radius: 16px;
		padding: 40px 32px 28px;
		max-width: 440px;
		width: 100%;
		position: relative;
		box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 215, 0, 0.1);
		animation: slideUp 0.35s ease;
	}

	.close-btn {
		position: absolute;
		top: 12px;
		right: 16px;
		background: none;
		border: none;
		color: #666;
		font-size: 24px;
		cursor: pointer;
		width: 32px;
		height: 32px;
		display: flex;
		align-items: center;
		justify-content: center;
		border-radius: 50%;
		transition: all 0.2s;
		line-height: 1;
		padding: 0;
	}
	.close-btn:hover {
		color: #F8F8F8;
		background: rgba(255, 255, 255, 0.08);
	}

	.modal-icon {
		display: flex;
		justify-content: center;
		margin-bottom: 16px;
	}

	.modal-title {
		color: #F8F8F8;
		font-size: 22px;
		font-weight: 700;
		text-align: center;
		margin: 0 0 8px 0;
		letter-spacing: -0.3px;
	}

	.modal-subtitle {
		color: #999;
		font-size: 14px;
		line-height: 1.6;
		text-align: center;
		margin: 0 0 24px 0;
	}
	.modal-subtitle strong {
		color: #FF8C00;
	}

	.benefit-row {
		display: flex;
		flex-direction: column;
		gap: 10px;
		margin: 0 0 24px 0;
		padding: 16px;
		background: rgba(255, 215, 0, 0.04);
		border: 1px solid rgba(255, 215, 0, 0.1);
		border-radius: 10px;
	}

	.benefit {
		display: flex;
		align-items: center;
		gap: 10px;
		color: #CCC;
		font-size: 13px;
		font-weight: 400;
	}
	.benefit-icon {
		font-size: 16px;
		width: 20px;
		text-align: center;
		flex-shrink: 0;
	}

	.honeypot {
		position: absolute;
		left: -9999px;
		opacity: 0;
		height: 0;
		width: 0;
		overflow: hidden;
	}

	.input-row {
		display: flex;
		gap: 8px;
	}

	.email-input {
		flex: 1;
		padding: 10px 14px;
		border: 1px solid #333;
		border-radius: 8px;
		background: #0A0A0A;
		color: #F8F8F8;
		font-size: 14px;
		outline: none;
		transition: border-color 0.2s;
	}
	.email-input:focus {
		border-color: #FF8C00;
	}
	.email-input:disabled {
		opacity: 0.6;
	}

	.subscribe-btn {
		padding: 10px 20px;
		background: linear-gradient(135deg, #FF8C00, #E6C200);
		color: #0A0A0A;
		border: none;
		border-radius: 8px;
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		transition: opacity 0.2s;
		white-space: nowrap;
	}
	.subscribe-btn:hover {
		opacity: 0.9;
	}
	.subscribe-btn:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.success-message {
		display: flex;
		align-items: center;
		gap: 10px;
		justify-content: center;
		padding: 12px;
		color: #FF8C00;
		font-size: 14px;
		font-weight: 500;
	}

	.error-text {
		color: #FF6B6B;
		font-size: 13px;
		text-align: center;
		margin: 12px 0 0 0;
	}

	.modal-footer {
		color: #666;
		font-size: 12px;
		text-align: center;
		margin: 24px 0 0 0;
	}
	.modal-footer a {
		color: #00E5FF;
		text-decoration: none;
	}
	.modal-footer a:hover {
		text-decoration: underline;
	}

	.modal-footer-meta {
		color: #555;
		font-size: 11px;
		text-align: center;
		margin: 6px 0 0 0;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 4px;
	}
	.footer-link {
		color: #00E5FF !important;
		text-decoration: none;
		cursor: pointer;
	}
	.footer-link:hover {
		text-decoration: underline;
	}

	@keyframes fadeIn {
		from { opacity: 0; }
		to { opacity: 1; }
	}

	@keyframes slideUp {
		from { opacity: 0; transform: translateY(20px); }
		to { opacity: 1; transform: translateY(0); }
	}
</style>