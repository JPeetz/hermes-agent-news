<script lang="ts">
	import { base } from "$app/paths";

	let email = "";
	let status: "idle" | "submitting" | "success" | "error" = "idle";
	let errorMsg = "";

	async function handleSubmit() {
		status = "submitting";
		errorMsg = "";
		try {
			const resp = await fetch("/api/subscribe", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ email })
			});
			if (resp.ok) {
				status = "success";
				email = "";
			} else {
				status = "error";
				errorMsg = "Subscription failed. Try again.";
			}
		} catch {
			status = "error";
			errorMsg = "Network error. Try again.";
		}
	}
</script>

<svelte:head>
	<title>Subscribe — Agent N's Hermes News</title>
	<meta name="description" content="Subscribe to the daily Agent N's Hermes News newsletter — Hermes Agent &amp; Hermes Desktop coverage delivered to your inbox." />
</svelte:head>

<div class="subscribe-page">
	<div class="subscribe-card">
		<h1>Stay in the Loop</h1>
		<p class="subtitle">
			Get daily Hermes Agent &amp; Hermes Desktop news delivered to your inbox.
			No spam, unsubscribe anytime.
		</p>

		{#if status === "success"}
			<div class="success-message">
				<span class="check-icon">✓</span>
				<p>You're subscribed! Check your inbox for a confirmation email.</p>
			</div>
		{:else}
			<form on:submit|preventDefault={handleSubmit} class="subscribe-form">
				<input
					type="email"
					bind:value={email}
					placeholder="you@example.com"
					required
					disabled={status === "submitting"}
					class="email-input"
				/>
				<button type="submit" disabled={status === "submitting"} class="subscribe-btn">
					{status === "submitting" ? "Subscribing..." : "Subscribe"}
				</button>
			</form>
			{#if status === "error"}
				<p class="error-msg">{errorMsg}</p>
			{/if}
		{/if}

		<p class="fine-print">
			Powered by <a href="https://buttondown.email" target="_blank" rel="noopener noreferrer">Buttondown</a>.
			Your email is safe and never shared.
		</p>
	</div>
</div>

<style>
	.subscribe-page {
		min-height: 60vh;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 40px 20px;
		background: #0A0A0A;
	}
	.subscribe-card {
		background: #1A1A1A;
		border: 1px solid #2A2A2A;
		border-radius: 12px;
		padding: 40px;
		max-width: 460px;
		width: 100%;
		text-align: center;
	}
	h1 {
		color: #FFD700;
		font-size: 24px;
		margin: 0 0 8px 0;
		font-weight: 600;
	}
	.subtitle {
		color: #AAAAAA;
		font-size: 14px;
		line-height: 1.5;
		margin: 0 0 24px 0;
	}
	.subscribe-form {
		display: flex;
		gap: 8px;
	}
	.email-input {
		flex: 1;
		padding: 12px 16px;
		border: 1px solid #333;
		border-radius: 8px;
		background: #0A0A0A;
		color: #F8F8F8;
		font-size: 14px;
		outline: none;
		transition: border-color 0.2s;
	}
	.email-input:focus {
		border-color: #FFD700;
	}
	.email-input:disabled {
		opacity: 0.6;
	}
	.subscribe-btn {
		padding: 12px 24px;
		background: linear-gradient(135deg, #FFD700, #F0C800);
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
		padding: 20px;
	}
	.check-icon {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 48px;
		height: 48px;
		border-radius: 50%;
		background: #22C55E;
		color: #0A0A0A;
		font-size: 24px;
		font-weight: bold;
		margin-bottom: 12px;
	}
	.success-message p {
		color: #CCCCCC;
		font-size: 14px;
		margin: 0;
	}
	.error-msg {
		color: #EF4444;
		font-size: 13px;
		margin: 8px 0 0 0;
	}
	.fine-print {
		color: #666;
		font-size: 11px;
		margin: 20px 0 0 0;
	}
	.fine-print a {
		color: #00FFFF;
	}
</style>