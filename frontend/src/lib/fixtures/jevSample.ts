/** Synthetic decisions for the explicitly labelled replay demo. No production data. */
export const JEV_SAMPLE_ARTICLES = [
	{ id: 'demo-model', title: 'Open model release brings a smaller inference footprint', source: 'Example AI lab', snippet: 'A research lab released model weights and inference benchmarks.', relevance: 'relevant', probabilities: { relevant: 0.42, irrelevant: 0.5, insufficient_evidence: 0.08 }, confidence: 0.1, critical_probability: 0.83, effective_keep: true, fallback_reason: null },
	{ id: 'demo-weather', title: 'Coastal forecast updated for the weekend', source: 'Example weather service', snippet: 'Rain and strong winds are expected along the coast.', relevance: 'irrelevant', probabilities: { relevant: 0.01, irrelevant: 0.98, insufficient_evidence: 0.01 }, confidence: 0.98, critical_probability: null, effective_keep: false, fallback_reason: null },
	{ id: 'demo-unclear', title: 'A new announcement is coming', source: 'Example publication', snippet: 'More information will be shared tomorrow.', relevance: 'insufficient_evidence', probabilities: { relevant: 0.16, irrelevant: 0.09, insufficient_evidence: 0.75 }, confidence: 0.67, critical_probability: null, effective_keep: true, fallback_reason: 'insufficient_evidence' }
];

const instruction = 'You are a bounded news relevance adjudicator. Judge only the supplied title, source, and snippet. Use relevant when the bounded evidence is about an AI/ML model, company, product, research, safety, policy, infrastructure, controversy, or other substantive AI news; use irrelevant when it is outside that scope; use insufficient_evidence when the evidence cannot support either call.';
const state: Record<string, unknown> = {};
const questions: Record<string, unknown> = {};
const answers: Record<string, unknown> = {};
JEV_SAMPLE_ARTICLES.forEach((article, i) => {
	const suffix = String(i).padStart(4, '0');
	const variable = `article_${suffix}`;
	const { id, title, source, snippet } = article;
	state[variable] = { id, title, source, snippet };
	questions[`r_${suffix}`] = { type: 'choice', instructions: { article: `\`${variable}\``, instruction }, criteria: { relevant: null, irrelevant: null, insufficient_evidence: null } };
	questions[`c_${suffix}`] = { type: 'noul', instructions: { article: `\`${variable}\``, question: 'Is this an important story supported by the supplied evidence?' } };
	answers[`r_${suffix}`] = { type: 'choice', choice: article.relevance, probabilities: article.probabilities, confidence: article.confidence };
	// The unused importance answers remain in the raw view, never the readable one.
	answers[`c_${suffix}`] = { type: 'noul', noul: article.critical_probability ?? 0.99 };
});
export const JEV_SAMPLE_REQUEST = JSON.stringify({ model: 'jev-1.13.0', state, questions });
export const JEV_SAMPLE_RESPONSE = JSON.stringify({
	schema_version: 'jev-relevance-replay/v1', articles: JEV_SAMPLE_ARTICLES,
	raw_response: { model: 'jev-1.13.0', answers, usage: { input_tokens: 980, output_tokens: 219 } }
});
