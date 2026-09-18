import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decodeItemTitle } from '../src/lib/services/itemTitles.ts';

test('decodes apostrophes from the reported social title without changing source data', () => {
	const original = {
		id: 'feef9fe16bc1',
		title: 'Meanwhile, down here on earth, this is what&#39;s really happening while we&#39;re all forced to ent...',
		url: 'https://example.com/post?first=1&second=2',
		importance: 75
	};
	const result = decodeItemTitle(original);
	assert.equal(result.title, "Meanwhile, down here on earth, this is what's really happening while we're all forced to ent...");
	assert.ok(original.title.includes('&#39;'));
	assert.equal(result.id, original.id);
	assert.equal(result.url, original.url);
	assert.equal(result.importance, original.importance);
});

test('handles named, decimal and hexadecimal entities as Unicode text', () => {
	assert.equal(
		decodeItemTitle({ title: '&ldquo;Q&amp;A&rdquo; &#x2014; it&#x27;s &eacute; &#128640;' }).title,
		'“Q&A” — it\'s é 🚀'
	);
});

test('keeps literal ampersands, unknown entities and already decoded titles intact', () => {
	for (const title of ['R&D & nothing; &notAnEntity;', 'An AI’s Q&A', '', 'Compare x < y']) {
		assert.equal(decodeItemTitle({ title }).title, title);
	}
});

test('does not recursively decode or treat decoded text as markup', () => {
	assert.equal(decodeItemTitle({ title: '&amp;#39;' }).title, '&#39;');
	assert.equal(
		decodeItemTitle({ title: '&lt;img src=x onerror=alert(1)&gt;' }).title,
		'<img src=x onerror=alert(1)>'
	);
});
