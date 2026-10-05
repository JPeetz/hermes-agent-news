#!/usr/bin/env python3
"""
Hero Image Generator

Generates daily hero images with the Agent N character via configured image provider.
The character is placed in topic-related scenes based on the day's top topics.

Supports two initialization modes:
1. New: HeroGenerator.from_config(config) - uses unified ImageClient abstraction
2. Legacy: HeroGenerator(api_key, endpoint, model) - backwards compatible, deprecated

Powered by JP Technologies — https://jptechnologies.vercel.app
"""

import os
import re
import logging
import warnings
from pathlib import Path
from typing import Optional, List, Any, Dict, TYPE_CHECKING

from agents.cost_tracker import price_image_usage
from generators.image_optimizer import optimize_hero_image

if TYPE_CHECKING:
    from agents.config import ImageProviderConfig
    from generators.image_client import BaseImageClient

logger = logging.getLogger(__name__)


class HeroGenerator:
    """Generates daily hero images with Agent N mascot via configured image provider."""

    # Agent N character sheet (used by kie mode as input_urls; others read bytes)
    AGENT_N_REFERENCE_URL = "https://files.catbox.moe/5llsue.jpg"
    AGENT_N_REFERENCE = Path(__file__).parent.parent / "frontend" / "static" / "assets" / "agent-n-reference.png"

    # Agent N character anchor — COPY-PASTED VERBATIM from the v2 character sheet
    # (vision-verified 2026-10-05: black bob, brown eyes, white high-collar top with
    # black 'N' on collar, fingerless gloves, cargo pants, sneakers, ORANGE accent
    # palette). The palette is ORANGE (#FF8C00) — NOT cyan. Including "cyan" in the
    # prompt dragged the generated palette off the sheet (the rendered hero came out
    # ~14% cyan-blue against the sheet's orange), a daily identity wobble source.
    CHARACTER_ANCHOR = (
        "Agent N: a young woman, 24, 168cm (5'6\"), #1A1A1A black hair, brown eyes, "
        "black bob, black over-ear headphones with 'N' logo, white high-collar sleeveless "
        "top with black trim and front zipper, black 'N' on collar, black tactical cargo "
        "pants with thigh pockets, black fingerless tactical gloves, black high-top "
        "sneakers with white stripe, utility belt. "
        "Palette: dark gray (#1A1A1A, #0F0F0F), orange (#FF8C00), white (#F8F8F8). "
        "No neon pink, no cyan, no graffiti."
    )

    # Topic-to-visual mapping for scene generation
    VISUAL_MAPPINGS = {
        "infrastructure": "server racks, cooling systems, blue LED glow, data center",
        "datacenter": "server racks, cooling systems, blue LED glow",
        "safety": "shield icons, protective barriers, guardrails",
        "alignment": "scales of balance, alignment targets",
        "research": "floating papers, neural network diagrams, lab setting",
        "papers": "scientific documents, equations, research environment",
        "robotics": "robot arms, mechanical components, factory setting",
        "model": "neural network visualization, glowing nodes, architecture",
        "release": "rocket launch, celebration confetti, announcement banners",
        "regulation": "gavel, scales of justice, official documents",
        "funding": "growth charts, money symbols, investment visuals",
        "multimodal": "eyes, cameras, sound waves, multiple sensory inputs",
        "agent": "autonomous systems, workflow diagrams, connected tools, Hermes interface",
        "open source": "connected nodes, community gathering, collaboration",
        "reasoning": "thought bubbles, chain of logic, decision trees",
        "benchmark": "performance charts, comparison graphs, trophy",
        "language": "floating text, speech bubbles, translation symbols",
        "vision": "camera lens, image processing, visual recognition",
        "code": "terminal screens, code snippets, developer workspace",
        "security": "locks, shields, firewall barriers, protection symbols",
        "training": "compute clusters, gradient flows, learning curves",
        "deployment": "cloud infrastructure, scaling arrows, production systems",
        "desktop": "computer desktop interface, window management, bot screen",
        "hermes": "robot assistant, AI agent interface, glowing cyan circuits",
        "bot": "automated systems, chat interface, streaming data",
    }

    def __init__(
        self,
        client: Optional['BaseImageClient'] = None,
        # Legacy parameters (deprecated)
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize hero generator.

        Preferred: Use from_config() classmethod for config-based initialization.

        Args:
            client: ImageClient instance (preferred, new pattern)
            api_key: DEPRECATED - API key for legacy mode. Use from_config() instead.
            endpoint: DEPRECATED - API endpoint for legacy mode.
            model: DEPRECATED - Model name for legacy mode.
        """
        if client is not None:
            # New pattern: use provided ImageClient
            self.client = client
            logger.info("HeroGenerator initialized with ImageClient")
        elif api_key is not None:
            # Legacy pattern: create OpenAICompatibleClient internally
            warnings.warn(
                "Passing api_key, endpoint, model to HeroGenerator() is deprecated. "
                "Use HeroGenerator.from_config(config) instead.",
                DeprecationWarning,
                stacklevel=2
            )
            from generators.image_client import OpenAICompatibleClient

            # Legacy defaults
            legacy_endpoint = endpoint
            legacy_model = model or "gemini-3-pro-image"

            self.client = OpenAICompatibleClient(
                api_key=api_key,
                endpoint=legacy_endpoint,
                model=legacy_model
            )
            logger.info(f"HeroGenerator initialized in legacy mode (deprecated)")
        else:
            raise ValueError(
                "HeroGenerator requires an ImageClient. Use HeroGenerator.from_config(config) "
                "or pass client parameter directly."
            )

        # Verify reference exists (warning only; kie mode uses remote URL)
        if hasattr(self.client, 'reference_url') and self.client.reference_url:
            logger.info(f"HeroGenerator initialized with remote reference URL")
        elif not self.AGENT_N_REFERENCE.exists():
            logger.warning(f"Agent N reference image not found at {self.AGENT_N_REFERENCE}")

    @classmethod
    def from_config(cls, config: 'ImageProviderConfig') -> 'HeroGenerator':
        """
        Create hero generator from ImageProviderConfig.

        Args:
            config: ImageProviderConfig with api_key, endpoint, model, mode

        Returns:
            Configured HeroGenerator instance

        Raises:
            ValueError: If ImageClient creation fails
        """
        from generators.image_client import ImageClient

        try:
            client = ImageClient.from_config(config)
        except ValueError as e:
            # Add mode-specific troubleshooting guidance
            if config.mode == "native":
                raise ValueError(
                    f"{e}\n\n"
                    f"Troubleshooting (native mode):\n"
                    f"- Verify your GOOGLE_API_KEY is valid\n"
                    f"- Ensure google-genai SDK is installed: pip install google-genai\n"
                    f"- Check that your API key has access to image generation models"
                ) from e
            elif config.mode == "openrouter":
                raise ValueError(
                    f"{e}\n\n"
                    f"Troubleshooting (openrouter mode):\n"
                    f"- Verify your OPENROUTER_API_KEY is valid and has credit\n"
                    f"- Check that '{config.model}' supports image output on OpenRouter\n"
                    f"- Endpoint defaults to https://openrouter.ai/api/v1 if unset"
                ) from e
            else:
                raise ValueError(
                    f"{e}\n\n"
                    f"Troubleshooting (openai-compatible mode):\n"
                    f"- Verify your endpoint URL is correct: {config.endpoint}\n"
                    f"- Check that your api_key has proper permissions\n"
                    f"- Ensure the proxy supports image generation"
                ) from e

        return cls(client=client)

    def _extract_visuals(self, topics: List[Any]) -> List[str]:
        """Extract visual elements from topic names."""
        visuals = []
        for topic in topics:
            name_lower = topic.name.lower() if hasattr(topic, 'name') else str(topic).lower()
            for keyword, visual in self.VISUAL_MAPPINGS.items():
                if keyword in name_lower:
                    visuals.append(visual)
                    break
        return visuals if visuals else ["abstract AI visualization, neural networks, data flow patterns"]

    def _get_topic_names(self, topics: List[Any]) -> List[str]:
        """Extract topic names from topic objects."""
        names = []
        for topic in topics:
            if hasattr(topic, 'name'):
                names.append(topic.name)
            elif isinstance(topic, dict) and 'name' in topic:
                names.append(topic['name'])
            else:
                names.append(str(topic))
        return names

    def _strip_markdown_links(self, text: str) -> str:
        """Strip markdown link syntax, keeping just the link text."""
        # Convert [text](url) to just text
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    def _get_topic_summaries(self, topics: List[Any]) -> List[Dict[str, str]]:
        """Extract topic names and clean descriptions for prompt context.

        Topic text descends from untrusted feed content, and this prompt is
        published verbatim as hero_image_prompt, so it gets the shared
        normalization (invisible/bidi characters stripped) plus length caps.
        The image model has no instruction/data channel split, so unlike the
        analyzer prompts there is no nonce fence here.
        """
        from agents.prompt_security import normalize_untrusted_text

        summaries = []
        for topic in topics:
            name = ""
            description = ""

            if hasattr(topic, 'name'):
                name = topic.name
            elif isinstance(topic, dict) and 'name' in topic:
                name = topic['name']
            else:
                name = str(topic)

            if hasattr(topic, 'description'):
                description = self._strip_markdown_links(topic.description)
            elif isinstance(topic, dict) and 'description' in topic:
                description = self._strip_markdown_links(topic['description'])

            summaries.append({
                "name": normalize_untrusted_text(name)[:160],
                "description": normalize_untrusted_text(description)[:600],
            })

        return summaries

    def _build_prompt(self, topic_summaries: List[Dict[str, str]], visual_elements: List[str], date: Optional[str] = None) -> str:
        """Build the image generation prompt from topics and visuals.

        Hybrid of (a) the editorial-infographic style the original AATF pipeline used
        (feed the REAL topic names + descriptions into the scene so the hero visualises
        actual news content — model names, story themes — instead of an abstract room)
        and (b) the Agent N character identity locked to the sheet. The mascot actively
        engages with the day's content (per topic), and composition variety is seeded
        per-date so consecutive days diverge.
        """
        # Talk about the character as we always do: identity is the sheet.
        # ------------------------------------------------------------------ #
        # Story blocks: the real news content that anchors the scene.
        story_blocks = []
        for i, summary in enumerate(topic_summaries, 1):
            part = f"Topic {i}: {summary['name']}"
            if summary.get('description'):
                desc = summary['description'].replace('**', '').replace('*', '')
                part += f" — {desc[:180]}"
            story_blocks.append(part)
        stories = "\n".join(story_blocks) if story_blocks else "the day's Hermes Agent news"

        lead_topic = topic_summaries[0]['name'] if topic_summaries else ""
        comp = self._pick_composition(lead_topic, date)

        # Scene: Agent N actively engaged with the content, editorial-infographic feel.
        visuals = ", ".join(visual_elements[:2]) if visual_elements else "holographic news panels"
        scene = (
            f"{comp.pose} as she lays out and studies the day's top stories.\n\n"
            f"## Today's stories\n{stories}\n\n"
            f"## Scene\nA playful, colorful editorial illustration in a dark tech "
            f"newsroom. Visible as glowing panels and motifs around her: {visuals}."
            f" {comp.scene_touch} Muted, readable short labels are allowed on the news "
            f"panels (topic names, model names) but no long sentences.\n"
            f"Composition: {comp.camera_angle}, {comp.lighting}. Vibrant, energetic, "
            f"tech-optimistic mood, dark background with orange-accent dataviz."
        )

        return f"{self.CHARACTER_ANCHOR}\n\n{scene}"

    # ------------------------------------------------------------------ #
    # Composition variety — combinatorial pose/scene pools with anti-repeat #

    # Pose pool — every pose shows Agent N actively ENGAGED with the day's news content
    # (an analyst laying out and studying today's stories), so the hero visualises the
    # actual topics. Variety comes from the pose + camera + lighting combos.
    _POSES = [
        "she leans forward at a single illuminated work terminal, one hand on a keyboard, scanning the day's top story open before her",
        "she points at a large translucent story panel, tracing the line between two news items",
        "she stands at a long desk facing one big screen, arms loosely crossed, reading the headline cards laid across it",
        "she pages through a stack of glowing story cards fanned out on the desk like a newsroom brief",
        "she flicks a story card up into the air toward the glowing panel wall, building today's line-up",
        "she sits at a night desk, elbows down, comparing two model-name cards side by side",
        "she raises one hand to her headphones, pivoting her head toward the panel with the day's top story",
        "she turns from the story wall back toward the viewer, one hand open as if presenting today's line-up",
        # The four that best animate a news-editorial scene (each stays one clause):
        "she slides a kanban story card from 'drafts' to 'live' with two fingers",
        "she holds up a slim phone, thumb hovering over a headline notification, the other hand free",
        "she underlines a pass on a glowing briefing card, distilling a wall of messages to a few lines",
        "she rewinds a glowing timeline with a pinch, pulling a corrected story card back into view",
    ]
    # Camera angle pool.
    _CAMERA_ANGLES = [
        "at eye level, close range",
        "at a low three-quarter angle",
        "from slightly above, medium distance",
        "in a wide three-quarter profile",
        "from a low heroic angle",
        "in a tight over-the-shoulder frame",
        "from a gentle high angle showing the whole workspace",
    ]
    # Lighting / environment pool (keeps the dark tech brand palette).
    _LIGHTING = [
        "soft orange rim light against a deep charcoal background with faint circuit traces",
        "a warm amber pool of light in an otherwise dark command room",
        "cool dark ambience broken by a single warm orange glow above the console",
        "low orange backlight that traces her silhouette edges against black",
        "dim room light with warm gold highlights tracing the panels",
        "a single desk lamp glow in an otherwise dark late-night workspace",
    ]
    # Scene touch per lead topic keyword (topic->scene, so the message stays clear).
    _SCENE_TOUCHES = {
        "release": "A launch banner glows softly overhead in amber.",
        "security": "Shield icons float in faint orange outline near her.",
        "hermes": "Faint orange Hermes 'H' motifs pulse in the background.",
        "agent": "Thin orange workflow lines connect floating nodes around the room.",
        "desktop": "A clean desktop workspace stands behind her, screen dimmed.",
        "bot": "Slender bot silhouettes stand ready in the shadows behind her.",
        "research": "She reviews a tall stack of summarized reports on the desk.",
        "model": "Abstract neural nodes drift through the air around her.",
        "code": "Screens glow softly with code-like falling vertical lines (no readable text).",
        "infrastructure": "Server racks recede in perspective behind her.",
        "community": "Connected-node patterns trace along the far wall.",
        "open source": "A loose constellation of contributing nodes orbits her workstation.",
        "reasoning": "Soft decision-branch lines trace across the dark floor.",
        "funding": "A gentle upward growth curve of light rises beside her.",
        "kanban": "A tall wall of kanban cards glows with faint orange outlines.",
        "phone": "A soft phone-screen glow reflects warm light onto her face.",
        "cron": "A wall clock face floats faintly in the dark behind her.",
        "migration": "Two sets of terminal panes sit side by side on the desk bridging a gap.",
        "memory": "Glowing notebook spines line a shelf behind her like a small library.",
    }
    _SCENE_TOUCH_DEFAULT = "The scene stays dark and tech-forward, matching her palette."

    def _pick_composition(self, lead_topic: str, date: Optional[str] = None) -> Any:
        """Choose a pose/camera/lighting combination, avoiding recent repeats.

        Topic words bias which pose is used so the composition still tells the day's
        story. Camera/lighting rotate on a deterministic per-date seed so consecutive
        days differ and a regenerate of the same date reproduces the same composition.
        """
        import hashlib
        from types import SimpleNamespace

        topic_l = (lead_topic or "").lower()
        seed = hashlib.sha256((date or "").encode("utf-8")).hexdigest()
        # Default pose index, biased by topic keyword -> a pose slot.
        pose_idx = int(seed[:8], 16) % len(self._POSES)
        for key, _pose_pool_key in (  # noqa: B023
            ("security", 1), ("release", 0), ("research", 5), ("desktop", 5),
            ("code", 1), ("infrastructure", 7), ("community", 3), ("agent", 4),
            ("hermes", 4), ("model", 6), ("reasoning", 6), ("funding", 2),
            ("bot", 3), ("open source", 8), ("multimodal", 8),
            # Story-inspired topic -> vignette pose (index into _POSES).
            ("kanban", 8), ("board", 8), ("task", 8),
            ("phone", 9), ("telegram", 9), ("mobile", 9), ("ios", 9),
            ("grill", 10), ("smoker", 10), ("bbq", 10), ("brisket", 10),
            ("deploy", 11), ("pr", 11), ("kubernetes", 11), ("k8s", 11),
            ("briefing", 12), ("summary", 12), ("digest", 12), ("ledger", 12),
            ("homelab", 13), ("self-host", 13), ("selfhost", 13), ("optiplex", 13),
            ("swarm", 14), ("team", 14), ("fleet", 14), ("agents", 14),
            ("terminal", 15), ("cli", 15), ("scroll", 15),
            ("rollback", 16), ("rewind", 16), ("roll back", 16),
            ("tablet", 17), ("companion", 17), ("dashboard", 17),
        ):
            if key in topic_l:
                pose_idx = _pose_pool_key % len(self._POSES)
                break

        total = len(self._POSES) * len(self._CAMERA_ANGLES) * len(self._LIGHTING)
        combo = int(seed[8:16], 16) % total

        c = combo % len(self._CAMERA_ANGLES)
        remainder = combo // len(self._CAMERA_ANGLES)
        l = remainder % len(self._LIGHTING)
        p = (remainder // len(self._LIGHTING)) % len(self._POSES)
        # Keep lead-topic pose outright (topic drives the scene for clarity).
        p = pose_idx

        scene_touch = self._SCENE_TOUCH_DEFAULT
        for key, touch in self._SCENE_TOUCHES.items():
            if key in topic_l:
                scene_touch = touch
                break

        return SimpleNamespace(
            pose=self._POSES[p],
            camera_angle=self._CAMERA_ANGLES[c],
            lighting=self._LIGHTING[l],
            scene_touch=scene_touch,
        )

    async def generate(
        self,
        top_topics: List[Any],
        date: str,
        output_dir: Path,
        custom_prompt: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Generate hero image based on top topics.

        Args:
            top_topics: List of TopTopic objects or dicts with topic info
            date: Date string (YYYY-MM-DD) for output path
            output_dir: Base output directory for web data
            custom_prompt: Optional custom prompt to override auto-generated prompt

        Returns:
            Dict with 'path' (relative URL path) and 'prompt' (used prompt), or None on failure
        """
        # Read reference image (optional: kie mode uses remote URL from client config)
        character_bytes = None
        if hasattr(self.client, 'reference_url') and self.client.reference_url:
            logger.info("Agent N reference provided via client config (remote URL)")
        elif self.AGENT_N_REFERENCE.exists():
            try:
                with open(self.AGENT_N_REFERENCE, "rb") as f:
                    character_bytes = f.read()
            except Exception as e:
                logger.warning(f"Failed to read agent N reference image: {e}")
        else:
            logger.warning("No reference image available; generating without one")

        # Extract visual elements and topic summaries from all available topics
        visual_elements = self._extract_visuals(top_topics)
        topic_summaries = self._get_topic_summaries(top_topics)

        # Build prompt
        if custom_prompt:
            instructions = custom_prompt
        else:
            instructions = self._build_prompt(topic_summaries, visual_elements, date)

        topic_names = [s['name'] for s in topic_summaries]
        logger.info(f"Generating hero image for {date} with topics: {topic_names}")

        # Log the prompt for replay
        prompt_logged = instructions
        replay_context = {
            "caller": "hero_generator.compose",
            "provider_model": self.client.model if hasattr(self.client, 'model') else "kie",
            "prompt": prompt_logged,
        }
        # Set callback URL on the Kie client if available
        if hasattr(self.client, 'callback_url') and not self.client.callback_url:
            import os as _os
            base_url = _os.environ.get("PIPELINE_BASE_URL", "")
            if base_url:
                self.client.callback_url = f"{base_url.rstrip('/')}/api/image-callback"
                logger.info(f"Set Kie callback URL: {self.client.callback_url}")
        import time as _time_module
        from agents.replay_recorder import get_recorder
        recorder = get_recorder()
        replay_call_id = recorder.start_call(
            request_id=None,
            context=replay_context,
        )
        recorder.mark_started(replay_call_id)
        _call_start = _time_module.time()

        try:
            # Use ImageClient for generation
            response = await self.client.generate(
                prompt=instructions,
                reference_image=character_bytes,
                aspect_ratio="3:2",
                image_size="2K"
            )

            # Save raw image from API
            png_path = output_dir / "data" / date / "hero.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)

            with open(png_path, "wb") as f:
                f.write(response.image_data)

            # Optimize to compressed WebP
            webp_path = optimize_hero_image(png_path)
            png_path.unlink()  # Remove original PNG

            # Return relative URL path for web serving
            relative_url = f"/data/{date}/hero.webp"

            # Try to extract creditsConsumed from Kie response for accurate cost
            kie_credits = None
            if hasattr(self.client, '__class__') and 'KieImageClient' in type(self.client).__name__:
                # Kie reports credits in the poll response's data.creditsConsumed field.
                # Fallback: known fixed cost for the model used in this pipeline.
                passthrough = getattr(response, 'passthrough', None) or {}
                kie_credits = passthrough.get('creditsConsumed', None)
                if kie_credits:
                    logger.info(f"Kie: actual credits consumed: {kie_credits}")
                else:
                    # Fixed rate: 4 credits per image, 1000 credits = $5
                    # 4 * ($5/1000) = $0.02
                    kie_credits = 4
                    logger.info(f"Kie: using fixed rate ({kie_credits} credits = $0.02)")

            # Token accounting, when the provider reported it. Priced separately from
            # LLM calls because an image response bills three token classes at three
            # different rates -- see agents.cost_tracker.price_image_usage.
            cost = price_image_usage(response.usage) if response.usage else None
            if not cost and kie_credits:
                from agents.cost_tracker import ImageCost
                kie_cost = (kie_credits * 5.0) / 1000.0
                cost = ImageCost(
                    input_tokens=0,
                    image_tokens=kie_credits,
                    image_cost=kie_cost,
                    text_tokens=0,
                )
            if cost:
                logger.info(
                    f"Hero image cost: ${cost.total_cost:.4f} "
                    f"({cost.image_tokens} image + {cost.text_tokens} thinking "
                    f"+ {cost.input_tokens} input tokens)"
                )

            result = {
                "path": relative_url,
                "prompt": instructions,
            }
            if cost:
                result["usage"] = {
                    "input_tokens": cost.input_tokens,
                    "image_tokens": cost.image_tokens,
                    "text_tokens": cost.text_tokens,
                    "cost_usd": round(cost.total_cost, 6),
                    "model": response.model,
                }
            # Record in replay as successful call
            recorder.finish_call(replay_call_id, response={
                "model": response.model or "kie",
                "usage": result.get("usage"),
                "path": relative_url,
            })
            return result

        except RuntimeError as e:
            # ImageClient raises RuntimeError for API errors
            logger.error(f"Hero image generation failed: {e}")
            recorder.finish_call(replay_call_id, error=e)
            return None
        except Exception as e:
            logger.error(f"Hero image generation failed unexpectedly: {e}")
            recorder.finish_call(replay_call_id, error=e)
            return None

    def generate_sync(
        self,
        top_topics: List[Any],
        date: str,
        output_dir: Path,
        custom_prompt: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """
        Synchronous wrapper for generate() for use in non-async contexts.
        """
        import asyncio

        # Create event loop if needed
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.generate(top_topics, date, output_dir, custom_prompt)
                )
                return future.result()
        except RuntimeError:
            # No running loop, create one
            return asyncio.run(self.generate(top_topics, date, output_dir, custom_prompt))

    async def edit(
        self,
        existing_image_path: Path,
        edit_instructions: str,
        date: str,
        output_dir: Path
    ) -> Optional[Dict[str, str]]:
        """
        Edit an existing hero image with specific changes.

        Args:
            existing_image_path: Path to the existing hero image
            edit_instructions: What to change in the image
            date: Date string (YYYY-MM-DD) for output path
            output_dir: Base output directory for web data

        Returns:
            Dict with 'path' (relative URL path) and 'prompt' (used prompt), or None on failure
        """
        # Read existing hero image
        try:
            with open(existing_image_path, "rb") as f:
                hero_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read existing hero image: {e}")
            return None

        # Build edit prompt
        instructions = f"""You are editing an existing hero image. The attached image is the current version which is GOOD.

DO NOT regenerate the entire image. Make ONLY the following specific change:

{edit_instructions}

IMPORTANT:
- Keep the overall composition, style, and colors the same
- Preserve everything else exactly as it appears
- Only modify what is explicitly requested above
- The result should look like a minor edit, not a new image"""

        logger.info(f"Editing hero image for {date}: {edit_instructions[:50]}...")

        try:
            # Use ImageClient for generation (edit is just generate with edit prompt and reference)
            response = await self.client.generate(
                prompt=instructions,
                reference_image=hero_bytes,
                aspect_ratio="3:2",
                image_size="2K"
            )

            # Save raw image from API
            png_path = output_dir / "data" / date / "hero.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)

            with open(png_path, "wb") as f:
                f.write(response.image_data)

            # Optimize to compressed WebP
            webp_path = optimize_hero_image(png_path)
            png_path.unlink()  # Remove original PNG

            # Return relative URL path for web serving
            relative_url = f"/data/{date}/hero.webp"

            logger.info(f"Hero image edited and optimized: {webp_path}")

            return {
                "path": relative_url,
                "prompt": instructions
            }

        except RuntimeError as e:
            logger.error(f"Hero image edit failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Hero image edit failed unexpectedly: {e}")
            return None

    def edit_sync(
        self,
        existing_image_path: Path,
        edit_instructions: str,
        date: str,
        output_dir: Path
    ) -> Optional[Dict[str, str]]:
        """
        Synchronous wrapper for edit() for use in non-async contexts.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.edit(existing_image_path, edit_instructions, date, output_dir)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(self.edit(existing_image_path, edit_instructions, date, output_dir))


def initialize_hero_generator(config: Optional['ImageProviderConfig']) -> Optional['HeroGenerator']:
    """
    Initialize HeroGenerator from config, returning None if not configured.

    This is the preferred entry point for pipeline code. It handles missing
    configuration gracefully with clear warning messages.

    Args:
        config: ImageProviderConfig or None

    Returns:
        HeroGenerator if configured, None otherwise (with warning logged)

    Note:
        When this returns None, the pipeline should:
        - Set hero_image_url to null in summary.json
        - Set hero_image_prompt to null in summary.json
        - Continue without hero images
    """
    if config is None:
        logger.warning(
            "Hero image generation disabled: no 'image' section in providers.yaml. "
            "To enable, add image provider config. You can run "
            "scripts/regenerate_hero.py later to generate images."
        )
        return None

    try:
        return HeroGenerator.from_config(config)
    except ValueError as e:
        logger.warning(
            f"Hero image generation disabled: {e}. "
            "Check your image provider configuration in providers.yaml."
        )
        return None
    except Exception as e:
        logger.warning(
            f"Hero image generation disabled due to initialization error: {e}. "
            "Pipeline will continue without hero images."
        )
        return None


if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Generate hero image for a date")
    parser.add_argument("date", help="Date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", default="./web", help="Output directory")
    parser.add_argument("--prompt", help="Custom prompt override")
    args = parser.parse_args()

    # Mock topics for testing
    mock_topics = [
        {"name": "AI Infrastructure Investments"},
        {"name": "Reasoning Model Advances"},
        {"name": "Open Source LLMs"}
    ]

    # Try to load config, fall back to legacy mode for testing
    try:
        from agents.config import load_config
        config = load_config("./config")
        if config.image:
            generator = HeroGenerator.from_config(config.image)
        else:
            print("No image config found, cannot generate hero image")
            sys.exit(1)
    except Exception as e:
        print(f"Config load failed: {e}")
        print("Falling back to legacy mode (deprecated)")
        generator = HeroGenerator()

    result = generator.generate_sync(
        mock_topics,
        args.date,
        Path(args.output_dir),
        args.prompt
    )

    if result:
        print(f"Generated hero image: {result['path']}")
    else:
        print("Failed to generate hero image")
        sys.exit(1)
