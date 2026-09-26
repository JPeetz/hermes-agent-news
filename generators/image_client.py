"""
Unified Image Client Abstraction

Provides mode-based image generation:
- native: Uses google-genai SDK directly for Google Gemini API
- openai-compatible: Uses REST chat/completions format for LiteLLM proxies
- openrouter: Uses OpenRouter's dedicated /api/v1/images endpoint (supports
  aspect_ratio, resolution tiers, and input_references for image-to-image)

Follows the same factory pattern as agents/llm_client.py for consistency.
"""

import asyncio
import io
import base64
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import httpx
from PIL import Image

from google import genai
from google.genai import types, errors

if TYPE_CHECKING:
    from agents.config import ImageProviderConfig

logger = logging.getLogger(__name__)

# Retry policy for transient image-generation failures (connection drops,
# timeouts, 429s, and 5xx). A single dropped connection used to lose the entire
# daily hero image with no retry (e.g. "Server disconnected without sending a
# response" on 2026-06-26). A short 3-attempt/~25s window then proved too
# shallow: on 2026-06-27 the RDSec proxy fast-failed (~3s each) on all 3 tries
# inside ~25s, dropping the hero, yet a manual regen later succeeded first try --
# the provider blip simply outlasted the tiny retry window. So we widen the
# window to span several minutes: more attempts with capped exponential backoff,
# so a multi-minute transient outage self-heals. The per-request timeout (180s)
# is unchanged -- it was never the issue (today's failures were instant
# disconnects, not timeouts).
#
# Backoff schedule (base 3.0, cap 60): ~3, 6, 12, 24, 48, 60 s between the 7
# attempts => ~153s of pure backoff + jitter + request time, i.e. the retry
# window now spans roughly 3 minutes instead of 25 seconds.
DEFAULT_MAX_ATTEMPTS = 7
DEFAULT_RETRY_BASE_DELAY = 3.0  # seconds; exponential: ~3s, 6s, 12s, 24s ...
DEFAULT_RETRY_MAX_DELAY = 60.0  # seconds; cap per-retry backoff so it stays bounded


def _is_retryable_status(status_code: int) -> bool:
    """True for transient HTTP statuses worth retrying (429 + any 5xx)."""
    return status_code == 429 or status_code >= 500


def _backoff_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with jitter for retry attempt N (1-indexed), capped."""
    base = base_delay * (2 ** (attempt - 1))
    base = min(base, max_delay)
    return base + random.uniform(0, base_delay / 2)


async def _post_json_with_retries(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout: float,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    troubleshooting: str = "",
) -> Dict[str, Any]:
    """POST a JSON body and return the parsed response, retrying transient failures.

    Shared by every REST image client so they inherit one retry policy: retries
    on connection drops, timeouts, 429s, and 5xx with capped exponential backoff
    (see the window rationale above); non-transient 4xx fails fast on the first
    attempt. ``troubleshooting`` is appended to the raised RuntimeError so each
    caller can give mode-specific guidance.
    """
    data = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body
                )
                response.raise_for_status()
                data = response.json()
            break  # success

        except httpx.TimeoutException as e:
            if attempt < max_attempts:
                delay = _backoff_delay(attempt, retry_base_delay, retry_max_delay)
                logger.warning(
                    f"Image generation timed out after {timeout}s "
                    f"(attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            error_msg = f"Image generation timed out after {timeout}s ({max_attempts} attempts)"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if _is_retryable_status(status) and attempt < max_attempts:
                delay = _backoff_delay(attempt, retry_base_delay, retry_max_delay)
                logger.warning(
                    f"Image generation API returned status {status} "
                    f"(attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            error_msg = (
                f"Image generation API error (status={status}): "
                f"{e.response.text[:500]}"
            )
            if troubleshooting:
                error_msg += f"\n\n{troubleshooting}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except httpx.RequestError as e:
            # Connection drops / DNS / read errors (e.g. "Server disconnected
            # without sending a response") -- treat as transient.
            if attempt < max_attempts:
                delay = _backoff_delay(attempt, retry_base_delay, retry_max_delay)
                logger.warning(
                    f"Image generation request failed: {e} "
                    f"(attempt {attempt}/{max_attempts}); retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
                continue
            error_msg = f"Image generation request failed: {e}"
            if troubleshooting:
                error_msg += f"\n\n{troubleshooting}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    return data


@dataclass
class ImageResponse:
    """Response from image generation."""
    image_data: bytes  # Raw image bytes
    mime_type: str = "image/png"
    # Token accounting, when the provider reports it.
    #
    # The openai-compatible path goes through /v1/chat/completions, whose schema
    # includes a `usage` block -- but whether a given proxy populates it for an
    # *image* response is a property of that deployment, not of the schema. The
    # native path exposes `usage_metadata` on the SDK response, likewise optional.
    #
    # None means "the provider told us nothing", which is different from zero. The
    # replay renders `n/a` for None and a real figure otherwise; it must never show
    # $0.000 for a call that was in fact billed.
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None


class BaseImageClient(ABC):
    """Abstract base class for image generation clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        reference_image: Optional[bytes] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K"
    ) -> ImageResponse:
        """
        Generate an image from a prompt.

        Args:
            prompt: Text prompt describing the image to generate
            reference_image: Optional reference image bytes for style/content guidance
            aspect_ratio: Image aspect ratio (default "21:9" for hero banners)
            image_size: Image resolution (default "2K")

        Returns:
            ImageResponse with raw image bytes and mime type
        """
        pass


class NativeGeminiClient(BaseImageClient):
    """
    Image client using google-genai SDK (native mode).

    Uses the official Google SDK for direct Gemini API access.
    Recommended for users with Google AI API keys.
    """

    DEFAULT_MODEL = "gemini-3-pro-image-preview"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        timeout: float = 180.0
    ):
        """
        Initialize native Gemini client.

        Args:
            api_key: Google AI API key (explicit, not from env vars)
            model: Model name (default: gemini-3-pro-image-preview)
            timeout: Request timeout in seconds (converted to ms for SDK)
        """
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout

        # Create client with explicit API key and retry options
        # SDK uses milliseconds for timeout
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout * 1000),
            )
        )

        logger.info(f"NativeGeminiClient initialized with model={self.model}, timeout={timeout}s")

    async def generate(
        self,
        prompt: str,
        reference_image: Optional[bytes] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K"
    ) -> ImageResponse:
        """Generate image using google-genai SDK."""
        contents = []

        # Add reference image if provided (must be PIL Image for SDK)
        if reference_image:
            pil_image = Image.open(io.BytesIO(reference_image))
            contents.append(pil_image)

        contents.append(prompt)

        try:
            # Use async client (client.aio)
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )
                )
            )
        except errors.APIError as e:
            error_msg = (
                f"Gemini API error (code={e.code}): {e.message}\n\n"
                f"Troubleshooting (native mode):\n"
                f"- Verify your Google API key has access to {self.model}\n"
                f"- Check quotas at https://console.cloud.google.com/apis/dashboard\n"
                f"- Ensure the model name is correct for your API access level"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # The SDK exposes counts as `usage_metadata`; normalise to the same key
        # names the openai-compatible path yields so downstream code sees one shape.
        usage = None
        meta = getattr(response, "usage_metadata", None)
        if meta is not None:
            usage = {
                k: v
                for k, v in (
                    ("prompt_tokens", getattr(meta, "prompt_token_count", None)),
                    ("completion_tokens", getattr(meta, "candidates_token_count", None)),
                    ("total_tokens", getattr(meta, "total_token_count", None)),
                )
                if v is not None
            } or None
        logger.info(
            f"Image usage reported: {usage}"
            if usage
            else "Image response carried no usage metadata; cost will show as n/a"
        )

        # Extract image from response parts
        for part in response.parts:
            if part.inline_data:
                return ImageResponse(
                    image_data=part.inline_data.data,
                    mime_type=part.inline_data.mime_type or "image/png",
                    usage=usage,
                    model=self.model,
                )

        raise RuntimeError(
            "No image returned from Gemini API. "
            "Check that the model supports image generation and the prompt is valid."
        )


class OpenAICompatibleClient(BaseImageClient):
    """
    Image client using OpenAI chat/completions format (openai-compatible mode).

    Uses REST API for LiteLLM or other OpenAI-compatible proxies.
    Refactored from existing HeroGenerator implementation.
    """

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str,
        timeout: float = 180.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY
    ):
        """
        Initialize OpenAI-compatible client.

        Args:
            api_key: API key for Bearer authentication
            endpoint: API endpoint URL (auto-appends /chat/completions if ends with /v1)
            model: Model name for the proxy
            timeout: Request timeout in seconds
            max_attempts: Total attempts (incl. first) on transient failures
            retry_base_delay: Base seconds for exponential backoff between retries
            retry_max_delay: Cap (seconds) on any single backoff sleep so a long
                retry window stays bounded per-step
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.retry_max_delay = max(0.0, retry_max_delay)

        # Auto-append /chat/completions if endpoint ends with /v1
        if endpoint.rstrip('/').endswith('/v1'):
            self.endpoint = endpoint.rstrip('/') + '/chat/completions'
        else:
            self.endpoint = endpoint

        logger.info(f"OpenAICompatibleClient initialized with endpoint={self.endpoint}, model={self.model}")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for retry attempt N (1-indexed), capped."""
        return _backoff_delay(attempt, self.retry_base_delay, self.retry_max_delay)

    @staticmethod
    def _troubleshooting(endpoint: str, model: str) -> str:
        return (
            f"Troubleshooting (openai-compatible mode):\n"
            f"- Verify your proxy endpoint supports image generation\n"
            f"- Check that the model name '{model}' is correct for your proxy\n"
            f"- Verify the endpoint URL is correct: {endpoint}\n"
            f"- Ensure the API key has proper permissions"
        )

    async def generate(
        self,
        prompt: str,
        reference_image: Optional[bytes] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K"
    ) -> ImageResponse:
        """Generate image using OpenAI chat/completions format."""
        # Build message content
        content = []
        if reference_image:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(reference_image).decode()}"
                }
            })
        content.append({"type": "text", "text": prompt})

        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 1.0,  # Required by Gemini image models
            "modalities": ["image", "text"],
            "image_config": {
                "aspect_ratio": aspect_ratio,
                "image_size": image_size
            }
        }

        # Transient failures (connection drops, timeouts, 429, 5xx) retry with
        # capped exponential backoff via the shared helper; 4xx fails fast.
        data = await _post_json_with_retries(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            body=request_body,
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            retry_base_delay=self.retry_base_delay,
            retry_max_delay=self.retry_max_delay,
            troubleshooting=self._troubleshooting(self.endpoint, self.model),
        )

        # Extract image from response
        message = data.get("choices", [{}])[0].get("message", {})
        images = message.get("images", [])

        if not images:
            error_content = message.get("content", "Unknown error - no images returned")
            raise RuntimeError(f"No image returned from API: {error_content}")

        image_url = images[0].get("image_url", {}).get("url", "")
        if not image_url or "," not in image_url:
            raise RuntimeError("Invalid image URL format in response - expected base64 data URL")

        # Parse base64 data URL (format: data:image/png;base64,<data>)
        image_base64 = image_url.split(",", 1)[1]

        # `usage` is optional in practice: the schema defines it, but whether this
        # proxy fills it in for an image response is a deployment detail. Log what
        # we got (keys only -- the values are counts, but the log is not the place
        # to grow) so a missing block is diagnosable rather than silent.
        usage = data.get("usage")
        if usage:
            logger.info(f"Image usage reported: {usage}")
        else:
            logger.info("Image response carried no usage block; cost will show as n/a")

        return ImageResponse(
            image_data=base64.b64decode(image_base64),
            mime_type="image/png",
            usage=usage if isinstance(usage, dict) else None,
            model=data.get("model") or self.model,
        )


class OpenRouterImageClient(BaseImageClient):
    """
    Image client using OpenRouter's dedicated /api/v1/images endpoint.

    Chosen over OpenRouter's chat/completions image path because /images
    natively supports what the daily hero needs:
    - aspect_ratio ("21:9" hero banners; chat/completions would ignore it)
    - resolution tiers ("2K")
    - input_references for image-to-image (the Agent N character reference)

    Response shape per OpenRouter docs: data[0].b64_json + media_type, plus a
    usage block with prompt/completion token counts.
    """

    DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemini-3-pro-image"

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        quality: Optional[str] = None,
        timeout: float = 180.0,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY
    ):
        """
        Initialize OpenRouter image client.

        Args:
            api_key: OpenRouter API key for Bearer authentication
            endpoint: API base URL ending at the version segment
                (default https://openrouter.ai/api/v1); /images is appended
            model: OpenRouter model slug (default google/gemini-3-pro-image)
            quality: Optional rendering quality (auto/low/medium/high)
            timeout: Request timeout in seconds
            max_attempts/retry_*: Shared transient-failure retry policy
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.quality = quality or None
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.retry_max_delay = max(0.0, retry_max_delay)
        # Normalize once so generate() can just append the resource path.
        self.endpoint = (endpoint or self.DEFAULT_ENDPOINT).rstrip('/')

        logger.info(
            f"OpenRouterImageClient initialized with endpoint={self.endpoint}/images, "
            f"model={self.model}, quality={self.quality or 'default'}"
        )

    async def generate(
        self,
        prompt: str,
        reference_image: Optional[bytes] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K"
    ) -> ImageResponse:
        """Generate image via POST {endpoint}/images."""
        body: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "resolution": image_size,
        }
        if self.quality:
            body["quality"] = self.quality
        if reference_image:
            body["input_references"] = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(reference_image).decode()}"
                }
            }]

        data = await _post_json_with_retries(
            f"{self.endpoint}/images",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            body=body,
            timeout=self.timeout,
            max_attempts=self.max_attempts,
            retry_base_delay=self.retry_base_delay,
            retry_max_delay=self.retry_max_delay,
            troubleshooting=(
                f"Troubleshooting (openrouter mode):\n"
                f"- Check that '{self.model}' supports image output on OpenRouter\n"
                f"- Verify the API key has credit available\n"
                f"- Verify the endpoint URL is correct: {self.endpoint}/images"
            ),
        )

        images = data.get("data") if isinstance(data, dict) else None
        if not images:
            error_content = data.get("error", "Unknown error - no images returned") if isinstance(data, dict) else str(data)[:500]
            raise RuntimeError(f"No image returned from OpenRouter API: {error_content}")

        first = images[0]
        b64 = first.get("b64_json", "")
        if not b64:
            raise RuntimeError("No b64_json in OpenRouter image response")

        usage = data.get("usage")
        if usage:
            logger.info(f"Image usage reported: {usage}")
        else:
            logger.info("Image response carried no usage block; cost will show as n/a")

        return ImageResponse(
            image_data=base64.b64decode(b64),
            mime_type=first.get("media_type") or "image/png",
            usage=usage if isinstance(usage, dict) else None,
            model=data.get("model") or self.model,
        )


class KieImageClient(BaseImageClient):
    """
    Image client using kie.ai task-based API (gpt-image/1.5-image-to-image).

    kie.ai uses an async task model: createTask -> poll recordInfo -> download result.
    Reference image is passed as input_urls (URL reference, not local bytes).
    Supports: gpt-image/1.5-image-to-image with character sheet reference.
    """

    KIE_BASE = "https://api.kie.ai/api/v1"
    DEFAULT_MODEL = "gpt-image/1.5-image-to-image"
    # Supported aspect ratios: "1:1", "3:2", "4:3", "16:9" (NOT 21:9)

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        reference_url: Optional[str] = None,
        quality: Optional[str] = None,
        timeout: float = 180.0,
        max_poll_attempts: int = 180,
        poll_interval: float = 15.0,
    ):
        self.api_key = api_key
        # Fallback: if the env-var-resolved key looks wrong (<10 chars),
        # try the GitHub Actions direct-write fallback file.
        if len(self.api_key) < 10:
            key_file = "/tmp/kie.key"
            try:
                with open(key_file) as f:
                    fallback = f.read().strip()
                    if len(fallback) >= 20:
                        self.api_key = fallback
                        logger.info(f"KieImageClient: using key from {key_file} ({len(fallback)} chars)")
            except (FileNotFoundError, OSError):
                pass
        self.model = model or self.DEFAULT_MODEL
        self.reference_url = reference_url
        self.quality = quality or "medium"
        self.timeout = timeout
        self.max_poll_attempts = max_poll_attempts
        self.poll_interval = max(3.0, poll_interval)

        logger.info(
            f"KieImageClient initialized with model={self.model}, "
            f"reference_url={'set' if reference_url else 'none'}, "
            f"quality={self.quality}, "
            f"api_key_set={'yes' if self.api_key else 'no'}, "
            f"api_key_len={len(self.api_key) if self.api_key else 0}"
        )

    async def generate(
        self,
        prompt: str,
        reference_image: Optional[bytes] = None,
        aspect_ratio: str = "16:9",
        image_size: str = "2K"
    ) -> ImageResponse:
        """Generate image via kie.ai task API."""
        import json, asyncio, urllib.request, urllib.error, urllib.parse

        # Build request body. Skip image_size: kie uses aspect_ratio only.
        # gpt-image/1.5-image-to-image does NOT accept output_format or strength.
        # CRITICAL: `input` must be a stringified JSON object, NOT a nested dict.
        input_dict = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "quality": self.quality,
        }
        # Add character sheet reference if configured
        if self.reference_url:
            input_dict["input_urls"] = [self.reference_url]
        elif reference_image:
            # Fall back to base64 data URI (handles legacy caller passing bytes)
            import base64
            b64 = base64.b64encode(reference_image).decode()
            input_dict["input_urls"] = [f"data:image/png;base64,{b64}"]

        body = {
            "model": self.model,
            "input": json.dumps(input_dict)
        }

        logger.info(f"Kie: creating task for {self.model}, aspect={aspect_ratio}")

        try:
            # Step 1: Create task
            req_body = json.dumps(body).encode()
            req = urllib.request.Request(
                f"{self.KIE_BASE}/jobs/createTask",
                data=req_body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            response = urllib.request.urlopen(req, timeout=self.timeout)
            raw_response = response.read().decode()
            data = json.loads(raw_response)
            task_id = data.get("data", {}).get("taskId") if isinstance(data.get("data"), dict) else None

            if not task_id:
                raise RuntimeError(f"Kie: no taskId in response ({len(raw_response)} chars): {raw_response[:500]}")

            # Step 2: Poll for result
            for i in range(self.max_poll_attempts):
                await asyncio.sleep(self.poll_interval)
                poll_req = urllib.request.Request(
                    f"{self.KIE_BASE}/jobs/recordInfo?taskId={task_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                poll_resp = urllib.request.urlopen(poll_req, timeout=self.timeout)
                poll_data = json.loads(poll_resp.read().decode())
                pd = poll_data.get("data", {})
                state = pd.get("state", "") if isinstance(pd, dict) else ""

                if state == "success":
                    result_json = pd.get("resultJson", "{}")
                    urls = json.loads(result_json).get("resultUrls", [])
                    if not urls:
                        raise RuntimeError("Kie: success but no resultUrls in response")
                    img_url = urls[0]
                    logger.info(f"Kie: task {task_id} succeeded, downloading {img_url}")
                    img_resp = urllib.request.urlopen(img_url, timeout=self.timeout)
                    image_bytes = img_resp.read()
                    return ImageResponse(
                        image_data=image_bytes,
                        mime_type="image/webp",
                        usage=None,
                        model=self.model,
                    )
                elif state in ("fail", "error"):
                    fail_msg = pd.get("failMsg", "unknown failure") if isinstance(pd, dict) else "unknown"
                    raise RuntimeError(f"Kie: task {task_id} failed: {fail_msg[:300]}")
                else:
                    logger.info(f"Kie: poll {i+1}/{self.max_poll_attempts} state={state}")

            raise RuntimeError(f"Kie: task {task_id} did not complete within {self.max_poll_attempts} polls")

        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Kie: HTTP {e.code}: {e.read().decode()[:300]}") from e


class ImageClient:
    """
    Factory class for creating image clients based on configuration.

    Usage:
        client = ImageClient.from_config(config)
        response = await client.generate(prompt, reference_image)
    """

    @classmethod
    def from_config(cls, config: 'ImageProviderConfig') -> BaseImageClient:
        """
        Create appropriate image client based on config mode.

        Args:
            config: ImageProviderConfig with mode, api_key, endpoint, model

        Returns:
            NativeGeminiClient for native mode
            OpenAICompatibleClient for openai-compatible mode
            OpenRouterImageClient for openrouter mode
            KieImageClient for kie mode

        Raises:
            ValueError: If mode is unknown
        """
        if config.mode == "native":
            return NativeGeminiClient(
                api_key=config.api_key,
                model=config.model
            )
        elif config.mode == "openai-compatible":
            return OpenAICompatibleClient(
                api_key=config.api_key,
                endpoint=config.endpoint,  # Already validated by schema
                model=config.model
            )
        elif config.mode == "openrouter":
            return OpenRouterImageClient(
                api_key=config.api_key,
                endpoint=config.endpoint,  # Optional; defaults to openrouter.ai/api/v1
                model=config.model,
                quality=getattr(config, 'quality', None)
            )
        elif config.mode == "kie":
            return KieImageClient(
                api_key=config.api_key,
                model=config.model,
                reference_url=config.reference_url,
                quality=getattr(config, 'quality', None)
            )
        else:
            raise ValueError(
                f"Unknown image mode: {config.mode}. "
                f"Expected 'native', 'openai-compatible', 'openrouter', or 'kie'."
            )
