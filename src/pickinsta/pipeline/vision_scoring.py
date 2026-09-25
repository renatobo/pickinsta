"""Batch coordination for CLIP, Claude, and Ollama scoring."""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional
from urllib.request import Request

from pickinsta.config import DEFAULT_ACCOUNT_CONTEXT, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL
from pickinsta.events import console_event
from pickinsta.models import ImageScore
from pickinsta.vision.schema import normalize_vision_payload

print = partial(console_event, "vision_scoring")


@dataclass(frozen=True, slots=True)
class FacadeCollaborators:
    """Live selector dependencies retained during the compatibility window."""

    resolve_ollama_timeout_seconds: Callable[..., float]
    resolve_ollama_max_image_edge: Callable[..., int]
    resolve_ollama_jpeg_quality: Callable[..., int]
    resolve_ollama_use_yolo_context: Callable[..., bool]
    resolve_ollama_concurrency: Callable[..., int]
    resolve_ollama_max_retries: Callable[..., int]
    resolve_ollama_retry_backoff_seconds: Callable[..., float]
    resolve_ollama_circuit_breaker_errors: Callable[..., int]
    build_vision_prompt: Callable[..., str]
    claude_prompt_sha256: Callable[..., str]
    resolve_optional_hf_token: Callable[..., Any]
    load_clip_model: Callable[..., Any]
    _clip_setup_hint: Callable[..., str]
    resolve_anthropic_api_key: Callable[..., str]
    resolve_claude_model: Callable[..., str]
    _claude_model_candidates: Callable[..., list[str]]
    _is_model_not_found_error: Callable[..., bool]
    resolve_account_context: Callable[..., str]
    _claude_setup_hint: Callable[..., str]
    resolve_ollama_base_url: Callable[..., str]
    resolve_ollama_model: Callable[..., str]
    resolve_ollama_keep_alive: Callable[..., str]
    _ollama_setup_hint: Callable[..., str]
    urlopen: Callable[..., Any]
    score_with_ollama: Callable[..., dict]
    _is_retryable_ollama_error: Callable[..., bool]
    _claude_crop_gate_multiplier: Callable[..., float]
    _file_sha256: Callable[..., str]
    claude_cache_options: Callable[..., dict]
    load_claude_score_from_file_cache: Callable[..., Any]
    score_with_claude: Callable[..., dict]
    save_claude_score_to_file_cache: Callable[..., Any]
    score_with_clip: Callable[..., dict]

    @classmethod
    def from_module(cls, module: ModuleType) -> "FacadeCollaborators":
        values = vars(module)
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float conversion with default fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _claude_crop_gate_multiplier(vision: dict) -> float:
    """Strongly gate ranking by Claude crop_4x5 confidence."""
    crop_score = _safe_float(vision.get("crop_4x5"), default=0.0)
    if crop_score <= 4.0:
        return 0.15
    if crop_score <= 5.0:
        return 0.35
    if crop_score <= 6.0:
        return 0.60
    if crop_score <= 7.0:
        return 0.80
    return 1.0


def batch_vision_score(
    candidates: list[ImageScore],
    scorer: str = "clip",
    env_search_dir: Optional[Path] = None,
    claude_model: Optional[str] = None,
    rescore: bool = False,
    cache_observer=None,
    preloaded_vision_cache: Optional[dict[Path, Optional[dict]]] = None,
    failure_observer=None,
    collaborators=None,
) -> list[ImageScore]:
    """Run vision scoring on candidate images using injected compatibility collaborators."""
    if collaborators is None:
        raise TypeError("collaborators are required")
    if not candidates:
        return []
    resolve_ollama_timeout_seconds = collaborators.resolve_ollama_timeout_seconds
    resolve_ollama_max_image_edge = collaborators.resolve_ollama_max_image_edge
    resolve_ollama_jpeg_quality = collaborators.resolve_ollama_jpeg_quality
    resolve_ollama_use_yolo_context = collaborators.resolve_ollama_use_yolo_context
    resolve_ollama_concurrency = collaborators.resolve_ollama_concurrency
    resolve_ollama_max_retries = collaborators.resolve_ollama_max_retries
    resolve_ollama_retry_backoff_seconds = collaborators.resolve_ollama_retry_backoff_seconds
    resolve_ollama_circuit_breaker_errors = collaborators.resolve_ollama_circuit_breaker_errors
    build_vision_prompt = collaborators.build_vision_prompt
    claude_prompt_sha256 = collaborators.claude_prompt_sha256
    resolve_optional_hf_token = collaborators.resolve_optional_hf_token
    load_clip_model = collaborators.load_clip_model
    _clip_setup_hint = collaborators._clip_setup_hint
    resolve_anthropic_api_key = collaborators.resolve_anthropic_api_key
    resolve_claude_model = collaborators.resolve_claude_model
    _claude_model_candidates = collaborators._claude_model_candidates
    _is_model_not_found_error = collaborators._is_model_not_found_error
    resolve_account_context = collaborators.resolve_account_context
    _claude_setup_hint = collaborators._claude_setup_hint
    resolve_ollama_base_url = collaborators.resolve_ollama_base_url
    resolve_ollama_model = collaborators.resolve_ollama_model
    resolve_ollama_keep_alive = collaborators.resolve_ollama_keep_alive
    _ollama_setup_hint = collaborators._ollama_setup_hint
    urlopen = collaborators.urlopen
    score_with_ollama = collaborators.score_with_ollama
    _is_retryable_ollama_error = collaborators._is_retryable_ollama_error
    _claude_crop_gate_multiplier = collaborators._claude_crop_gate_multiplier
    _file_sha256 = collaborators._file_sha256
    claude_cache_options = collaborators.claude_cache_options
    load_claude_score_from_file_cache = collaborators.load_claude_score_from_file_cache
    score_with_claude = collaborators.score_with_claude
    save_claude_score_to_file_cache = collaborators.save_claude_score_to_file_cache
    score_with_clip = collaborators.score_with_clip

    clip_model, clip_processor = None, None
    claude_api_key = None
    claude_client = None
    claude_model_used = None
    ollama_model_used = None
    ollama_base_url = None
    ollama_timeout_seconds = resolve_ollama_timeout_seconds()
    ollama_max_image_edge = resolve_ollama_max_image_edge()
    ollama_jpeg_quality = resolve_ollama_jpeg_quality()
    ollama_keep_alive = "10m"
    ollama_use_yolo_context = resolve_ollama_use_yolo_context()
    ollama_concurrency = resolve_ollama_concurrency()
    ollama_max_retries = resolve_ollama_max_retries()
    ollama_retry_backoff = resolve_ollama_retry_backoff_seconds()
    ollama_circuit_breaker_errors = resolve_ollama_circuit_breaker_errors()
    claude_base_prompt = build_vision_prompt(DEFAULT_ACCOUNT_CONTEXT)
    claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
    claude_cache_hits = 0
    claude_api_calls = 0
    if scorer == "clip":
        resolve_optional_hf_token(search_dir=env_search_dir)
        print("  Loading CLIP model (first run downloads ~1.7GB)...")
        try:
            clip_model, clip_processor = load_clip_model()
        except Exception as e:
            print(f"  ⚠ CLIP unavailable: {e}")
            print(f"  💡 {_clip_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "CLIP unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates
    elif scorer == "claude":
        claude_api_key = resolve_anthropic_api_key(search_dir=env_search_dir)
        try:
            import anthropic

            claude_client = anthropic.Anthropic(api_key=claude_api_key)

            preferred_model = resolve_claude_model(cli_model=claude_model)
            last_error = None
            for candidate in _claude_model_candidates(preferred_model):
                try:
                    # Preflight once to avoid repeated 404s per image.
                    claude_client.messages.create(
                        model=candidate,
                        max_tokens=1,
                        messages=[{"role": "user", "content": "ok"}],
                    )
                    claude_model_used = candidate
                    if candidate != preferred_model:
                        print(f"  ↪ Claude model fallback: {preferred_model} -> {candidate}")
                    break
                except Exception as e:
                    last_error = e
                    if _is_model_not_found_error(e):
                        continue
                    raise

            if claude_model_used is None:
                raise RuntimeError(
                    f"No available Claude model found. Tried: {', '.join(_claude_model_candidates(preferred_model))}. "
                    f"Last error: {last_error}"
                )

            account_context = resolve_account_context(search_dir=env_search_dir)
            claude_base_prompt = build_vision_prompt(account_context)
            claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
        except Exception as e:
            print(f"  ⚠ Claude unavailable: {e}")
            print(f"  💡 {_claude_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "Claude unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates
    elif scorer == "ollama":
        try:
            ollama_base_url = resolve_ollama_base_url(search_dir=env_search_dir)
            ollama_model_used = resolve_ollama_model(search_dir=env_search_dir)
            ollama_keep_alive = resolve_ollama_keep_alive(search_dir=env_search_dir)
            request = Request(
                f"{ollama_base_url.rstrip('/')}/api/tags",
                headers={"Accept": "application/json"},
                method="GET",
            )
            with urlopen(request, timeout=min(60, ollama_timeout_seconds)):
                pass

            account_context = resolve_account_context(search_dir=env_search_dir)
            claude_base_prompt = build_vision_prompt(account_context)
            claude_prompt_hash = claude_prompt_sha256(claude_base_prompt)
        except Exception as e:
            print(f"  ⚠ Ollama unavailable: {e}")
            print(f"  💡 {_ollama_setup_hint(e)}")
            print("  ↪ Falling back to technical-only ranking for this run.")
            for item in candidates:
                item.final_score = item.technical.get("composite", 0.0)
                item.one_line = "Ollama unavailable — ranked by technical score only"
            candidates.sort(key=lambda x: x.final_score, reverse=True)
            return candidates

    if scorer == "ollama":
        progress_write = print
        progress_bar = None
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=len(candidates),
                desc="  Ollama scoring",
                unit="img",
            )
            progress_write = tqdm.write
        except Exception as e:
            print(f"  ⚠ Progress bar unavailable: {e}")

        def finalize_item(item: ImageScore, vision: dict) -> None:
            vision_normalized = vision.get("total", 30) / 60.0
            base_score = item.technical["composite"] * 0.3 + vision_normalized * 0.7
            gate = _claude_crop_gate_multiplier(vision)
            item.vision = vision
            item.final_score = base_score * gate
            item.one_line = vision.get("one_line", "")

        def mark_failed(item: ImageScore, message: str) -> None:
            item.final_score = item.technical["composite"] * 0.3
            item.one_line = message

        def score_one_with_retry(item: ImageScore) -> dict:
            last_error = None
            for attempt in range(ollama_max_retries + 1):
                try:
                    return score_with_ollama(
                        item.path,
                        base_url=ollama_base_url or DEFAULT_OLLAMA_BASE_URL,
                        model=ollama_model_used or DEFAULT_OLLAMA_MODEL,
                        use_yolo_context=ollama_use_yolo_context,
                        prompt=claude_base_prompt,
                        timeout_seconds=ollama_timeout_seconds,
                        max_image_edge=ollama_max_image_edge,
                        jpeg_quality=ollama_jpeg_quality,
                        keep_alive=ollama_keep_alive,
                    )
                except Exception as e:
                    last_error = e
                    if attempt >= ollama_max_retries or not _is_retryable_ollama_error(e):
                        break
                    sleep_seconds = ollama_retry_backoff * (2**attempt)
                    time.sleep(sleep_seconds)
            raise RuntimeError(
                f"Ollama scoring failed after {ollama_max_retries + 1} attempt(s): {last_error}"
            ) from last_error

        scored = 0
        failed = 0
        consecutive_failures = 0
        stop_submissions = False
        next_index = 0
        total_items = len(candidates)
        pending: dict = {}

        with ThreadPoolExecutor(max_workers=ollama_concurrency) as executor:
            while next_index < total_items and len(pending) < ollama_concurrency:
                future = executor.submit(score_one_with_retry, candidates[next_index])
                pending[future] = next_index
                next_index += 1

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    item = candidates[idx]
                    try:
                        vision = future.result()
                        finalize_item(item, vision)
                        scored += 1
                        consecutive_failures = 0
                    except Exception as e:
                        progress_write(f"  ⚠ Vision score failed for {item.path.name}: {e}")
                        mark_failed(item, "Vision scoring failed — ranked by technical score only")
                        if failure_observer is not None:
                            failure_observer(item.path, scorer, str(e))
                        failed += 1
                        consecutive_failures += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

                while (
                    not stop_submissions
                    and next_index < total_items
                    and len(pending) < ollama_concurrency
                ):
                    if consecutive_failures >= ollama_circuit_breaker_errors:
                        stop_submissions = True
                        progress_write(
                            "  ⚠ Ollama circuit breaker opened due to consecutive failures; "
                            "remaining images will use technical-only fallback."
                        )
                        break
                    future = executor.submit(score_one_with_retry, candidates[next_index])
                    pending[future] = next_index
                    next_index += 1

        if stop_submissions and next_index < total_items:
            remaining = candidates[next_index:]
            for item in remaining:
                mark_failed(item, "Ollama circuit breaker active — ranked by technical score only")
                if failure_observer is not None:
                    failure_observer(item.path, scorer, "Ollama circuit breaker active")
            failed += len(remaining)
            if progress_bar is not None:
                progress_bar.update(len(remaining))

        if progress_bar is not None:
            progress_bar.close()

        yolo_label = "on" if ollama_use_yolo_context else "off"
        print(f"  🖥️  Ollama server: {ollama_base_url} | model: {ollama_model_used}")
        print(
            "  ⚙️  Ollama tuning: "
            f"timeout={ollama_timeout_seconds}s, max_edge={ollama_max_image_edge}px, "
            f"jpeg_quality={ollama_jpeg_quality}, keep_alive={ollama_keep_alive}, yolo={yolo_label}"
        )
        print(
            "  🔁 Ollama resilience: "
            f"concurrency={ollama_concurrency}, retries={ollama_max_retries}, "
            f"retry_backoff={ollama_retry_backoff:.2f}s, "
            f"circuit_breaker_errors={ollama_circuit_breaker_errors}"
        )
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        print(f"  ✅ Vision scoring: {scored} scored, {failed} failed")
        return candidates

    scored = 0
    failed = 0
    active_claude_model = claude_model_used or resolve_claude_model(cli_model=claude_model)

    if scorer == "claude":
        # --- Adaptive concurrent Claude scoring ---
        CLAUDE_INITIAL_CONCURRENCY = 3
        CLAUDE_MAX_CONCURRENCY = 8
        CLAUDE_MIN_CONCURRENCY = 1
        CLAUDE_MAX_RETRIES = 3
        CLAUDE_RETRY_BASE_SEC = 1.0

        concurrency = CLAUDE_INITIAL_CONCURRENCY
        rate_limit_hits = 0
        progress_bar = None
        progress_write = print
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=len(candidates),
                desc="  Claude scoring",
                unit="img",
            )
            progress_write = tqdm.write
        except Exception:
            pass

        def _is_rate_limit(e: Exception) -> bool:
            text = str(e).lower()
            return "429" in text or "rate" in text or "overloaded" in text or "529" in text

        def _claude_score_one(
            item: ImageScore,
        ) -> tuple[ImageScore, Optional[dict], Optional[Exception], bool]:
            """Score one image with Claude. Returns (item, vision_dict, error, was_cached)."""
            source_for_cache = item.source_path or item.path
            source_sha = _file_sha256(source_for_cache)
            cache_options = claude_cache_options(
                score_path=item.path,
                source_path=source_for_cache,
            )
            if not rescore:
                if (
                    preloaded_vision_cache is not None
                    and source_for_cache in preloaded_vision_cache
                ):
                    cached = preloaded_vision_cache[source_for_cache]
                else:
                    cached = load_claude_score_from_file_cache(
                        source_path=source_for_cache,
                        source_sha256=source_sha,
                        model=active_claude_model,
                        prompt_sha256=claude_prompt_hash,
                        scorer="claude",
                        scoring_options=cache_options,
                        strict_model=True,
                    )
                if cached is not None:
                    try:
                        cached = normalize_vision_payload(cached)
                    except ValueError:
                        cached = None
                    if cached is not None:
                        if cache_observer is not None:
                            cache_observer(True)
                        return (item, cached, None, True)
                if cache_observer is not None:
                    cache_observer(False)
            try:
                vision = score_with_claude(
                    item.path,
                    api_key=claude_api_key,
                    model=active_claude_model,
                    client=claude_client,
                    use_yolo_context=True,
                    prompt=claude_base_prompt,
                )
                vision = normalize_vision_payload(vision)
                try:
                    save_claude_score_to_file_cache(
                        source_path=source_for_cache,
                        source_sha256=source_sha,
                        model=active_claude_model,
                        prompt_sha256=claude_prompt_hash,
                        scorer="claude",
                        scoring_options=cache_options,
                        vision=vision,
                    )
                except Exception:
                    pass
                return (item, vision, None, False)
            except Exception as e:
                return (item, None, e, False)

        def _finalize(item: ImageScore, vision: dict) -> None:
            item.vision = vision
            vision_normalized = vision.get("total", 30) / 60.0
            base_score = item.technical["composite"] * 0.3 + vision_normalized * 0.7
            gate = _claude_crop_gate_multiplier(vision)
            item.final_score = base_score * gate
            item.one_line = vision.get("one_line", "")

        pending: dict = {}
        next_idx = 0
        retry_queue: list[tuple[ImageScore, int]] = []  # (item, attempt)

        with ThreadPoolExecutor(max_workers=CLAUDE_MAX_CONCURRENCY) as executor:
            # Fill initial batch
            while next_idx < len(candidates) and len(pending) < concurrency:
                future = executor.submit(_claude_score_one, candidates[next_idx])
                pending[future] = (candidates[next_idx], 0)
                next_idx += 1

            while pending or retry_queue:
                # Submit retries if we have capacity
                while retry_queue and len(pending) < concurrency:
                    retry_item, attempt = retry_queue.pop(0)
                    future = executor.submit(_claude_score_one, retry_item)
                    pending[future] = (retry_item, attempt)

                if not pending:
                    break

                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    item, attempt = pending.pop(future)
                    result_item, vision, error, was_cached = future.result()

                    if vision is not None:
                        _finalize(result_item, vision)
                        if was_cached:
                            claude_cache_hits += 1
                        else:
                            claude_api_calls += 1
                        scored += 1
                        # Scale up on success (slowly)
                        if concurrency < CLAUDE_MAX_CONCURRENCY and claude_api_calls % 5 == 0:
                            concurrency = min(concurrency + 1, CLAUDE_MAX_CONCURRENCY)
                    elif error is not None:
                        if _is_rate_limit(error) and attempt < CLAUDE_MAX_RETRIES:
                            rate_limit_hits += 1
                            # Back off: reduce concurrency and retry with delay
                            concurrency = max(CLAUDE_MIN_CONCURRENCY, concurrency - 1)
                            backoff = CLAUDE_RETRY_BASE_SEC * (2**attempt)
                            progress_write(
                                f"  ⏳ Rate limited, backing off {backoff:.1f}s "
                                f"(concurrency → {concurrency})"
                            )
                            time.sleep(backoff)
                            retry_queue.append((result_item, attempt + 1))
                        elif attempt < CLAUDE_MAX_RETRIES and "timeout" in str(error).lower():
                            retry_queue.append((result_item, attempt + 1))
                        else:
                            progress_write(
                                f"  ⚠ Vision score failed for {result_item.path.name}: {error}"
                            )
                            result_item.final_score = result_item.technical["composite"] * 0.3
                            result_item.one_line = (
                                "Vision scoring failed — ranked by technical score only"
                            )
                            if failure_observer is not None:
                                failure_observer(result_item.path, scorer, str(error))
                            failed += 1

                    if progress_bar is not None:
                        progress_bar.update(1)

                    # Submit next items up to current concurrency
                    while next_idx < len(candidates) and len(pending) < concurrency:
                        future = executor.submit(_claude_score_one, candidates[next_idx])
                        pending[future] = (candidates[next_idx], 0)
                        next_idx += 1

        if progress_bar is not None:
            progress_bar.close()
        print(
            f"  📦 Claude: {claude_cache_hits} cached, {claude_api_calls} API calls, "
            f"{failed} failed, {rate_limit_hits} rate limits, concurrency peak {concurrency}"
        )

    else:
        # CLIP scoring (sequential, local)
        progress_write = print
        score_iter = candidates
        try:
            from tqdm.auto import tqdm

            score_iter = tqdm(
                candidates,
                total=len(candidates),
                desc=f"  {scorer.capitalize()} scoring",
                unit="img",
            )
            progress_write = tqdm.write
        except Exception:
            pass

        for item in score_iter:
            try:
                item.vision = normalize_vision_payload(
                    score_with_clip(item.path, clip_model, clip_processor)
                )
                vision_normalized = item.vision.get("total", 30) / 60.0
                item.final_score = item.technical["composite"] * 0.3 + vision_normalized * 0.7
                item.one_line = item.vision.get("one_line", "")
                scored += 1
            except Exception as e:
                progress_write(f"  ⚠ Vision score failed for {item.path.name}: {e}")
                item.final_score = item.technical["composite"] * 0.3
                item.one_line = "Vision scoring failed — ranked by technical score only"
                if failure_observer is not None:
                    failure_observer(item.path, scorer, str(e))
                failed += 1

        if hasattr(score_iter, "close"):
            score_iter.close()

    candidates.sort(key=lambda x: x.final_score, reverse=True)
    print(f"  ✅ Vision scoring: {scored} scored, {failed} failed")
    return candidates
