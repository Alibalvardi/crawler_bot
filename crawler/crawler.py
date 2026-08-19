from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import ssl

import httpx
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

USER_AGENT = "CrawlerBot/0.1 (academic project; polite crawler)"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
SKIP_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".zip",
    ".rar",
    ".7z",
    ".mp3",
    ".mp4",
    ".avi",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".xml",
    ".json",
    ".csv",
}


@dataclass
class Page:
    url: str
    status_code: int
    title: str
    text: str
    depth: int


@dataclass
class FailedUrl:
    url: str
    depth: int
    reason: str
    status_code: int | None = None


@dataclass
class CrawlResult:
    seed_url: str
    crawled_at: str
    pages: list[Page] = field(default_factory=list)
    errors: list[FailedUrl] = field(default_factory=list)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seed_url": self.seed_url,
            "crawled_at": self.crawled_at,
            "page_count": len(self.pages),
            "pages": [asdict(page) for page in self.pages],
            "errors": [asdict(item) for item in self.errors],
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output


def crawl(
    start_url: str | None = None,
    *,
    max_pages: int = 500,
    max_depth: int = 5,
    concurrency: int = 16,
    delay: float = 0.0,
    timeout: float = 30.0,
    retries: int = 2,
    verify_ssl: bool = True,
    resume_from: str | Path | None = None,
) -> CrawlResult:
    return asyncio.run(
        acrawl(
            start_url,
            max_pages=max_pages,
            max_depth=max_depth,
            concurrency=concurrency,
            delay=delay,
            timeout=timeout,
            retries=retries,
            verify_ssl=verify_ssl,
            resume_from=resume_from,
        )
    )


async def acrawl(
    start_url: str | None = None,
    *,
    max_pages: int = 500,
    max_depth: int = 5,
    concurrency: int = 16,
    delay: float = 0.0,
    timeout: float = 30.0,
    retries: int = 2,
    verify_ssl: bool = True,
    resume_from: str | Path | None = None,
) -> CrawlResult:
    resume_data = _load_crawl_file(resume_from) if resume_from else None
    if resume_data:
        seed = resume_data.seed_url
        result = resume_data
        result.crawled_at = datetime.now(timezone.utc).isoformat()
        initial = [(item.url, item.depth) for item in result.errors]
        result.errors = []
        seen = {page.url for page in result.pages}
        seen.update(url for url, _ in initial)
    else:
        if not start_url:
            raise ValueError("website url is empty")
        seed = to_absolute_url(start_url)
        result = CrawlResult(seed_url=seed, crawled_at=datetime.now(timezone.utc).isoformat())
        initial = [(seed, 0)]
        seen = {seed}

    robots = await asyncio.to_thread(load_robots, seed, timeout, verify_ssl)
    workers = max(1, concurrency)
    timeout_config = httpx.Timeout(timeout, connect=min(20.0, timeout))
    limits = httpx.Limits(max_connections=workers, max_keepalive_connections=workers)
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

    async with httpx.AsyncClient(
        headers=headers,
        timeout=timeout_config,
        follow_redirects=True,
        limits=limits,
        http2=False,
        verify=verify_ssl,
    ) as client:
        state = _CrawlState(seen=seen, in_flight=0)
        await _run_workers(
            client=client,
            jobs=initial,
            state=state,
            result=result,
            seed=seed,
            robots=robots,
            max_pages=max_pages,
            max_depth=max_depth,
            delay=delay,
            workers=workers,
            verify_ssl=verify_ssl,
        )
        for round_no in range(1, max(0, retries) + 1):
            pending = list(result.errors)
            if not pending or len(result.pages) >= max_pages:
                break
            result.errors = []
            logger.info("retry round %s: %s urls", round_no, len(pending))
            await asyncio.sleep(1)
            await _run_workers(
                client=client,
                jobs=[(item.url, item.depth) for item in pending],
                state=state,
                result=result,
                seed=seed,
                robots=robots,
                max_pages=max_pages,
                max_depth=max_depth,
                delay=max(delay, 0.2),
                workers=max(1, min(8, workers)),
                verify_ssl=verify_ssl,
            )

    return result


async def _run_workers(
    *,
    client: httpx.AsyncClient,
    jobs: list[tuple[str, int]],
    state: _CrawlState,
    result: CrawlResult,
    seed: str,
    robots: RobotFileParser | None,
    max_pages: int,
    max_depth: int,
    delay: float,
    workers: int,
    verify_ssl: bool = True,
) -> None:
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    if queue.empty():
        return
    tasks = [
        asyncio.create_task(
            _worker(
                client=client,
                queue=queue,
                state=state,
                result=result,
                seed=seed,
                robots=robots,
                max_pages=max_pages,
                max_depth=max_depth,
                delay=delay,
                verify_ssl=verify_ssl,
            )
        )
        for _ in range(workers)
    ]
    await asyncio.gather(*tasks)


@dataclass
class _CrawlState:
    seen: set[str]
    in_flight: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def _worker(
    *,
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[str, int]],
    state: _CrawlState,
    result: CrawlResult,
    seed: str,
    robots: RobotFileParser | None,
    max_pages: int,
    max_depth: int,
    delay: float,
    verify_ssl: bool = True,
) -> None:
    while True:
        async with state.lock:
            if len(result.pages) >= max_pages:
                return

        try:
            url, depth = await asyncio.wait_for(queue.get(), timeout=0.3)
        except TimeoutError:
            async with state.lock:
                if state.in_flight == 0 and queue.empty():
                    return
            continue

        async with state.lock:
            if len(result.pages) >= max_pages:
                return
            state.in_flight += 1
        try:
            if not allowed_by_robots(robots, url):
                logger.info("skipped by robots.txt: %s", url)
                continue
            if delay > 0:
                await asyncio.sleep(delay)
            await _fetch_one(
                client=client,
                url=url,
                depth=depth,
                queue=queue,
                state=state,
                result=result,
                seed=seed,
                max_pages=max_pages,
                max_depth=max_depth,
                verify_ssl=verify_ssl,
            )
        finally:
            async with state.lock:
                state.in_flight -= 1


async def _fetch_one(
    *,
    client: httpx.AsyncClient,
    url: str,
    depth: int,
    queue: asyncio.Queue[tuple[str, int]],
    state: _CrawlState,
    result: CrawlResult,
    seed: str,
    max_pages: int,
    max_depth: int,
    verify_ssl: bool = True,
) -> None:
    async with state.lock:
        if len(result.pages) >= max_pages:
            return

    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        if verify_ssl and _is_ssl_error(exc):
            logger.warning("SSL verify failed for %s, retrying without certificate check", url)
            try:
                response = await _get_without_ssl_verify(client, url, timeout=client.timeout)
            except httpx.HTTPError as retry_exc:
                reason = f"{type(retry_exc).__name__}: {retry_exc}".strip()
                logger.warning("%s %s", url, reason)
                async with state.lock:
                    result.errors.append(FailedUrl(url=url, depth=depth, reason=reason))
                return
        else:
            reason = f"{type(exc).__name__}: {exc}".strip()
            logger.warning("%s %s", url, reason)
            async with state.lock:
                result.errors.append(FailedUrl(url=url, depth=depth, reason=reason))
            return

    final_url = normalize_url(str(response.url)) or url
    if not same_domain(seed, final_url):
        logger.info("left domain after redirect: %s", final_url)
        return

    if response.status_code == 404:
        logger.info("skipped [404] %s", final_url)
        return

    if not response.is_success:
        reason = f"HTTP {response.status_code}"
        logger.info("failed [%s] %s", response.status_code, final_url)
        if response.status_code in RETRYABLE_STATUS:
            async with state.lock:
                result.errors.append(
                    FailedUrl(url=url, depth=depth, reason=reason, status_code=response.status_code)
                )
        return

    content_type = response.headers.get("content-type", "")
    looks_html = "html" in content_type.lower() or final_url.endswith(("/", ".html", ".htm"))
    if content_type and not looks_html:
        return

    # IMPORTANT: use the server's actual URL (str(response.url)) as the base for
    # resolving relative links, NOT the normalized final_url. normalize_url() strips
    # trailing slashes, which breaks relative-link resolution for directory-style URLs
    # like /~user/ (urljoin treats a slash-less base as a "file", dropping the last
    # path segment when joining relative hrefs).
    link_base_url = str(response.url)
    title, text, links = await asyncio.to_thread(parse_page, response.text, link_base_url)

    async with state.lock:
        if len(result.pages) >= max_pages:
            return
        if any(page.url == final_url for page in result.pages):
            return
        result.pages.append(
            Page(
                url=final_url,
                status_code=response.status_code,
                title=title,
                text=text,
                depth=depth,
            )
        )
        logger.info("[%s] %s/%s depth=%s %s", response.status_code, len(result.pages), max_pages, depth, final_url)
        if depth < max_depth and response.is_success:
            for link in links:
                if link in state.seen or not same_domain(seed, link):
                    continue
                state.seen.add(link)
                queue.put_nowait((link, depth + 1))


def _is_ssl_error(exc: BaseException) -> bool:
    """Walk the exception chain (__cause__/__context__) looking for an ssl.SSLError."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return True
        current = current.__cause__ or current.__context__
    return False


async def _get_without_ssl_verify(client: httpx.AsyncClient, url: str, timeout: httpx.Timeout) -> httpx.Response:
    """Retry a single request with certificate verification disabled."""
    async with httpx.AsyncClient(
        headers=dict(client.headers),
        timeout=timeout,
        follow_redirects=True,
        verify=False,
    ) as insecure_client:
        return await insecure_client.get(url)


def to_absolute_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError("website url is empty")
    if not urlparse(value).scheme:
        value = "https://" + value
    normalized = normalize_url(value)
    if not normalized:
        raise ValueError(f"invalid url: {raw}")
    return normalized


def normalize_url(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    path = str(parsed.path or "/")
    suffix = Path(path).suffix.lower()
    if suffix in SKIP_EXTENSIONS:
        return None
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    host = str(parsed.netloc).lower()
    query = str(parsed.query)
    return urlunparse((str(parsed.scheme), host, path, "", query, ""))


def host_key(url: str) -> str:
    host = str(urlparse(url).netloc).lower()
    if host.startswith("www."):
        return host[4:]
    return host


def same_domain(left: str, right: str) -> bool:
    return host_key(left) == host_key(right)


def load_robots(seed_url: str, timeout: float = 10.0, verify_ssl: bool = True) -> RobotFileParser | None:
    robots_url = urljoin(seed_url, "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
            verify=verify_ssl,
        ) as client:
            response = client.get(robots_url)
        if response.status_code >= 400:
            return None
        parser.parse(response.text.splitlines())
        return parser
    except httpx.HTTPError as exc:
        logger.warning("could not read robots.txt (%s): %s", robots_url, exc)
        return None


def allowed_by_robots(parser: RobotFileParser | None, url: str) -> bool:
    if parser is None:
        return True
    return parser.can_fetch(USER_AGENT, url)


def parse_page(html: str, url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    links: list[str] = []
    for tag in soup.find_all("a", href=True):
        href = tag.get("href")
        if not isinstance(href, str):
            continue
        absolute = normalize_url(urljoin(url, href))
        if absolute:
            links.append(absolute)
    text = trafilatura.extract(html, url=url, include_tables=True) or ""
    if not text:
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
    return title, text.strip(), links


def _load_crawl_file(path: str | Path) -> CrawlResult:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = [Page(**page) for page in payload.get("pages", [])]
    errors: list[FailedUrl] = []
    for item in payload.get("errors", []):
        if isinstance(item, dict):
            errors.append(FailedUrl(**item))
            continue
        url, _, reason = str(item).partition(": ")
        errors.append(FailedUrl(url=url.strip(), depth=1, reason=reason.strip() or "unknown"))
    return CrawlResult(
        seed_url=payload["seed_url"],
        crawled_at=payload.get("crawled_at", datetime.now(timezone.utc).isoformat()),
        pages=pages,
        errors=errors,
    )