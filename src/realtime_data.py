"""
Msty Admin MCP - Real-Time Web/Data Integration

Tools for real-time data access including web search, URL fetching,
and YouTube transcript extraction for Knowledge Stacks.
"""

import json
import logging
import re
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from typing import Optional, List, Dict, Any

from .paths import get_msty_paths

logger = logging.getLogger("msty-admin-mcp")


# ============================================================================
# WEB SEARCH
# ============================================================================

def realtime_search(
    query: str,
    max_results: int = 10,
    search_type: str = "web"
) -> Dict[str, Any]:
    """
    Perform a web search (meta-search using DuckDuckGo HTML).

    Note: This is a basic implementation. For production use,
    consider integrating with Msty's built-in search capabilities.

    Args:
        query: Search query
        max_results: Maximum results to return
        search_type: Type of search (web, news, images)

    Returns:
        Dict with search results
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "search_type": search_type,
        "results": [],
    }

    try:
        # Use DuckDuckGo HTML (no API key required)
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8', errors='ignore')

            # Simple parsing of DuckDuckGo HTML results
            # Look for result blocks
            result_pattern = r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a class="result__snippet"[^>]*>([^<]+)</a>'

            links = re.findall(result_pattern, html)
            snippets = re.findall(snippet_pattern, html)

            for i, (link, title) in enumerate(links[:max_results]):
                # Decode DuckDuckGo redirect URL
                if 'uddg=' in link:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                    actual_url = parsed.get('uddg', [link])[0]
                else:
                    actual_url = link

                result["results"].append({
                    "title": title.strip(),
                    "url": actual_url,
                    "snippet": snippets[i].strip() if i < len(snippets) else "",
                    "position": i + 1
                })

        result["total_results"] = len(result["results"])

    except urllib.error.URLError as e:
        result["error"] = f"Network error: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    if not result["results"] and "error" not in result:
        result["note"] = "No results found or search service unavailable"
        result["suggestion"] = "Try using Msty's built-in web search for better results"

    return result


# ============================================================================
# URL FETCHING
# ============================================================================

def realtime_fetch(
    url: str,
    extract_text: bool = True,
    max_length: int = 50000
) -> Dict[str, Any]:
    """
    Fetch content from a URL.

    Args:
        url: URL to fetch
        extract_text: Extract plain text from HTML
        max_length: Maximum content length

    Returns:
        Dict with fetched content
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "fetched": False,
    }

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get('Content-Type', '')
            result["content_type"] = content_type

            raw_content = response.read()

            # Handle encoding
            encoding = 'utf-8'
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip()

            try:
                content = raw_content.decode(encoding, errors='ignore')
            except:
                content = raw_content.decode('utf-8', errors='ignore')

            if extract_text and 'text/html' in content_type:
                # Simple HTML to text extraction
                # Remove script and style elements
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                # Clean up whitespace
                content = re.sub(r'\s+', ' ', content).strip()
                # Decode HTML entities
                content = content.replace('&nbsp;', ' ')
                content = content.replace('&amp;', '&')
                content = content.replace('&lt;', '<')
                content = content.replace('&gt;', '>')
                content = content.replace('&quot;', '"')

            # Truncate if too long
            if len(content) > max_length:
                content = content[:max_length] + "... [truncated]"

            result["content"] = content
            result["content_length"] = len(content)
            result["fetched"] = True

            # Extract title if HTML
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', raw_content.decode('utf-8', errors='ignore'), re.IGNORECASE)
            if title_match:
                result["title"] = title_match.group(1).strip()

    except urllib.error.HTTPError as e:
        result["error"] = f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        result["error"] = f"URL Error: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================================
# YOUTUBE TRANSCRIPT
# ============================================================================

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def realtime_youtube_transcript(
    url_or_id: str,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Extract transcript from a YouTube video.

    Note: This uses YouTube's transcript API. For some videos,
    transcripts may not be available.

    Args:
        url_or_id: YouTube URL or video ID
        language: Preferred language code

    Returns:
        Dict with transcript data
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "input": url_or_id,
        "video_id": None,
        "transcript": None,
    }

    # Extract video ID
    if 'youtube' in url_or_id or 'youtu.be' in url_or_id:
        video_id = extract_youtube_id(url_or_id)
    else:
        video_id = url_or_id if len(url_or_id) == 11 else None

    if not video_id:
        result["error"] = "Could not extract valid YouTube video ID"
        return result

    result["video_id"] = video_id

    try:
        # Try to get video info page
        info_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        req = urllib.request.Request(info_url, headers=headers)

        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')

            # Extract title
            title_match = re.search(r'"title":"([^"]+)"', html)
            if title_match:
                result["title"] = title_match.group(1)

            # Extract channel
            channel_match = re.search(r'"ownerChannelName":"([^"]+)"', html)
            if channel_match:
                result["channel"] = channel_match.group(1)

            # Try to find transcript data in page
            # YouTube embeds caption track info in the page
            caption_match = re.search(r'"captionTracks":\[(.*?)\]', html)

            if caption_match:
                caption_data = caption_match.group(1)

                # Find transcript URL
                url_match = re.search(r'"baseUrl":"([^"]+)"', caption_data)

                if url_match:
                    transcript_url = url_match.group(1).replace('\\u0026', '&')

                    # Fetch transcript
                    transcript_req = urllib.request.Request(transcript_url, headers=headers)

                    with urllib.request.urlopen(transcript_req, timeout=15) as transcript_response:
                        transcript_xml = transcript_response.read().decode('utf-8', errors='ignore')

                        # Parse XML transcript
                        text_segments = re.findall(r'<text[^>]*>([^<]+)</text>', transcript_xml)

                        if text_segments:
                            # Clean up text
                            full_transcript = ' '.join(text_segments)
                            full_transcript = full_transcript.replace('&#39;', "'")
                            full_transcript = full_transcript.replace('&amp;', '&')
                            full_transcript = full_transcript.replace('&quot;', '"')

                            result["transcript"] = full_transcript
                            result["word_count"] = len(full_transcript.split())
                            result["segments"] = len(text_segments)

                            # Ready for Knowledge Stack
                            result["knowledge_stack_ready"] = {
                                "title": result.get("title", f"YouTube: {video_id}"),
                                "content": full_transcript,
                                "source": f"https://youtube.com/watch?v={video_id}",
                                "type": "youtube_transcript"
                            }

                            return result

            result["error"] = "No transcript available for this video"
            result["suggestion"] = "The video may not have captions enabled"

    except urllib.error.URLError as e:
        result["error"] = f"Network error: {e.reason}"
    except Exception as e:
        result["error"] = str(e)

    return result


__all__ = [
    "realtime_search",
    "realtime_fetch",
    "realtime_youtube_transcript",
    "extract_youtube_id",
]
