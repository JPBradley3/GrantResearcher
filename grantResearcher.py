import pandas as pd
import requests
from curl_cffi import requests as curl_requests
import random
import time
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm
import re
from urllib.parse import urlparse, parse_qs, urljoin
import ollama
import json
import os
import hashlib
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
import argparse
import sys


# -------------------------
# Load URLs
# -------------------------

df = pd.read_csv("data/grants_urls.csv")

# Ensure column exists
if "url" not in df.columns:
    raise ValueError("CSV must have a column named 'url'")

def unwrap_url(url):
    """Decode HTML entities and extract the real URL from Google redirect wrappers."""
    import html
    url = html.unescape(str(url).strip())
    if "google.com/url" in url:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        url = (qs.get("q") or qs.get("url") or [url])[0]
    # Decode any remaining percent-encoding in the extracted URL
    from urllib.parse import unquote
    return unquote(url)

df["url"] = df["url"].apply(lambda u: unwrap_url(u) if pd.notna(u) else u)

# -------------------------
# Shared HTTP session + optional cache
# -------------------------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "DNT": "1"
}

import threading
_session_local = threading.local()

def get_session():
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.headers.update(HEADERS)
        _session_local.session = s
    return _session_local.session

session = get_session()

parser = argparse.ArgumentParser(description="Grant researcher scraper and LLM processor")
parser.add_argument("--mode", choices=["scrape", "llm", "both"], default="both", help="Operation mode: scrape only, llm only, or both")
parser.add_argument("--force-refresh", action="store_true", help="Force refresh cached pages/results")
parser.add_argument("--llm-workers", type=int, default=1, help="Number of parallel LLM workers for llm-only mode")
args = parser.parse_args()

try:
    import requests_cache
    requests_cache.install_cache("grant_cache", expire_after=86400)
except ImportError:
    requests_cache = None

CACHE_DIR = Path("cache")
PAGE_CACHE_DIR = CACHE_DIR / "pages"
RESULT_CACHE_DIR = CACHE_DIR / "results"
PAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url):
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def cache_file_path(url, folder, suffix):
    return folder / f"{hash_url(url)}.{suffix}"


def page_cache_path(url):
    return cache_file_path(url, PAGE_CACHE_DIR, "html")


def result_cache_path(url):
    return cache_file_path(url, RESULT_CACHE_DIR, "json")


def load_cached_html(url):
    path = page_cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


def save_cached_html(url, html):
    path = page_cache_path(url)
    path.write_text(html, encoding="utf-8", errors="ignore")


def load_cached_result(url):
    path = result_cache_path(url)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return None
    return None


_result_cache_lock = threading.Lock()

def save_cached_result(url, result):
    path = result_cache_path(url)
    with _result_cache_lock:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def is_asset_url(url):
    return bool(re.search(r"\.(css|js|png|jpe?g|gif|svg|ico|pdf|zip|rar|mp4|webm|mp3)(?:[?#]|$)", url, re.IGNORECASE))


def is_json_text(text):
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def is_xml_text(text):
    stripped = text.lstrip()
    return stripped.startswith("<?xml") or stripped.startswith("<rss") or stripped.startswith("<feed") or stripped.startswith("<xml")


def extract_jsonld_from_html(html):
    json_objects = []
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type=lambda t: t and "json" in t.lower()):
        raw = script.string or script.get_text() or ""
        try:
            parsed = json.loads(raw.strip())
            json_objects.append(parsed)
        except Exception:
            continue
    return json_objects


def extract_xml_from_html(html):
    xml_strings = []
    if "<rss" not in html.lower() and "<feed" not in html.lower() and "<?xml" not in html.lower() and "<xml" not in html.lower():
        return xml_strings
    try:
        xml_soup = BeautifulSoup(html, "xml")
        xml_strings.append(str(xml_soup))
    except Exception:
        pass
    return xml_strings

BLOCK_SIGNALS = [
    "too many requests", "access denied", "403 forbidden", "429",
    "cloudflare", "captcha", "robot", "enable javascript", "checking your browser",
    "just a moment", "ddos-guard"
]

POOL_SIGNALS = [
    r"\$[\d.,]+\s*(million|billion)\s*(total|commitment|invested|endowment|fund|over|last|annual|per year)",
    r"(total|committed|invested|endowment|over [\d]+ years?|last year|annually)\s*[:\-]?\s*\$[\d.,]+",
    r"more than \$[\d.,]+\s*(million|billion)",
    r"nearly \$[\d.,]+\s*(million|billion)",
    r"awarded .{0,30}\$[\d.,]+\s*(million|billion)",
    r"\$[\d.,]+\s*(million|billion) (in |over |across ).{0,20}(grant|year|program)",
]

def is_blocked_page(html):
    sample = html[:3000].lower()
    return any(sig in sample for sig in BLOCK_SIGNALS)


def sanitize_llm_field(value):
    """Return None if the LLM field contains an HTTP error, block message, or is a JSON object string."""
    if not isinstance(value, str):
        return value
    low = value.lower()
    if any(sig in low for sig in BLOCK_SIGNALS):
        return None
    # Reject deadline fields that are JSON objects
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return None
    return stripped or None


def sanitize_amount(value):
    """Return None if the amount looks like a pool/total figure rather than a per-grant award."""
    if not isinstance(value, str):
        return value
    for pattern in POOL_SIGNALS:
        if re.search(pattern, value, re.IGNORECASE):
            return None
    return value.strip() or None


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_emails(text):
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return "; ".join(set(emails)) if emails else None


def extract_amount(text):
    # Focus on keywords that suggest individual award amounts to avoid total pool numbers
    keywords = ["award of", "amount is", "up to", "maximum", "stipend", "grant of", "per applicant", "each", "award size", "funding level"]
    pattern = r"\$\d+(?:[.,]\d+)?(?:\s?[kKMBmb]|(?:\s?(?:million|billion|thousand)))?(?:\s?-\s?\$\d+(?:[.,]\d+)?(?:\s?[kKMBmb]|(?:\s?(?:million|billion|thousand)))?)?"
    
    found_amounts = []
    for kw in keywords:
        for match in re.finditer(rf"\b{kw}\b", text, re.IGNORECASE):
            # Look at a short window after the keyword for the amount
            window = text[match.end() : match.end() + 60]
            matches = re.findall(pattern, window, re.IGNORECASE)
            found_amounts.extend(matches)
            
    # Fallback to general search if no specific context was found
    if not found_amounts:
        found_amounts = re.findall(pattern, text, re.IGNORECASE)

    # Filter out very short matches and remove duplicates
    valid = [m.strip() for m in set(found_amounts) if len(m) > 2]
    return "; ".join(valid) if valid else None


def extract_dates(text):
    # Focus on dates near deadline/application keywords to reduce noise from news/copyrights
    keywords = ["deadline", "due date", "closes", "opens", "apply by", "application period"]
    date_regex = r"(\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b)"
    
    found_dates = []
    for kw in keywords:
        # Find the keyword and look at the next 100 characters for a date
        for match in re.finditer(rf"{kw}", text, re.IGNORECASE):
            window = text[match.end() : match.end() + 100]
            dates_in_window = re.findall(date_regex, window, re.IGNORECASE)
            found_dates.extend(dates_in_window)
            
    # Fallback to general dates if keywords didn't yield results (but limit to likely current years)
    if not found_dates:
        found_dates = [d for d in re.findall(date_regex, text, re.IGNORECASE) if "202" in d]
        
    return "; ".join(set(found_dates)) if found_dates else None


def analyze_with_llm(text):
    """Uses LLM to extract structured data from grant descriptions."""
    prompt = f"""
    Extract the following details from the grant text below.
    Return the result strictly as a JSON object with these keys:
    'summary', 'precise_amount', 'deadline', 'eligibility_criteria'.

    Guidelines for 'precise_amount':
    1. Extract the award amount given TO A SINGLE APPLICANT. This can be:
       - A fixed dollar amount (e.g., '$5,000')
       - A dollar range (e.g., '$10,000 - $25,000')
       - A percentage of project costs (e.g., '50% of eligible project costs', 'up to 80%')
       - A matching ratio (e.g., '1:1 match up to $10,000')
       - A non-dollar award described in words (e.g., 'product donations up to $5,000 retail value')
    2. IGNORE total program budgets, endowments, or aggregate amounts. The following are NOT per-grant amounts and must be ignored:
       - Total fund sizes (e.g., '$100 million housing fund', '$2 billion endowment')
       - Multi-year commitments to all grantees (e.g., '$27 million over three years')
       - Annual giving totals (e.g., 'we gave $45M last year')
       - Any amount described with: 'total', 'commitment', 'invested', 'endowment', 'over X years', 'last year'
    3. If the ONLY amounts mentioned are pool/total figures with NO per-grant amount stated, return null.
    4. Return a plain descriptive string, not an object or list.

    Guidelines for 'deadline':
    1. If the grant has a SPECIFIC deadline, return it as a plain string (e.g., 'March 15, 2025').
    2. If the grant accepts applications on a ROLLING or CONTINUOUS basis with no fixed close date, return 'Rolling / Ongoing'.
    3. If the grant opens and closes in defined CYCLES or WINDOWS throughout the year, describe the cycle plainly (e.g., 'Four cycles: Education March 15, Community June 15, Environment June 15, Arts September 15').
    4. If multiple future deadlines are listed, return all of them separated by ' | '.
    5. Return ONLY a plain string. Do NOT return a JSON object or dictionary.
    6. If no deadline information is found, return null.

    Guidelines for 'eligibility_criteria':
    1. If the grant is invite-only, nomination-only, or by referral, start the value with 'INVITE ONLY - ' followed by any other eligibility details.
    2. Otherwise, summarize who qualifies: org type (e.g., 501(c)(3), nonprofit), location restrictions, size, focus area, or any stated requirements.
    3. Return a plain descriptive string, not a list or object.
    4. If no eligibility information is found, return null.

    Text:
    {text[:12000]}
    """
    try:
        response = ollama.generate(model='llama3', prompt=prompt, format='json', keep_alive="1h")
        raw_content = response['response'].strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()
        return json.loads(raw_content)
    except Exception as e:
        print(f"\n[LLM Error] Could not analyze text: {e}")
        return {
            "summary": "Error in LLM analysis",
            "precise_amount": None,
            "deadline": None,
            "eligibility_criteria": None
        }


CURL_IMPERSONATE = "chrome120"
CURL_BROWSERS = ["chrome120", "chrome110", "chrome107", "edge101"]

def fetch_url(url, session, use_cache=True):
    url = normalize_url(url)
    if use_cache:
        cached_html = load_cached_html(url)
        if cached_html is not None:
            cached_type = None
            if is_json_text(cached_html):
                cached_type = "application/json"
            elif is_xml_text(cached_html):
                cached_type = "application/xml"
            return cached_html, cached_type

    # Random delay to appear human
    time.sleep(random.uniform(1.0, 3.5))

    html = None
    content_type = ""

    # Try curl_cffi first (bypasses Cloudflare/TLS fingerprinting)
    try:
        impersonate = random.choice(CURL_BROWSERS)
        r = curl_requests.get(url, impersonate=impersonate, timeout=20, allow_redirects=True)
        if r.status_code in (403, 429, 503):
            return None
        html = r.text
        content_type = r.headers.get("Content-Type", "").split(";")[0]
    except Exception:
        pass

    # Fall back to requests if curl_cffi failed or returned a blocked page
    if not html or is_blocked_page(html):
        try:
            r = session.get(url, timeout=15, allow_redirects=True)
            if r.status_code in (403, 429, 503):
                return None
            html = r.text
            content_type = r.headers.get("Content-Type", "").split(";")[0]
        except Exception:
            return None

    if not html or is_blocked_page(html):
        return None

    if use_cache:
        save_cached_html(url, html)
    return html, content_type


def crawl_site(start_url, session, max_pages=300, use_cache=True):
    start_url = normalize_url(start_url)
    queue = [start_url]
    seen_urls = {start_url}
    pages = []
    base_domain = urlparse(start_url).netloc

    while queue and len(pages) < max_pages:
        current_url = queue.pop(0)
        result = fetch_url(current_url, get_session(), use_cache=use_cache)
        if result is None:
            continue
        html, content_type = result
        if html is None:
            continue

        pages.append((current_url, html, content_type))

        if content_type and "html" not in content_type.lower():
            continue

        soup = BeautifulSoup(html, "lxml")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith(("mailto:", "tel:", "javascript:")):
                continue

            sub_url = normalize_url(urljoin(current_url, href))
            parsed_sub = urlparse(sub_url)
            if parsed_sub.scheme not in {"http", "https"}:
                continue
            if parsed_sub.netloc != base_domain:
                continue
            if sub_url in seen_urls or is_asset_url(sub_url):
                continue

            seen_urls.add(sub_url)
            queue.append(sub_url)
            if len(seen_urls) >= max_pages:
                break

    return pages


def scrape_page(url, session, force_refresh=False, run_llm=True):
    title = "No Title"
    try:
        cached_result = None if force_refresh else load_cached_result(url)
        if cached_result is not None:
            return cached_result

        page_items = crawl_site(url, session, max_pages=300, use_cache=not force_refresh)
        if not page_items:
            raise Exception("Failed to crawl site pages")

        all_descriptions = []
        all_raw_content = []
        extracted_json = []
        extracted_xml = []
        content_types = set()

        for index, (page_url, page_html, content_type) in enumerate(page_items):
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            page_label = "Main Page" if index == 0 else f"Page {index + 1}"
            content_types.add(content_type or "unknown")

            if content_type and "json" in content_type.lower() or is_json_text(page_html):
                try:
                    parsed = json.loads(page_html)
                    extracted_json.append(parsed)
                    pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
                    all_descriptions.append(f"--- {page_label} ({page_url}) [json] ---\n" + pretty)
                    all_raw_content.append(pretty)
                except Exception:
                    all_descriptions.append(f"--- {page_label} ({page_url}) [json] ---\n" + page_html)
                    all_raw_content.append(page_html)
                continue

            if content_type and "xml" in content_type.lower() or is_xml_text(page_html):
                xml_parts = extract_xml_from_html(page_html)
                if xml_parts:
                    extracted_xml.extend(xml_parts)
                    all_descriptions.append(f"--- {page_label} ({page_url}) [xml] ---\n" + "\n\n".join(xml_parts))
                    all_raw_content.append("\n\n".join(xml_parts))
                else:
                    all_descriptions.append(f"--- {page_label} ({page_url}) [xml] ---\n" + page_html)
                    all_raw_content.append(page_html)
                continue

            page_soup = BeautifulSoup(page_html, "lxml")
            for element in page_soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            lines = page_soup.get_text(separator="\n", strip=True).splitlines()
            page_meaningful = [line.strip() for line in lines if len(line.strip()) > 5]

            all_descriptions.append(f"--- {page_label} ({page_url}) ---\n" + "\n\n".join(page_meaningful))
            all_raw_content.append(" ".join(lines))
            extracted_json.extend(extract_jsonld_from_html(page_html))
            extracted_xml.extend(extract_xml_from_html(page_html))

        full_description = "\n\n".join(all_descriptions)
        clean_content = " ".join(all_raw_content) # For regex search

        emails = extract_emails(clean_content)
        amounts = extract_amount(clean_content)
        dates = extract_dates(clean_content)

        # eligibility signals (very useful later)
        eligibility_keywords = [
            "nonprofit", "501(c)(3)", "organization",
            "eligible", "must be", "requirements"
        ]

        eligibility_hits = [
            k for k in eligibility_keywords
            if k.lower() in clean_content.lower()
        ]

        # build result without LLM first; LLM may be run separately
        result = {
            "url": url,
            "title": title,
            "summary": None,
            "llm_amount": None,
            "llm_deadline": None,
            "llm_eligibility": None,
            "emails": emails,
            "grant_amount": amounts,
            "application_dates": dates,
            "eligibility_signals": "; ".join(eligibility_hits),
            "pages_crawled": len(page_items),
            "content_types": ", ".join(sorted(content_types)),
            "embedded_json": json.dumps(extracted_json, ensure_ascii=False) if extracted_json else None,
            "embedded_xml": "\n\n".join(extracted_xml) if extracted_xml else None
        }
        # save raw_text to allow later LLM processing without re-crawling
        result["raw_text"] = full_description

        # Optionally run LLM now
        if run_llm:
            try:
                llm_data = analyze_with_llm(full_description)
                result["summary"] = sanitize_llm_field(llm_data.get("summary"))
                result["llm_amount"] = sanitize_amount(sanitize_llm_field(llm_data.get("precise_amount")))
                result["llm_deadline"] = sanitize_llm_field(llm_data.get("deadline"))
                result["llm_eligibility"] = sanitize_llm_field(llm_data.get("eligibility_criteria"))
            except Exception:
                pass

        save_cached_result(url, result)
        return result

    except Exception as e:
        result = {
            "url": url,
            "title": None,
            "summary": None,
            "llm_amount": None,
            "llm_deadline": None,
            "llm_eligibility": None,
            "emails": None,
            "grant_amount": None,
            "application_dates": None,
            "eligibility_signals": None,
            "error": str(e)
        }
        save_cached_result(url, result)
        return result


def process_llm_on_cached(urls, force_refresh=False, max_workers=4):
    """Process LLM extraction from cached raw_text or by crawling if needed."""
    results = []

    def process_one(url):
        # load cached result
        cached = None if force_refresh else load_cached_result(url)
        raw_text = None
        if cached is not None and cached.get("summary") and not force_refresh:
            return cached

        if cached is not None and cached.get("raw_text"):
            raw_text = cached.get("raw_text")
        else:
            # attempt to crawl site to build raw_text
            try:
                page_items = crawl_site(url, get_session(), max_pages=300, use_cache=not force_refresh)
                if page_items:
                    parts = []
                    for index, (page_url, page_html, content_type) in enumerate(page_items):
                        if content_type and "json" in (content_type or ""):
                            parts.append(page_html)
                        else:
                            soup = BeautifulSoup(page_html, "lxml")
                            for element in soup(["script", "style", "nav", "footer", "header"]):
                                element.decompose()
                            lines = soup.get_text(separator="\n", strip=True).splitlines()
                            parts.append("\n\n".join([l.strip() for l in lines if len(l.strip()) > 0]))
                    raw_text = "\n\n".join(parts)
            except Exception:
                raw_text = None

        if not raw_text:
            # nothing to process
            res = cached or {"url": url, "error": "no raw text available"}
            save_cached_result(url, res)
            return res

        # call LLM
        try:
            llm_data = analyze_with_llm(raw_text)
        except Exception as e:
            llm_data = {"summary": None, "precise_amount": None, "deadline": None, "eligibility_criteria": None}

        result = cached or {"url": url}
        result.update({
            "summary": sanitize_llm_field(llm_data.get("summary")),
            "llm_amount": sanitize_amount(sanitize_llm_field(llm_data.get("precise_amount"))),
            "llm_deadline": sanitize_llm_field(llm_data.get("deadline")),
            "llm_eligibility": sanitize_llm_field(llm_data.get("eligibility_criteria")),
            "raw_text": raw_text
        })
        save_cached_result(url, result)
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({"url": futures[future], "error": str(e)})

    return results


# -------------------------
# Run scraper
# -------------------------

if args.mode != "scrape":
    try:
        models_response = ollama.list()
        models = getattr(models_response, "models", models_response)
        available_models = []
        for m in models:
            if isinstance(m, dict):
                available_models.append(m.get("name") or m.get("model"))
            elif hasattr(m, "name"):
                available_models.append(m.name)
            elif hasattr(m, "model"):
                available_models.append(m.model)
            else:
                available_models.append(str(m))
        if not any('llama3' in str(m) for m in available_models if m):
            print("Model 'llama3' not found locally. Pulling now (this may take a few minutes)...")
            ollama.pull('llama3')
            print("Model 'llama3' downloaded successfully.")
        else:
            print("Ollama connection verified and 'llama3' model is ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not connect to the Ollama server. Please ensure the Ollama application is running in your system tray.\nError: {e}")
        if args.mode == "scrape":
            print("Continuing in scrape-only mode (no LLM).")
        else:
            exit(1)

results = []

all_urls = df["url"].dropna().tolist()
if args.mode == "llm":
    # Run LLM processing on cached pages/results
    results = process_llm_on_cached(all_urls, force_refresh=args.force_refresh, max_workers=args.llm_workers)
    out = pd.DataFrame(results)
else:
    # scrape or both
    max_workers = min(12, max(4, (os.cpu_count() or 1) * 2))
    results = [None] * len(all_urls)
    run_llm_flag = args.mode == "both"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(scrape_page, url, get_session(), args.force_refresh, run_llm_flag): index
            for index, url in enumerate(all_urls)
        }

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            results[index] = future.result()

    out = pd.DataFrame(results)

# -------------------------
# Save output
# -------------------------

# Drop large per-row fields from the CSV output so the file is easier to open
csv_out = out.copy()
for col in ["raw_text", "embedded_json", "embedded_xml"]:
    if col in csv_out.columns:
        csv_out = csv_out.drop(columns=[col])

chunk_size = 1000
if len(csv_out) <= chunk_size:
    csv_out.to_csv("output/grants_fleshed.csv", index=False, quoting=1)
    print("Done -> output/grants_fleshed.csv")
else:
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(csv_out), chunk_size):
        chunk = csv_out.iloc[i : i + chunk_size]
        part_num = i // chunk_size + 1
        out_path = out_dir / f"grants_fleshed_part_{part_num:02d}.csv"
        chunk.to_csv(out_path, index=False)
    print(f"Done -> split into {((len(csv_out) - 1) // chunk_size) + 1} files in output/")