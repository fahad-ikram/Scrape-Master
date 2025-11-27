import re
import html
import time
import random
import smtplib
import requests
import dns.resolver
import pandas as pd
import streamlit as st

from PIL import Image
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import unquote
from urllib.parse import urlparse
from urllib.parse import urljoin, urlparse
from validate_email_address import validate_email
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------

icon = Image.open(r"Fahad.png")
st.set_page_config(page_title="Fahad Ikram", page_icon=icon, layout="wide")

# ---------- Helpers (all in-memory, no files) ----------
USELESS_SITES = set([
    'youtube.com','facebook.com','instagram.com','twitter.com','linkedin.com','tiktok.com',
    'pinterest.com','snapchat.com','google.com','unsplash.com','.gov','freepik.com','pexels.com',
    'pixabay.com','reddit.com','whatsapp.com','telegram.org','tumblr.com','discord.com','vimeo.com',
    'x.com','bsky.app','threads.com','#','img'
])

# Simple URL validator
def url_validator(url):
    try:
        p = urlparse(url if url.startswith(('http://','https://')) else 'http://' + url)
        return bool(p.netloc)
    except:
        return False

def clean_domain(url):
    p = urlparse(url)
    return f"{p.scheme or 'http'}://{p.netloc}/"

# Generate page range for blog pagination
def generate_pages(base_url, start, end):
    """
    Generate paginated URLs from a base URL.
    Handles:
    - base URL ending with or without slash
    - base URL already containing /page/{number}/
    """
    # remove trailing slash
    base_url = base_url.rstrip('/')

    # check if URL already ends with /page/{number}
    match = re.search(r'/page/(\d+)$', base_url)
    if match:
        # strip the existing /page/{number} part
        base_url = base_url[:match.start()]

    # now append /page/{i}/ for each page
    return [f"{base_url}/page/{i}/" for i in range(start, end+1)]



def generate_user_agents():
    # Operating systems
    desktop_os = [
        "Windows NT 10.0; Win64; x64",
        "Windows NT 6.1; Win64; x64",
        "Macintosh; Intel Mac OS X 10_15_7",
        "Macintosh; Intel Mac OS X 13_0",
        "X11; Ubuntu; Linux x86_64"
    ]
    mobile_os = [
        "Linux; Android 13; Pixel 6",
        "Linux; Android 12; SM-G998B",
        "Linux; Android 11; Redmi Note 10",
        "iPhone; CPU iPhone OS 17_0 like Mac OS X",
        "iPad; CPU OS 16_3 like Mac OS X"
    ]

    browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]

    user_agents_set = set()

    while len(user_agents_set) < 1000:
        is_mobile = random.choice([True, False])
        os = random.choice(mobile_os if is_mobile else desktop_os)
        browser = random.choice(browsers)
        version = random.randint(90, 130)

        # ----- Desktop User-Agents -----
        if not is_mobile:
            if browser == "Safari":
                user_agent = (
                    f"Mozilla/5.0 ({os}) AppleWebKit/605.1.15 "
                    f"(KHTML, like Gecko) Version/{version}.0 Safari/{version}.0"
                )
            elif browser == "Edge":
                user_agent = (
                    f"Mozilla/5.0 ({os}) AppleWebKit/537.36 "
                    f"(KHTML, like Gecko) Chrome/{version}.0 Safari/537.36 Edg/{version}.0"
                )
            elif browser == "Opera":
                user_agent = (
                    f"Mozilla/5.0 ({os}) AppleWebKit/537.36 "
                    f"(KHTML, like Gecko) Chrome/{version}.0 Safari/537.36 OPR/{version}.0"
                )
            elif browser == "Firefox":
                user_agent = (
                    f"Mozilla/5.0 ({os}; rv:{version}.0) Gecko/20100101 Firefox/{version}.0"
                )
            else:  # Chrome
                user_agent = (
                    f"Mozilla/5.0 ({os}) AppleWebKit/537.36 "
                    f"(KHTML, like Gecko) Chrome/{version}.0 Safari/537.36"
                )

        # ----- Mobile User-Agents -----
        else:
            if "Android" in os:
                if browser == "Chrome":
                    user_agent = (
                        f"Mozilla/5.0 ({os}) AppleWebKit/537.36 "
                        f"(KHTML, like Gecko) Chrome/{version}.0 Mobile Safari/537.36"
                    )
                elif browser == "Firefox":
                    user_agent = (
                        f"Mozilla/5.0 ({os}; rv:{version}.0) Gecko/20100101 Firefox/{version}.0"
                    )
                else:
                    user_agent = (
                        f"Mozilla/5.0 ({os}) AppleWebKit/537.36 "
                        f"(KHTML, like Gecko) {browser}/{version}.0 Mobile Safari/537.36"
                    )
            else:  # iPhone or iPad
                user_agent = (
                    f"Mozilla/5.0 ({os}) AppleWebKit/605.1.15 "
                    f"(KHTML, like Gecko) Version/{version}.0 Mobile/15E148 Safari/604.1"
                )

        user_agents_set.add(user_agent)

    return list(user_agents_set)

agents = agents = random.choice(generate_user_agents())



# Fetch function (synchronous) with timeout
def fetch_url(url, timeout=20):
    try:
        headers = {'User-Agent': agents}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return ''

# Extract article links by class name
def get_article_links_from_page(html, base_url=None):
    """
    Automatically find internal article links without CSS class or pattern.
    - Keeps only same-domain links.
    - Skips contact/about/etc. pages.
    - Keeps only links longer than base_url + 20 characters.
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = set()

    if not base_url:
        return links

    base_domain = urlparse(base_url).netloc.lower()
    base_len = len(base_url.rstrip('/'))

    for a in soup.find_all('a', href=True):
        href = a['href']
        if not href or '#' in href or 'javascript:' in href:
            continue

        full_url = urljoin(base_url, href)
        link_domain = urlparse(full_url).netloc.lower()

        # 1️⃣ Only same-domain links
        if link_domain != base_domain:
            continue

        # 3️⃣ Only keep URLs at least 20 chars longer than the base
        if len(full_url) < base_len + 20:
            continue

        links.add(full_url)

    return links

# Extract external links from article content
def get_external_links_from_html(html, useless_domains, base_url=None):
    links = set()

    try:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if not href:
                continue
            full = urljoin(base_url or '', href)
            domain = urlparse(full).netloc.lower()
            if domain and not any(u in domain for u in useless_domains):
                # normalize
                links.add(clean_domain(full))
    except:
        pass
    return links

# For email finder: find candidate pages (contact/about/etc.) then scrape emails
KEYWORDS = ['contact', 'about', 'privacy', 'policy', 'accessibility','disclaimer', 'author', 'terms', 'write', 'advertise','team','impressum','legal','conditions','cookies','support','help','faq','customer','service','press','media','get-in-touch','reach-us','who-we-are','our-story']

def find_candidate_pages_for_emails(html, base_url=None):
    pages = set()
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if href and any(k in href.lower() for k in KEYWORDS):
                pages.add(urljoin(base_url or '', href))
    except:
        pass
    return pages



# ---------- Cloudflare data-cfemail decode ----------
def _decode_cfemail(cfhex: str) -> str:
    try:
        # allow values with or without "0x" and strip non-hex chars
        cfhex = re.sub(r'[^0-9a-fA-F]', '', cfhex or '')
        b = bytes.fromhex(cfhex)
        if not b:
            return ''
        key = b[0]
        return ''.join(chr(x ^ key) for x in b[1:])
    except Exception:
        return ''

# ---------- cleaning / deobfuscation used for mailto parts and text ----------
def _clean_obfuscation(s: str) -> str:
    if not s:
        return ''
    s = s.replace('\xa0', ' ')
    s = html.unescape(s)

    # decode \uXXXX sequences like u003e -> >
    s = re.sub(r'u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    # common textual obfuscations (case-insensitive)
    s = re.sub(r'\s*\[\s*at\s*\]\s*', '@', s, flags=re.I)
    s = re.sub(r'\s*\(\s*at\s*\)\s*', '@', s, flags=re.I)
    s = re.sub(r'\s+at\s+', '@', s, flags=re.I)
    s = re.sub(r'\s*\[\s*dot\s*\]\s*', '.', s, flags=re.I)
    s = re.sub(r'\s*\(\s*dot\s*\)\s*', '.', s, flags=re.I)
    s = re.sub(r'\s+dot\s+', '.', s, flags=re.I)
    # remove stray angle brackets/quotes/brackets/spaces leftover
    s = re.sub(r'[<>\s\(\)\[\]"\']+', '', s)
    return s.strip()


# ---------- strict-ish final email regex ----------
EMAIL_RE = re.compile(r'^[A-Za-z0-9!#$%&\'*+/=?^_`{|}~\.-]{1,64}@[A-Za-z0-9\.-]{1,253}\.[A-Za-z]{2,24}$')
VALID_EMAIL = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,24}$")

def _is_sane_email(e: str) -> bool:
    try:
        local, domain = e.split('@', 1)
    except ValueError:
        return False
    if '.' not in domain:
        return False
    last = domain.rsplit('.', 1)[-1]
    if not last.isalpha() or not (2 <= len(last) <= 24):
        return False
    if not (1 <= len(local) <= 64):
        return False
    if not EMAIL_RE.match(e):
        return False
    return True

def _verify_in_source(candidate, raw_html, visible_text):
    cand = html.unescape(candidate or '').replace('\xa0', ' ')
    return (cand.lower() in raw_html.lower()) or (cand.lower() in visible_text.lower())

def _sanitize_email_candidate(e: str) -> str:
    """
    Remove leading/trailing junk (like '>', '<', ':', whitespace, or u003e)
    before final validation.
    """
    if not e:
        return ''
    # decode Unicode escapes like u003e → >
    e = re.sub(r'u0*([0-9a-fA-F]{2,4})', lambda m: chr(int(m.group(1),16)), e)
    # remove prefixes like //, /?, mailto:, http://, https://
    e = re.sub(r'^(?:mailto:|https?:\/\/|\/\/|\/\?)', '', e, flags=re.I)
    # remove leading/trailing junk characters
    e = e.strip(' \t\n\r<>:;"\'\u200b')
    return e


def clean_email(text):
    # remove everything before consecutive dots
    cleaned = re.split(r"\.{2,}", text)[-1]
    # extract valid email from remaining text
    match = re.search(r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~\.-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}", cleaned)
    return match.group(0) if match else False


# ---------- Main email extraction function ----------
def extract_emails(html_text: str) -> list:
    """
    Extract and return a sorted list of VERIFIED email addresses found in html_text.
    Verified sources:
      - Cloudflare protected (`data-cfemail`) decoded
      - mailto: links (cleaned and verified)
      - direct visible/textual emails (cleaned + verified they appear in page)
    The function normalizes, filters placeholders/junk and returns sorted unique emails (lowercased).
    """
    soup = BeautifulSoup(html_text or '', 'html.parser')

    # 🧩 NEW: Remove placeholder text from input and textarea elements
    for inp in soup.find_all(['input', 'textarea']):
        if inp.has_attr('placeholder'):
            inp['placeholder'] = ''

    raw_html = html_text or ''
    visible_text = html.unescape(soup.get_text(separator=' ', strip=True)).replace('\xa0', ' ')
    found = set()

    # 1) Cloudflare protected: data-cfemail
    for el in soup.find_all(attrs={'data-cfemail': True}):
        cf = el.get('data-cfemail')
        if cf:
            e = _decode_cfemail(cf)
            if e:
                e = e.strip().lower()
                if _is_sane_email(e) and validate_email(e) and VALID_EMAIL.match(e):
                    if '@' not in e:
                        continue
                    found.add(e)

    # 2) mailto: links (may be obfuscated inside href)
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href.lower().startswith('mailto:'):
            addr = href.split(':', 1)[1].split('?')[0]
            addr = _clean_obfuscation(addr)
            addr = unquote(addr)               # decode %20, %40 etc
            addr = _sanitize_email_candidate(addr)
            addr = addr.strip().lower()
            if _is_sane_email(addr) and _verify_in_source(addr, raw_html, visible_text):
                # 🧩 Skip if address appears inside a placeholder attribute
                if re.search(r'placeholder\s*=\s*["\'].*' + re.escape(addr) + r'.*["\']', raw_html, flags=re.I):
                    continue
                if '@' not in addr:
                    continue
                if '@' in addr and validate_email(addr) and VALID_EMAIL.match(addr):
                    found.add(addr)

    # 3) visible textual emails: scan visible_text for common patterns and deobfuscate nearby tokens
    # This regex catches a wide variety; results are then cleaned and validated.
    textual_candidates = re.findall(
        r'([A-Za-z0-9!#$%&\'*+/=?^_`{|}~\.-]+\s*(?:@|\[at\]|\(at\)|\sat\s)\s*[A-Za-z0-9\.-]+\s*(?:\.|\[dot\]|\(dot\)|\sdot\s)\s*[A-Za-z]{2,24})',
        visible_text, flags=re.I
    )
    # Also pull plain-looking emails (already well-formed) from visible text
    textual_candidates += re.findall(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24})', visible_text)

    for cand in textual_candidates:
        # c = _clean_obfuscation(cand)
        # c = unquote(c).strip().lower()
        c = _clean_obfuscation(cand)
        c = unquote(c)
        c = _sanitize_email_candidate(c).lower()

        # remove trailing punctuation often captured like comma/period
        c = re.sub(r'[,\.;:]+$', '', c)
        if '@' not in c:
            continue
        if _is_sane_email(c) and _verify_in_source(c, raw_html, visible_text):
            # 🧩 Skip if appears inside placeholder
            if re.search(r'placeholder\s*=\s*["\'].*' + re.escape(c) + r'.*["\']', raw_html, flags=re.I):
                continue
            found.add(c)

    # 4) Fallback: scan entire raw HTML for common email-like tokens (clean & filter)
    raw_candidates = re.findall(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24})', raw_html)
    for rc in raw_candidates:
        em = unquote(rc).strip().lower()
        em = re.sub(r'[<>\'"]', '', em)
        if '@' not in em:
            continue
        if _is_sane_email(em) and _verify_in_source(em, raw_html, visible_text):
            found.add(em)

    # 5) Final filtering to drop placeholders, hashes, image junk, or common no-reply addresses
    cleaned = set()
    for em in found:
        if not em:
            continue

        lower = em.lower()

        # ✅ HARD FILTER: stop all garbage
        if '@' not in lower:
            continue

        # Skip common garbage
        if any(x in lower for x in [
            'img','u003e','you','your','mysite.com','doe.com',
            'png','jpg','jpeg','gif','svg','webp',
            'example','domain.com','invalid',
            'no-reply@','noreply@','do-not-reply@','test.com',
            'subject=','body=','/?','http://','https://','%','@xxx.xx',
            '.gov','.edu','email.com','address.com','myemail'
        ]):
            continue

        # Skip image-ending fake emails
        if re.search(r'@\S+\.(jpg|jpeg|png|gif|svg|webp)$', lower):
            continue

        if re.search(r'^[0-9a-f]{20,}@', lower):
            continue

        email = clean_email(lower)
        if email:
            if not VALID_EMAIL.match(email):
                continue
            if not validate_email(email):
                continue
            cleaned.add(email)

    return sorted(cleaned)


# ----------- Get base url ----------
def get_base_url(full_url):
    parsed = urlparse(full_url)
    return parsed.netloc



# ---------- Concurrency wrappers ----------

def parallel_fetch(urls, max_workers):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = { ex.submit(fetch_url, u): u for u in urls }
        for fut in as_completed(futures):
            u = futures[fut]
            try:
                results[u] = fut.result()
            except:
                results[u] = ''
    return results



# ---------- Email verification logic ----------
def verify_email(email):
    """
    Enhanced email verification with classification:
    Valid, Invalid, Catch-All, Risky
    """
    result = {
        "Email": email,
        "Status": "Invalid",  # Default
        "Catch_All": False
    }

    # --- 1️⃣ Syntax check ---
    try:
        if not validate_email(email):
            result["Status"] = "Invalid"
            return result
    except Exception:
        result["Status"] = "Invalid"
        return result

    # --- 2️⃣ MX Record Check ---
    try:
        domain = email.split('@')[-1]
        if not domain:
            result["Status"] = "Invalid"
            return result
        resolver = dns.resolver.Resolver()
        resolver.nameservers = ["1.1.1.1", "8.8.8.8"]  # reliable DNS
        mx_records = sorted([(r.preference, str(r.exchange).rstrip('.')) 
                             for r in resolver.resolve(domain, 'MX')])
        mx_host = mx_records[0][1]
    except Exception:
        result["Status"] = "Invalid"
        return result

    # --- 3️⃣ SMTP Check ---
    try:
        with smtplib.SMTP(mx_host, timeout=8) as server:
            server.helo("example.com")
            server.mail("check@example.com")

            # Check actual email
            code, response = server.rcpt(email)
            response_text = response.decode() if isinstance(response, bytes) else str(response)

            # Catch-All Test with random email
            random_user = f"nonexistent_{random.randint(100000,999999)}@{domain}"
            server.mail("check@example.com")
            fake_code, _ = server.rcpt(random_user)
            catch_all = 200 <= fake_code < 300
            result["Catch_All"] = catch_all

            # --- 4️⃣ Classification Logic ---
            if 200 <= code < 300 and not catch_all:
                result["Status"] = "Valid"
            elif catch_all:
                result["Status"] = "Catch-All"
            elif code in (450, 451, 452) or "greylist" in response_text.lower():
                result["Status"] = "Risky"
            else:
                # If SMTP rejects, mark risky instead of invalid (many servers hide real status)
                result["Status"] = "Risky"

    except Exception:
        # Any timeout or connection error → Risky
        result["Status"] = "Risky"

    return result


def normalize_domain(url):
    if not url:
        return None
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path   # handles urls like 'example.com'
        domain = domain.replace("www.", "")     # remove www
        return f"https://{domain}/"             # final uniform format
    except:
        return None



# ---------- Streamlit UI ----------
st.title('💻⚡ ScrapeMaster — Focus is the best time-saver. ⏱️')


mode = st.sidebar.selectbox('Choose Function', ['Blog Research', 'Email Finder', 'Verify Emails'])


if mode == 'Blog Research':
    st.header('')
    # Common controls
    concurrency = st.sidebar.slider('Max concurrent threads', min_value=10, max_value=100, value=20, step=5)
    col1, col2 = st.columns(2)
    with col1:
        base_url = st.text_input('Website URL (e.g. https://example.com)')
        page_mode = st.radio('Pages', ['Single Page', 'Multiple Pages'])
    with col2:
        admin_upload = st.file_uploader('Optional: Upload admin list (.txt or .csv)', type=['txt','csv'])
        all_or_new = st.radio('Add all data or just new entries', ['All Data', 'New Data'])
        if page_mode == 'Multiple Pages':
            start_page = st.number_input('Start page', min_value=1, value=1)
            end_page = st.number_input('End page', min_value=2, value=5)

    run = st.button('Run Blog Research')

    if run:
        if not base_url:
            st.error('Enter a valid URL')
        elif not url_validator(base_url):
            st.error('Enter a valid URL')
        else:
            t0 = time.time()
            fetch_info = st.empty()
            fetch_info.info('Fetching pages...')


            # prepare admin avoid list
            given_domain = get_base_url(base_url)
            avoid_domains = set(USELESS_SITES) | {given_domain}
            if admin_upload is not None:
                try:
                    text = admin_upload.getvalue().decode('utf-8')
                except:
                    text = str(admin_upload.getvalue())
                for line in StringIO(text):
                    d = line.strip()
                    if d:
                        avoid_domains.add(urlparse(d).netloc if url_validator(d) else d)

            # Get article pages
            page_urls = [base_url]
            if page_mode == 'Multiple Pages':
                page_urls = generate_pages(base_url, int(start_page), int(end_page))

            # Fetch listing pages
            listing_html = parallel_fetch(page_urls, max_workers=concurrency)

            # Extract article links
            article_links = set()
            for u, html in listing_html.items():
                article_links.update(get_article_links_from_page(html, base_url=u))

            st.write(f'{len(article_links)} Articles Found!')
            fetch_info.empty()
            # Fetch article pages and extract external links
            # article_html = parallel_fetch(article_links or [], max_workers=concurrency)
            # external_domains = set()
            # for u, html in article_html.items():
            #     external_domains.update(get_external_links_from_html(html, avoid_domains, base_url=u))

            # # Convert to dataframe (unique)
            # df = pd.DataFrame(sorted(external_domains), columns=['client_url'])

            # # Simulate all/new behaviour: if 'New Data' we simply remove duplicates in the shown dataframe (virtual)
            # if all_or_new == 'New Data':
            #     df = df.drop_duplicates()

            # st.success(f'Done — extracted {len(df)} external client URLs in {time.time()-t0:.1f}s')
            # st.dataframe(df)

            # csv_bytes = df.to_csv(index=False).encode('utf-8')
            # st.download_button('Download CSV', data=csv_bytes, file_name='clients.csv', mime='text/csv')
            article_html = parallel_fetch(article_links or [], max_workers=concurrency)

            # --- Progress bar ---
            progress_text = st.empty()
            progress_bar = st.progress(0)

            total_articles = len(article_links)
            processed = 0

            rows = []
            for u, html in article_html.items():
                processed += 1
                progress_bar.progress(processed / total_articles)
                progress_text.info(f"Processing articles: {processed}/{total_articles}")
                links = get_external_links_from_html(html, avoid_domains, base_url=u)
                for link in links:
                    rows.append({'source_article': u, 'client_url': link})
            # Clear progress UI
            progress_bar.empty()
            progress_text.empty()
            if rows:
                df = pd.DataFrame(rows)
                df['client_url'] = df['client_url'].apply(normalize_domain)
                # drop duplicates
                df = df = df.drop_duplicates().reset_index(drop=True)
            else:
                df = pd.DataFrame(columns=['source_article', 'client_url'])

            if all_or_new == 'New Data':
                df = df.drop_duplicates()

            st.success(f'Done — extracted {len(set(df['client_url']))} unique external client URLs in {time.time()-t0:.1f}s')
            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv_bytes, file_name='clients.csv', mime='text/csv')


elif mode == 'Email Finder':
    st.header('')
    # Common controls
    concurrency = st.sidebar.slider('Max concurrent threads', min_value=10, max_value=200, value=20, step=5)
    col1, col2 = st.columns(2)
    with col1:
        urls_upload = st.file_uploader('Upload .txt or .csv with URLs (one per line) or paste below', type=['txt','csv'])
        pasted = st.text_area('Or paste URLs here (one per line)')
        use_range = st.checkbox('Use a range (from-to) on the pasted/uploaded list')
        if use_range:
            r_from = st.number_input('From (1-based index)', min_value=1, value=1)
            r_to = st.number_input('To (inclusive)', min_value=1, value=10)
    with col2:
        extract_from_candidates = st.checkbox('Also visit candidate pages (contact/about/etc.)', value=True)
        max_emails_per_site = st.number_input('Max emails per site to keep', min_value=1, value=30)

    run = st.button('Run Email Finder')

    if run:
        # collect URLs from upload or paste
        text_urls = []
        if urls_upload is not None:
            try:
                raw = urls_upload.getvalue().decode('utf-8')
            except:
                raw = str(urls_upload.getvalue())
            text_urls = [line.strip() for line in StringIO(raw) if line.strip()]
        if pasted:
            text_urls += [line.strip() for line in StringIO(pasted) if line.strip()]

        # normalize
        text_urls = [u if u.startswith(('http://','https://')) else 'http://' + u for u in text_urls]
        text_urls = [u for u in text_urls if url_validator(u)]

        if not text_urls:
            st.error('No valid URLs provided')
        else:
            # apply range
            if use_range:
                start_i = max(1, int(r_from)) - 1
                end_i = min(len(text_urls), int(r_to))
                text_urls = text_urls[start_i:end_i]

            st.info(f'Processing {len(text_urls)} sites...')
            t0 = time.time()

            # Step 1: fetch homepage of each site
            homepage_html = parallel_fetch(text_urls, max_workers=concurrency)

            rows = []   # each dict will become one row in the final DataFrame

            for site, i in homepage_html.items():
                found_emails = set()
                # extract emails from homepage
                found_emails.update(extract_emails(i))

                if extract_from_candidates:
                    candidate_pages = find_candidate_pages_for_emails(i, base_url=site)
                    candidate_pages = list(candidate_pages)
                    candidates_html = parallel_fetch(candidate_pages, max_workers=concurrency)
                    for ch in candidates_html.values():
                        found_emails.update(extract_emails(ch))


                # limit number of emails per site
                emails_list = sorted(found_emails)[:int(max_emails_per_site)]

                if emails_list:
                    for em in emails_list:
                        # repeat site for every email
                        rows.append({'site': site, 'email': em.lower()})
                else:
                    rows.append({'site': site, 'email': ''})

            # make a DataFrame with one email per row
            df = pd.DataFrame(rows)

            # show success message
            st.success(f"Done — scanned {len(set([r['site'] for r in rows]))} sites in {time.time()-t0:.1f}s")

            # group by site to check if site has any emails
            site_has_email = df.groupby('site')['email'].apply(lambda x: any(e.strip() for e in x))

            # sites with emails
            sites_with_emails = site_has_email[site_has_email].index.tolist()
            # sites without emails
            sites_without_emails = site_has_email[~site_has_email].index.tolist()

            # create final DataFrames
            df_with = df[(df['site'].isin(sites_with_emails)) & (df['email'].str.strip() != '')].reset_index(drop=True)
            df_without = pd.DataFrame({'site': sites_without_emails, 'email': ['']*len(sites_without_emails)})

            # create tabs
            tab1, tab2 = st.tabs(["✅Contacts Found", "❌Contacts Not Found"])

            with tab1:
                st.dataframe(df_with)
                if not df_with.empty:
                    csv_bytes_with = df_with.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV', data=csv_bytes_with, file_name='emails_with.csv', mime='text/csv')
                else:
                    st.info("No emails found for any site.")

            with tab2:
                st.dataframe(df_without)
                if not df_without.empty:
                    csv_bytes_without = df_without.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV', data=csv_bytes_without, file_name='emails_without.csv', mime='text/csv')
                else:
                    st.info("All sites have at least one email.")


elif mode == 'Verify Emails':
    
    st.title('')

    # ---------------- Sidebar ----------------
    threads = st.sidebar.slider("Max Parallel Threads", min_value=2, max_value=20, value=5, step=1)

    emails = []

    # ---------------- Input Section ----------------
    input_text = st.text_area("Enter email addresses (one per line):")

    if input_text:
        emails = [e.strip() for e in input_text.splitlines() if e.strip()]
    else:
        uploaded_file = st.file_uploader("Upload a CSV file with 'email' column", type=["csv",'xls','xlsx'])

        if uploaded_file:
        # Determine file type and read accordingly
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:  # .xls or .xlsx
                df = pd.read_excel(uploaded_file)
            email_col = st.selectbox("Select the email column:", df.columns)
            emails = df[email_col].dropna().astype(str).tolist()

    # ---------------- Verification ----------------
    if st.button("🚀 Verify Emails"):
        if not emails:
            st.warning("Please enter or upload some emails first.")
        else:
            total = len(emails)
            results = []

            # Placeholder for info message
            info_placeholder = st.empty()
            # Progress bar
            progress_bar = st.progress(0)

            info_placeholder.info(f"Verifying 0/{total} emails... ⏳")

            # --- Parallel Execution ---
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(verify_email, e): e for e in emails}
                completed = 0
                for future in as_completed(futures):
                    results.append(future.result())
                    completed += 1
                    # Update progress
                    progress_bar.progress(completed / total)
                    info_placeholder.info(f"Verifying {completed}/{total} emails... ⏳")

            # Clear the info message after completion
            info_placeholder.empty()
            progress_bar.empty()

            # After collecting results in a list
            df_results = pd.DataFrame(results)

            # Remove duplicate emails
            df_results = df_results.drop_duplicates(subset="Email").reset_index(drop=True)

            # --- Split by Status ---
            valid_df = df_results[(df_results["Status"] == "Valid") | (df_results["Status"] == "Catch-All")]
            invalid_df = df_results[df_results["Status"] == "Invalid"]
            risky_df = df_results[df_results["Status"] == "Risky"]

            st.success("✅ Verification Completed")

            # --- Tabs ---
            tab1, tab2, tab3 = st.tabs([
                f"✅ Valid ({len(valid_df)})",
                f"❌ Invalid ({len(invalid_df)})",
                f"❓ Risky ({len(risky_df)})"
            ])

            # --- Helper: Download Buttons ---
            def download_emails(df, label):
                if not df.empty:
                    csv = StringIO()
                    df.to_csv(csv, index=False)
                    st.download_button(
                        f"⬇️ Download {label}",
                        csv.getvalue(),
                        file_name=f"{label.lower().replace(' ','_')}.csv",
                        mime="text/csv"
                    )

            # --- Display Tabs ---
            with tab1:
                st.subheader(f"✅ Valid Emails — {len(valid_df)} found")
                if not valid_df.empty:
                    st.dataframe(valid_df[["Email"]].reset_index(drop=True))  # Only Email column
                    download_emails(valid_df[["Email"]], "Valid Emails")      # Download only emails
                else:
                    st.info("No valid emails found.")

            with tab2:
                st.subheader(f"❌ Invalid Emails — {len(invalid_df)} found")
                if not invalid_df.empty:
                    st.dataframe(invalid_df[["Email"]].reset_index(drop=True))
                    download_emails(invalid_df[["Email"]], "Invalid Emails")
                else:
                    st.info("No invalid emails found.")

            with tab3:
                st.subheader(f"❓ Risky Emails — {len(risky_df)} found")
                if not risky_df.empty:
                    st.dataframe(risky_df[["Email"]].reset_index(drop=True))
                    download_emails(risky_df[["Email"]], "Risky Emails")
                else:
                    st.info("No risky emails found.")

# Footer / help
st.markdown('---')