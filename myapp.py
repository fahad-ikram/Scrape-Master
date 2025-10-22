import io
import re
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

EMAIL_RE = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

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

# Extract and clean emails from HTML
import re
import html
from urllib.parse import unquote
from bs4 import BeautifulSoup

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
EMAIL_RE = re.compile(
    r'^[A-Za-z0-9!#$%&\'*+/=?^_`{|}~\.-]{1,64}@[A-Za-z0-9\.-]{1,253}\.[A-Za-z]{2,24}$'
)

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
                if _is_sane_email(e):
                    found.add(e)

    # 2) mailto: links (may be obfuscated inside href)
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href.lower().startswith('mailto:'):
            addr = href.split(':', 1)[1].split('?')[0]
            addr = unquote(addr)               # decode %20, %40 etc
            addr = _clean_obfuscation(addr)
            addr = addr.strip().lower()
            if _is_sane_email(addr) and _verify_in_source(addr, raw_html, visible_text):
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
        c = _clean_obfuscation(cand)
        c = unquote(c).strip().lower()
        # remove trailing punctuation often captured like comma/period
        c = re.sub(r'[,\.;:]+$', '', c)
        if _is_sane_email(c) and _verify_in_source(c, raw_html, visible_text):
            found.add(c)

    # 4) Fallback: scan entire raw HTML for common email-like tokens (clean & filter)
    raw_candidates = re.findall(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24})', raw_html)
    for rc in raw_candidates:
        em = unquote(rc).strip().lower()
        em = re.sub(r'[<>\'"]', '', em)
        if _is_sane_email(em) and _verify_in_source(em, raw_html, visible_text):
            found.add(em)

    # 5) Final filtering to drop placeholders, hashes, image junk, or common no-reply addresses
    cleaned = set()
    for em in found:
        if not em:
            continue
        if re.match(r'^[0-9a-f]{20,}@', em):  # long hex hashes
            continue
        if re.search(r'@\S+\.(jpg|jpeg|png|gif|svg|webp)$', em):
            continue
        lower = em.lower()
        if any(x in lower for x in ['example.com', 'invalid', 'no-reply@', 'noreply@', 'do-not-reply@']):
            continue
        cleaned.add(lower)

    return sorted(cleaned)


# ----------- Get base url ----------
def get_base_url(full_url):
    parsed = urlparse(full_url)
    return parsed.netloc



# ---------- Concurrency wrappers ----------

def parallel_fetch(urls, max_workers=12):
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

# ---------- Streamlit UI ----------
st.title('💻⚡ ScrapeMaster — Focus is the best time-saver. ⏱️')


mode = st.sidebar.selectbox('Choose Function', ['Blog Research', 'Email Finder'])


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
            start_page = st.number_input('Start page (>=2)', min_value=2, value=2)
            end_page = st.number_input('End page', min_value=2, value=5)

    run = st.button('Run Blog Research')

    if run:
        if not base_url:
            st.error('Enter a valid URL')
        elif not url_validator(base_url):
            st.error('Enter a valid URL')
        else:
            t0 = time.time()
            st.info('Fetching pages...')

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

            st.write(f'Found {len(article_links)} article links — now fetching article pages to extract external links...')

            # Fetch article pages and extract external links
            article_html = parallel_fetch(article_links or [], max_workers=concurrency)
            external_domains = set()
            for u, html in article_html.items():
                external_domains.update(get_external_links_from_html(html, avoid_domains, base_url=u))

            # Convert to dataframe (unique)
            df = pd.DataFrame(sorted(external_domains), columns=['client_url'])

            # Simulate all/new behaviour: if 'New Data' we simply remove duplicates in the shown dataframe (virtual)
            if all_or_new == 'New Data':
                df = df.drop_duplicates()

            st.success(f'Done — extracted {len(df)} external client URLs in {time.time()-t0:.1f}s')
            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv_bytes, file_name='clients.csv', mime='text/csv')

elif mode == 'Email Finder':
    st.header('')
    # Common controls
    concurrency = st.sidebar.slider('Max concurrent threads', min_value=5, max_value=100, value=20, step=5)
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
                    candidates_html = parallel_fetch(candidate_pages, max_workers=min(8, concurrency))
                    for ch in candidates_html.values():
                        found_emails.update(extract_emails(ch))

                # fallback: look for mailto links if nothing found
                if len(found_emails) < 1:
                    mailtos = set()
                    try:
                        soup = BeautifulSoup(i, 'html.parser')
                        for a in soup.find_all('a', href=True):
                            if a['href'].lower().startswith('mailto:'):
                                mailtos.add(a['href'].split(':',1)[1])
                    except:
                        pass
                    found_emails.update(mailtos)

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

# Footer / help
st.markdown('---')

