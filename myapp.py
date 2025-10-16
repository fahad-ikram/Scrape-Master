import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, random

st.set_page_config(page_title="ScrapeMaster (Web)", layout="wide")

# ---------- Helpers (all in-memory, no files) ----------
USELESS_SITES = set([
    'youtube.com','facebook.com','instagram.com','twitter.com','linkedin.com','tiktok.com',
    'pinterest.com','snapchat.com','google.com','unsplash.com','.gov','freepik.com','pexels.com',
    'pixabay.com','reddit.com','whatsapp.com','telegram.org','tumblr.com','discord.com','vimeo.com',
    'x.com','linkedin.com','bsky.app','threads.com'
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
    if not base_url.endswith('/'):
        base_url = base_url + '/'
    return [f"{base_url}page/{i}/" for i in range(start, end+1)]


def generate_user_agents():
    operating_systems = [
    "Windows NT 10.0; Win64; x64",
    "Windows NT 6.1; Win64; x64",
    "Macintosh; Intel Mac OS X 10_15_4",
    "X11; Ubuntu; Linux x86_64",
    "Windows NT 6.1; WOW64",
    "Windows NT 5.1; Win64; x64",
    "Macintosh; Intel Mac OS X 10_14_6"
    ]
    browsers = [
        "Chrome/{version}.0",
        "Firefox/{version}.0",
        "Safari/{version}.0",
        "Edge/{version}.0",
        "Opera/{version}.0"
    ]
    user_agents_set = set()
    while len(user_agents_set) < 500:
        os = random.choice(operating_systems)
        browser = random.choice(browsers)
        version = random.randint(60, 100)
        if browser == "Safari/{version}.0":
            user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Safari/{version}.0"
        elif browser == "Edge/{version}.0":
            user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version}.0 Safari/537.36 Edge/{version}.0"
        elif browser == "Opera/{version}.0":
            user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version}.0 Safari/537.36 OPR/{version}.0"
        else:
            user_agent = f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) {browser} Safari/537.36"
        user_agents_set.add(user_agent)
    return list(user_agents_set)

def get_random_user_agent():
    return random.choice(generate_user_agents())


# Fetch function (synchronous) with timeout
def fetch_url(url, timeout=20):
    try:
        headers = {'User-Agent': get_random_user_agent()}
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
KEYWORDS = ['contact', 'about', 'privacy', 'policy', 'accessibility','disclaimer', 'author', 'terms', 'write', 'advertise','team','impressum','legal','conditions','cookies','support','help','faq','customer','service','press','media','about-us','contact-us','get-in-touch','reach-us','who-we-are','our-story']

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



def extract_emails_from_html(html):
    """
    Extracts all valid emails from both text and <a href="mailto:..."> links.
    Filters out junk, duplicates, and placeholder emails.
    """
    soup = BeautifulSoup(html or '', 'html.parser')
    found = set()

    # 1️⃣ Extract from plain text using regex
    text = soup.get_text(separator=' ', strip=True)
    found.update(
        re.findall(
            r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',
            text
        )
    )

    # 2️⃣ Extract from <a href="mailto:...">
    for a in soup.find_all('a', href=True):
        href = a['href'].strip().lower()
        if href.startswith('mailto:'):
            email = href.split(':', 1)[1].split('?')[0]  # handle mailto:abc@abc.com?subject
            found.add(email)

    # 3️⃣ Clean invalid or placeholder emails
    valid_emails = set()
    for em in found:
        if re.match(r'^[0-9a-f]{20,}@', em):  # random hash emails like sentry.io
            continue
        if re.search(r'@\S+\.(jpg|jpeg|png|gif|svg|webp)$', em):  # image-based placeholders
            continue
        if any(x in em for x in ['example.com', 'invalid', 'no-reply@', 'noreply@']):
            continue
        valid_emails.add(em)

    return valid_emails






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

# Common controls
concurrency = st.sidebar.slider('Max concurrent threads', min_value=20, max_value=200, value=20, step=5)

if mode == 'Blog Research':
    st.header('')

    col1, col2 = st.columns(2)
    with col1:
        base_url = st.text_input('Website URL (e.g. https://example.com)')
        page_mode = st.radio('Pages', ['Single Page', 'Multiple Pages'])
    with col2:
        admin_upload = st.file_uploader('Optional: Upload admin list (.txt or .csv)', type=['txt','csv'])
        all_or_new = st.radio('Add all data or just new entries (virtual behaviour)', ['All Data', 'New Data'])
        if page_mode == 'Multiple Pages':
            start_page = st.number_input('Start page (>=2)', min_value=2, value=2)
            end_page = st.number_input('End page', min_value=2, value=5)

    run = st.button('Run Blog Research')

    if run:
        if not base_url or not url_validator(base_url):
            st.error('Enter a valid URL')
        else:
            t0 = time.time()
            st.info('Fetching pages...')

            # prepare admin avoid list
            avoid_domains = set(USELESS_SITES)
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

            for site, html in homepage_html.items():
                found_emails = set()
                # extract emails from homepage
                found_emails.update(extract_emails_from_html(html))

                if extract_from_candidates:
                    candidate_pages = find_candidate_pages_for_emails(html, base_url=site)
                    candidate_pages = list(candidate_pages)[:6]
                    candidates_html = parallel_fetch(candidate_pages, max_workers=min(8, concurrency))
                    for ch in candidates_html.values():
                        found_emails.update(extract_emails_from_html(ch))

                # fallback: look for mailto links if nothing found
                if len(found_emails) < 1:
                    mailtos = set()
                    try:
                        soup = BeautifulSoup(html, 'html.parser')
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
                        rows.append({'site': site, 'email': em})
                else:
                    rows.append({'site': site, 'email': ''})

            # make a DataFrame with one email per row
            df = pd.DataFrame(rows)

            # Sort so sites with emails appear first
            df['has_email'] = df['email'].apply(lambda x: bool(x.strip()))
            df = df.sort_values(by='has_email', ascending=False).drop(columns=['has_email']).reset_index(drop=True)

            st.success(f"Done — scanned {len(set([r['site'] for r in rows]))} sites in {time.time()-t0:.1f}s")

            st.dataframe(df)

            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button('Download CSV', data=csv_bytes, file_name='emails.csv', mime='text/csv')

# Footer / help
st.markdown('---')
