import aiohttp
import argparse
import asyncio
import functools
import hashlib
import json
import io
import re

from asyncio import gather, Semaphore
from pathlib import Path
from urllib.parse import urljoin

from lxml import html
from PIL import Image
from tqdm.auto import tqdm

import os
DD_API_KEY = None
DD_USER_NAME = None

MD5_PREFER_FNAME = True
LONG_SIDE = 768
iqdb_url = "https://danbooru.iqdb.org/"
UA = 'Mozilla/4.08 (compatible; MSIE 6.0; Windows NT 5.1)'

known_img_suffix = set(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'tif'])
md5_m = re.compile(r'(?<![a-zA-Z0-9])([0-9a-fA-F]{32})(?![a-zA-Z0-9])')
pid_m = re.compile(r'(?<![a-zA-Z0-9])(\d+_p\d+)(?![a-zA-Z0-9])')


def async_retry(timeout: float = 1, backoff: float = 2, max_retry: int = 5):
    def inner(func):
        _timeout = timeout

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timeout = _timeout
            for i in range(max_retry):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    funcname = func.__name__
                    print(f'\033[33m Caught {e} in {funcname}, retrying in {timeout} seconds...\033[0m')
                    await asyncio.sleep(timeout)
                timeout *= backoff
            raise Exception('Too many retries')
        return wrapper
    return inner


def img_dirwalk(path: Path):
    for fname in path.iterdir():
        if fname.is_dir():
            yield from img_dirwalk(fname)
        else:
            if fname.suffix[1:] in known_img_suffix:
                yield fname


def img_generator(path: Path, retry_nomatch: bool = False):
    for p in img_dirwalk(path):
        if p.with_suffix('.json').exists():
            continue
        if (not retry_nomatch) and p.with_suffix('.nomatch').exists():
            continue
        yield p


def get_md5(path: Path):
    if MD5_PREFER_FNAME:
        m = md5_m.search(path.name)
        if m:
            return m.group(1).lower()
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_pidstr(path: Path):
    m = pid_m.search(path.name)
    if m:
        return m.group(1)
    return None


@async_retry(max_retry=7, timeout=1, backoff=2)
async def md5_lookup(sess: aiohttp.ClientSession, path: Path, sema: Semaphore):
    md5 = get_md5(path)
    await sema.acquire()
    params = {'tags': f'md5:{md5}'}
    if DD_API_KEY and DD_USER_NAME:
        params['api_key'] = DD_API_KEY
        params['login'] = DD_USER_NAME
    r = await sess.get('https://danbooru.donmai.us/posts.json', params=params)
    r.raise_for_status()
    js = await r.json()
    sema.release()
    if len(js) == 0:
        return None
    return js[0]


@async_retry(max_retry=7, timeout=1, backoff=2)
async def pidstr_lookup(sess: aiohttp.ClientSession, path: Path, sema: Semaphore):
    pid_str = get_pidstr(path)
    if not pid_str:
        return None
    pid, _ = pid_str.split('_')
    await sema.acquire()
    params = {'tags': f'pixiv:{pid}'}
    if DD_API_KEY and DD_USER_NAME:
        params['api_key'] = DD_API_KEY
        params['login'] = DD_USER_NAME
    r = await sess.get('https://danbooru.donmai.us/posts.json', params=params)
    r.raise_for_status()
    js = await r.json()
    sema.release()
    for j in js:
        if pid_str in j['source']:
            return j


@async_retry(max_retry=7, timeout=1, backoff=2)
async def iqdb_lookup(
    sess: aiohttp.ClientSession,
    path: Path,
    sema_iqdb: Semaphore,
    sema_danbooru: Semaphore
):
    img = Image.open(path)
    h, w = img.size
    r = LONG_SIDE / max(h, w)
    if r < 1:
        nh, nw = int(h * r), int(w * r)
        img = img.resize((nw, nh), Image.Resampling.BICUBIC)
    img_bytes = io.BytesIO()
    fmt = 'JPEG' if img.mode == 'RGB' else 'PNG'
    img.save(img_bytes, format=fmt, quality=90)
    files = {'file': img_bytes.getvalue()}
    await sema_iqdb.acquire()
    r = await sess.post(iqdb_url, data=files)
    ht = html.fromstring(await r.text())
    sema_iqdb.release()
    matched = ht.xpath('//parent::table[tr/th/text()="Best match"]')
    if not matched:
        return None
    similarity = matched[0].xpath('.//td[contains(text(), "similarity")]/text()')[0]
    similarity = float(similarity.strip().split('%')[0])
    danbooru_url = matched[0].xpath('.//a')[0].attrib['href']
    danbooru_url = urljoin(iqdb_url, danbooru_url) + '.json'
    await sema_danbooru.acquire()
    r = await sess.get(danbooru_url)
    j = await r.json()
    sema_danbooru.release()
    j['similarity'] = similarity
    j['lookup_method'] = 'iqdb'
    return j


async def lookup(sess: aiohttp.ClientSession, path: Path, sema_iqdb: Semaphore, sema_danbooru: Semaphore):
    j = await pidstr_lookup(sess, path, sema_danbooru)
    if j:
        return j
    j = await md5_lookup(sess, path, sema_danbooru)
    if j:
        return j
    j = await iqdb_lookup(sess, path, sema_iqdb, sema_danbooru)
    if j:
        return j
    return None


async def lookup_and_save(
    sess: aiohttp.ClientSession,
    path: Path,
    sema_iqdb: Semaphore,
    sema_danbooru: Semaphore,
    pbar: tqdm | None = None
):
    j = await lookup(sess, path, sema_iqdb, sema_danbooru)
    if j:
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(j, f, indent=2)
    else:
        with open(path.with_suffix('.nomatch'), 'w') as f:
            pass
    if pbar is not None:
        pbar.update(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=Path)
    parser.add_argument('--retry-nomatch', action='store_true')
    parser.add_argument('--md5-prefer-fname', action='store_false')
    parser.add_argument('--danbooru-concurrency', type=int, default=2)
    parser.add_argument('--iqdb-concurrency', type=int, default=2)
    parser.add_argument('--iqdb-long-side', type=int, default=768)
    return parser.parse_args()


async def main(path: str | Path, iqdb_concurrency=2, danbooru_concurrency=2, retry_nomatch=False):
    imgs = list(img_generator(Path(path), retry_nomatch=retry_nomatch))
    if not imgs:
        print('All metadata fetched.')
        return
    sema_iqdb = Semaphore(iqdb_concurrency)
    sema_danbooru = Semaphore(danbooru_concurrency)
    sess = aiohttp.ClientSession(headers={'User-Agent': UA})

    pbar = tqdm(total=len(imgs))
    jobs = [lookup_and_save(sess, p, sema_iqdb, sema_danbooru, pbar) for p in imgs]
    await gather(*jobs)
    await sess.close()


if __name__ == '__main__':

    if 'DD_API_KEY' in os.environ and 'DD_USER_NAME' in os.environ:
        DD_API_KEY = os.environ['DD_API_KEY']
        DD_USER_NAME = os.environ['DD_USER_NAME']
        print(f'Found Danbooru API key and user name in environment variables. Logging in as \033[96m{DD_USER_NAME}\033[0m.')
    elif 'DD_API_KEY' in os.environ or 'DD_USER_NAME' in os.environ:
        print('You set one of DD_API_KEY or DD_USER_NAME but not both. Ignoring both.')
    else:
        print('No Danbooru API key and user name found. Continue as anonymous user.')
    args = get_args()
    LONG_SIDE = args.iqdb_long_side
    MD5_PREFER_FNAME = args.md5_prefer_fname
    asyncio.run(main(args.path, args.iqdb_concurrency, args.danbooru_concurrency, args.retry_nomatch))
