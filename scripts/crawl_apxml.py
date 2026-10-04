#!/usr/bin/env python3
"""Archive public ApX courses through an isolated Playwright CLI session."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urljoin

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.crawl' / 'cache'
BASE = 'https://apxml.com'
CLI_CANDIDATES = sorted(
    (Path.home() / '.npm' / '_npx').glob('*/node_modules/@playwright/cli/playwright-cli.js'),
    key=lambda path: path.stat().st_mtime, reverse=True,
)
DEFAULT_CLI = str(CLI_CANDIDATES[0]) if CLI_CANDIDATES else None


def flight_records(html):
    soup = BeautifulSoup(html, 'html.parser')
    chunks = []
    for script in soup.find_all('script'):
        text = script.string or ''
        if text.startswith('self.__next_f.push('):
            item = json.loads(text[19:-1])
            if item[0] == 1:
                chunks.append(item[1])
    payload = ''.join(chunks).encode('utf-8')
    records = {}
    pos = 0
    while pos < len(payload):
        if payload[pos:pos + 1] == b'\n':
            pos += 1
            continue
        match = re.match(rb'([0-9a-f]*):', payload[pos:])
        if not match:
            raise ValueError(f'Invalid Flight record at byte {pos}')
        key = match[1].decode()
        pos += match.end()
        text_match = re.match(rb'T([0-9a-f]+),', payload[pos:])
        if text_match:
            pos += text_match.end()
            length = int(text_match[1], 16)
            records[key] = payload[pos:pos + length].decode('utf-8')
            pos += length
        else:
            end = payload.find(b'\n', pos)
            end = len(payload) if end < 0 else end
            value = payload[pos:end].decode('utf-8')
            try:
                records[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
            pos = end + 1
    return records


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def resolve(value, records):
    if isinstance(value, str) and re.fullmatch(r'\$[0-9a-f]+', value):
        target = records.get(value[1:])
        return resolve(target, records) if target is not None and target != value else value
    if isinstance(value, dict):
        return {key: resolve(child, records) for key, child in value.items()}
    if isinstance(value, list):
        return [resolve(child, records) for child in value]
    return value


def extract_course(html):
    records = flight_records(html)
    for value in records.values():
        for item in objects(value):
            if 'chapters' in item and 'slug' in item and 'title' in item:
                return resolve(item, records)
    raise ValueError('Course curriculum missing from response')


def safe_name(title):
    title = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '-', title).strip(' .')
    return title[:90] or 'untitled'


def md_link(label, path):
    return f'[{markdown_label(label)}]({quote(str(path), safe="/-._~")})'


def markdown_label(value):
    return str(value).replace('\\', '\\\\').replace('[', '\\[').replace(']', '\\]').replace('`', '\\`')


def markdown_url(value):
    return quote(value, safe=':/?#&=%+@!$,;~-._')


def reference_urls(value):
    parts = re.split(r'(?:,\s*|\s+and\s+)(?=https?://)', value.strip())
    return [source_link(re.sub(r'\s+\[\d+\]$', '', part).strip()) for part in parts if part.strip()]


class Browser:
    def __init__(self, cli, session, delay):
        if not cli or not Path(cli).is_file():
            raise ValueError('Playwright CLI not found; install @playwright/cli or pass --cli PATH')
        self.cli = cli
        self.session = session
        self.delay = delay

    def run(self, code, timeout=180):
        result = subprocess.run(
            ['node', self.cli, '--session', self.session, 'run-code', code, '--raw'],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError((result.stdout + result.stderr)[-1200:])
        return json.loads(result.stdout)

    def fetch(self, urls):
        # Two in-flight reads share one rate gate; recovery navigation stays serial.
        code = r"""async (page) => {
            const args = ARGS;
            const fetchOne = async url => page.evaluate(async url => {
                try {
                    const response = await fetch(url, {
                        headers: {'Accept-Language': 'zh'},
                        signal: AbortSignal.timeout(45000)
                    });
                    return {url, status: response.status, body: await response.text()};
                } catch (error) {return {url, status: 0, error: String(error)};}
            }, url);
            const results = await page.evaluate(async args => {
                const results = new Array(args.urls.length);
                let cursor = 0;
                let gate = Promise.resolve();
                let nextStart = Date.now() + args.delay;
                const rateSlot = () => {
                    const scheduled = gate.then(async () => {
                        await new Promise(resolve => setTimeout(resolve, Math.max(0, nextStart - Date.now())));
                        nextStart = Date.now() + args.delay;
                    });
                    gate = scheduled;
                    return scheduled;
                };
                const worker = async () => {
                    while (cursor < args.urls.length) {
                        const index = cursor++;
                        const url = args.urls[index];
                        await rateSlot();
                        try {
                            const response = await fetch(url, {
                                headers: {'Accept-Language': 'zh'},
                                signal: AbortSignal.timeout(45000)
                            });
                            results[index] = {url, status: response.status, body: await response.text()};
                        } catch (error) {results[index] = {url, status: 0, error: String(error)};}
                    }
                };
                await Promise.all(Array.from({length: Math.min(2, args.urls.length)}, () => worker()));
                return results;
            }, args);
            const challenge = result => result.status === 403 && /Just a moment|challenge-platform|\u8bf7\u7a0d\u5019/.test(result.body);
            for (let index = 0; index < results.length; index++) {
                let result = results[index];
                for (let attempt = 0; attempt < 3 && result.status !== 200; attempt++) {
                    if (![0, 429, 500, 502, 503, 504].includes(result.status) && !challenge(result)) break;
                    await page.waitForTimeout(args.delay + attempt * 2000);
                    try {
                        if (challenge(result)) {
                            await page.goto(result.url, {waitUntil: 'domcontentloaded', timeout: 45000});
                            await page.waitForFunction(() =>
                                Array.from(document.scripts).some(script => script.textContent.startsWith('self.__next_f.push(')),
                                null, {timeout: 45000});
                            await page.waitForLoadState('domcontentloaded');
                            result = {url: result.url, status: 200, body: await page.content()};
                        } else {result = await fetchOne(result.url);}
                    } catch (error) {result = {url: result.url, status: 0, error: String(error)};}
                }
                results[index] = result;
            }
            return results;
        }""".replace('ARGS', json.dumps({'urls': urls, 'delay': int(self.delay * 1000)}))
        return self.run(code, timeout=max(240, len(urls) * 180))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def course_dir(index, course):
    return ROOT / f'{index:03d}-{safe_name(course["title"])}'


def chapter_dir(course_path, chapter):
    return course_path / f'{chapter["number"]:02d}-{safe_name(chapter["title"])}'


def section_path(chapter_path, section):
    return chapter_path / f'{section["order"]:02d}-{safe_name(section["title"])}.md'


def chapter_url(course, chapter):
    return f'{BASE}/zh/courses/{course["slug"]}/chapter-{chapter["number"]}-{chapter["slug"]}'


def source_link(url, source_url=BASE + '/'):
    if not url or url.startswith('#'):
        return url
    hostname = re.match(r'^[a-z0-9][a-z0-9.-]*\.[a-z]{2,63}(?:[/:?#]|$)', url, re.I)
    first_part = re.split(r'[/:?#]', url, maxsplit=1)[0]
    extension = first_part.rsplit('.', 1)[-1].lower()
    file_extensions = {'md', 'html', 'htm', 'pdf', 'ipynb', 'py', 'go', 'ts', 'css',
                       'png', 'jpg', 'jpeg', 'svg', 'webp', 'js', 'txt', 'json', 'csv'}
    if hostname and extension not in file_extensions:
        return 'https://' + url
    return urljoin(source_url, url)


def render_html(content, source_url=BASE + "/"):
    from markdownify import MarkdownConverter

    class Converter(MarkdownConverter):
        def escape(self, text, parent_tags):
            return super().escape(text, parent_tags).replace('`', '\\`').replace('$', '\\$')

        def convert_pre(self, el, text, parent_tags):
            code = el.get_text().strip('\n')
            if not code:
                return ''
            callback = self.options['code_language_callback']
            code_language = (callback(el) if callback else '') or self.options['code_language']
            longest = max((len(run) for run in re.findall(r'`+', code)), default=0)
            fence = '`' * max(3, longest + 1)
            return f'\n\n{fence}{code_language}\n{code}\n{fence}\n\n'

        def convert_img(self, el, text, parent_tags):
            src = el.get('src', '')
            alt = el.get('alt') or '\u56fe\u7247'
            return f'[{markdown_label(alt)}]({markdown_url(src)})' if src else ''

        def convert_input(self, el, text, parent_tags):
            if el.get('type') == 'checkbox':
                return '[x] ' if el.has_attr('checked') else '[ ] '
            return ''

    if not content:
        return ''
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    if not re.search(r'<[A-Za-z][^>]*>', content):
        return content.strip()
    soup = BeautifulSoup(content, 'html.parser')
    formulas = {}
    plots = {}

    for plot in soup.select('[data-plot-definition]'):
        definition = plot['data-plot-definition']
        try:
            parsed = json.loads(definition)
            title = parsed.get('layout', {}).get('title', '')
            if isinstance(title, dict):
                title = title.get('text', '')
            definition = json.dumps(parsed, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, AttributeError):
            title = ''
        title = title or '\u4ea4\u4e92\u56fe\u8868'
        plot_id = plot.get('data-plot-id') or plot.get('id') or ''
        url = source_url + ('#' + plot_id if plot_id else '')
        token = f'APXMLPLOTTOKEN{len(plots)}END'
        fence = '`' * max(3, max((len(run) for run in re.findall(r'`+', definition)), default=0) + 1)
        plots[token] = (
            f'\n\n[\u4ea4\u4e92\u56fe\u8868\uff1a{markdown_label(title)}]({markdown_url(url)})\n\n'
            '<details>\n<summary>\u56fe\u8868\u6570\u636e\uff08JSON\uff09</summary>\n\n'
            f'{fence}json\n{definition}\n{fence}\n\n</details>\n\n'
        )
        paragraph = soup.new_tag('p')
        paragraph.string = token
        plot.replace_with(paragraph)

    def store_formula(expression, display):
        token = f'APXMLMATHTOKEN{len(formulas)}END'
        expression = expression.strip()
        display = display or '\n' in expression
        formulas[token] = '\n$$\n' + expression + '\n$$\n' if display else '$' + expression + '$'
        return token
    block_tags = {'p', 'div', 'section', 'ul', 'ol', 'li', 'table', 'blockquote',
                  'pre', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    # Paragraph boundaries keep inline text after block containers separate.
    for block in soup.find_all(sorted(block_tags - {'p', 'li', 'hr'})):
        following = []
        for sibling in block.next_siblings:
            if getattr(sibling, 'name', None) in block_tags:
                break
            following.append(sibling)
        if any(node.get_text().strip() for node in following):
            paragraph = soup.new_tag('p')
            block.insert_after(paragraph)
            for node in following:
                paragraph.append(node.extract())
    # Use the source formula once instead of duplicating KaTeX visual glyphs.
    for katex in soup.select('.katex-display, .katex'):
        if katex.parent is None or any(
                {'katex-display', 'katex'} & set(parent.get('class', []))
                for parent in katex.parents):
            continue
        annotation = katex.find('annotation', attrs={'encoding': 'application/x-tex'})
        if annotation:
            display = 'katex-display' in katex.get('class', [])
            katex.replace_with(store_formula(annotation.get_text(), display))
    for tag in soup.select('script, style, button'):
        tag.decompose()
    for a in soup.select('a[href]'):
        if not a['href'].startswith('#'):
            a['href'] = markdown_url(source_link(a['href'], source_url))
    for image in soup.select('img[src]'):
        image['src'] = source_link(image['src'], source_url)
    def language(el):
        code = el.find('code')
        classes = (code or el).get('class', [])
        return next((c[9:] for c in classes if c.startswith('language-')), '')
    pattern = re.compile(r'\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|(?<!\$)\$(?!\$)([^$\n]+)\$(?!\$)', re.S)
    def protect_formula(match):
        expression = next(group for group in match.groups() if group is not None)
        display = match.group(1) is not None or match.group(3) is not None
        if match.group(4) is not None and match.end() < len(match.string) and match.string[match.end()].isdigit():
            return match.group(0)
        return store_formula(expression, display)
    for node in list(soup.find_all(string=True)):
        if not node.find_parent(['pre', 'code']):
            protected = pattern.sub(protect_formula, str(node))
            if protected != str(node):
                node.replace_with(protected)
    result = Converter(heading_style='ATX', bullets='-', code_language_callback=language).convert_soup(soup)

    def restore_block(token, value):
        nonlocal result
        position = result.find(token)
        if position < 0:
            return
        prefix = result[result.rfind('\n', 0, position) + 1:position]
        context = re.match(r'^(?:[ \t]*>[ \t]?)*[ \t]*', prefix).group()
        marker = re.match(r'(?:[-+*]|\d+\.)[ \t]+', prefix[len(context):])
        if marker:
            context += ' ' * len(marker.group())
        result = result.replace(token, value.replace('\n', '\n' + context), 1)

    for token, formula in formulas.items():
        restore_block(token, formula)
    for token, plot in plots.items():
        restore_block(token, plot)
    return result.strip()


def render_curriculum(index, course):
    path = course_dir(index, course)
    path.mkdir(parents=True, exist_ok=True)
    text = [f'# {course["title"]}', '', f'\u6765\u6e90\uff1a[{course["title"]}]({BASE}/zh/courses/{course["slug"]})', '',
            render_html(course.get('description') or course.get('short_description', ''), f'{BASE}/zh/courses/{course["slug"]}'), '',
            f'\u9884\u8ba1\u5b66\u65f6\uff1a{course.get("duration", "\u672a\u63d0\u4f9b")} \u5c0f\u65f6', '',
            f'\u5148\u4fee\u8981\u6c42\uff1a{course.get("prerequisites", "\u672a\u63d0\u4f9b")}', '',
            '## \u8bfe\u7a0b\u76ee\u5f55', '']
    for chapter in sorted(course.get('chapters', []), key=lambda c: c['number']):
        cp = chapter_dir(path, chapter)
        cp.mkdir(parents=True, exist_ok=True)
        text.extend([f'### {chapter["number"]}. {md_link(chapter["title"], cp.relative_to(path) / "README.md")}', ''])
        ct = [f'# \u7b2c {chapter["number"]} \u7ae0\uff1a{chapter["title"]}', '',
              f'\u6765\u6e90\uff1a[\u539f\u7ae0\u8282]({chapter_url(course, chapter)})', '',
              md_link('\u8fd4\u56de\u8bfe\u7a0b\u76ee\u5f55', Path('..') / 'README.md'), '',
              render_html(chapter.get('content', ''), chapter_url(course, chapter)), '', '## \u5c0f\u8282', '']
        for section in sorted(chapter.get('sections', []), key=lambda s: s['order']):
            sp = section_path(cp, section)
            text.append(f'- {section["order"]}. {md_link(section["title"], sp.relative_to(path))}')
            ct.append(f'- {section["order"]}. {md_link(section["title"], sp.name)}')
            if not sp.exists() or '\u6b63\u6587\u5f85\u6293\u53d6\u3002' in sp.read_text(encoding='utf-8'):
                summary = section.get('meta_description') or ''
                sp.write_text(f'# {section["title"]}\n\n\u6765\u6e90\uff1a[\u539f\u6587]({chapter_url(course, chapter)}/{section["slug"]})\n\n[\u8fd4\u56de\u7ae0\u8282\u76ee\u5f55](README.md) \u00b7 [\u8fd4\u56de\u8bfe\u7a0b\u76ee\u5f55](../README.md)\n\n{summary}\n\n\u6b63\u6587\u5f85\u6293\u53d6\u3002\n', encoding='utf-8')
        if chapter.get('has_quiz'):
            ct.extend(['', f'\u7ae0\u8282\u6d4b\u9a8c\uff1a[\u5728\u7ebf\u6d4b\u9a8c]({chapter_url(course, chapter)}/quiz)'])
            text.append(f'- [\u7ae0\u8282\u6d4b\u9a8c]({chapter_url(course, chapter)}/quiz)')
        text.append('')
        (cp / 'README.md').write_text('\n'.join(ct).strip() + '\n', encoding='utf-8')
    if course.get('learning_outcomes'):
        text.extend(['## \u5b66\u4e60\u76ee\u6807', ''])
        for outcome in course['learning_outcomes']:
            text.append(f'- **{outcome["topic"]}**\uff1a{outcome["description"]}')
        text.append('')
    project = course.get('project')
    if isinstance(project, dict) and project.get('content'):
        project_text = [f'# {project.get("title") or course["title"]}', '',
                        f'\u6765\u6e90\uff1a[\u539f\u8bfe\u7a0b]({BASE}/zh/courses/{course["slug"]})', '',
                        md_link('\u8fd4\u56de\u8bfe\u7a0b\u76ee\u5f55', 'README.md'), '',
                        project['content'].strip(), '']
        (path / 'PROJECT.md').write_text('\n'.join(project_text), encoding='utf-8')
        text.extend(['## \u8bfe\u7a0b\u9879\u76ee', '', '[\u8bfe\u7a0b\u9879\u76ee](PROJECT.md)', ''])
    (path / 'README.md').write_text('\n'.join(text).strip() + '\n', encoding='utf-8')


def write_index(catalog, details, report):
    text = ['# ApX \u4e2d\u6587\u8bfe\u7a0b', '', f'\u6765\u6e90\uff1a[ApX \u8bfe\u7a0b\u76ee\u5f55]({BASE}/zh/courses)', '',
            '\u8bfe\u7a0b\u6309\u6293\u53d6\u65f6\u7f51\u7ad9\u7684\u201c\u6700\u53d7\u6b22\u8fce\u201d\u6392\u5e8f\uff1b\u7ae0\u8282\u4e0e\u5c0f\u8282\u6309\u539f\u8bfe\u7a0b\u7f16\u53f7\u6392\u5217\u3002\u56fe\u7247\u4fdd\u7559\u8fdc\u7a0b\u94fe\u63a5\uff0c\u672a\u4e0b\u8f7d\u3002\u4ea4\u4e92\u56fe\u8868\u4fdd\u7559\u5728\u7ebf\u5165\u53e3\u548c\u539f\u59cb JSON \u6570\u636e\u3002', '',
            f'\u8bfe\u7a0b\uff1a{len(catalog)} \u95e8\uff1b\u5df2\u83b7\u53d6\u76ee\u5f55\uff1a{len(details)} \u95e8\uff1b\u5df2\u4fdd\u5b58\u6b63\u6587\uff1a{report.get("completed_sections", 0)} \u8282\u3002', '',
            '## \u8bfe\u7a0b\u7d22\u5f15', '']
    for index, course in enumerate(catalog, 1):
        path = course_dir(index, course)
        detail = details.get(course['slug'])
        suffix = ''
        if detail:
            count = sum(len(ch['sections']) for ch in detail['chapters'])
            suffix = f' \u2014 {len(detail["chapters"])} \u7ae0 / {count} \u8282'
        text.append(f'{index}. {md_link(course["title"], path.relative_to(ROOT) / "README.md")}{suffix}')
    text.extend(['', '## \u6293\u53d6\u8bb0\u5f55', '', '[\u6293\u53d6\u72b6\u6001](CRAWL_STATUS.md)', ''])
    (ROOT / 'README.md').write_text('\n'.join(text), encoding='utf-8')
    status = ['# \u6293\u53d6\u72b6\u6001', '', f'\u8bfe\u7a0b\u603b\u6570\uff1a{len(catalog)}', '', f'\u8bfe\u7a0b\u76ee\u5f55\uff1a{len(details)}', '',
              f'\u6b63\u6587\u603b\u6570\uff1a{report.get("total_sections", 0)}', '', f'\u6b63\u6587\u5df2\u5b8c\u6210\uff1a{report.get("completed_sections", 0)}', '',
              '\u56fe\u7247\u672a\u4e0b\u8f7d\uff1b\u7ae0\u8282\u6d4b\u9a8c\u4fdd\u7559\u5728\u7ebf\u5165\u53e3\u3002', '']
    if report.get('errors'):
        status.extend(['## \u672a\u5b8c\u6210\u9879', ''])
        for error in report['errors']:
            status.append(f'- [{error["url"]}]({error["url"]})\uff1a{error["error"]}')
    audit_path = ROOT / '.crawl' / 'audit-summary.json'
    if audit_path.exists():
        audit = load_json(audit_path)
        status.extend([
            '## \u5185\u5bb9\u590d\u6838', '',
            f'\u590d\u6838\u65f6\u95f4\uff1a{audit["finished_at"]}', '',
            f'\u7ebf\u4e0a\u590d\u6838\uff1a{audit["live_curricula_checked"]} \u95e8\u8bfe\u7a0b\u5927\u7eb2\uff1b\u6bcf\u95e8\u8bfe\u62bd\u67e5\u4e00\u8282\u6b63\u6587\uff0c\u5171 {audit["live_lessons_sampled"]} \u8282\u3002', '',
            f'\u672c\u5730\u5168\u91cf\u590d\u6838\uff1a{audit["lessons_checked"]} \u8282\u6b63\u6587\u3001{audit["code_blocks_checked"]} \u4e2a\u4ee3\u7801\u5757\u3001{audit["formulas_checked"]} \u4e2a\u516c\u5f0f\u3001{audit["tables_checked"]} \u4e2a\u8868\u683c\u548c {audit["plots_checked"]} \u4e2a\u4ea4\u4e92\u56fe\u8868\u6570\u636e\u3002', '',
            f'\u5df2\u66f4\u65b0 {audit["course_descriptions_updated"]} \u95e8\u8bfe\u7a0b\u6982\u89c8\uff1b\u4fee\u590d\u884c\u5185\u8f6c\u4e49\u3001\u516c\u5f0f\u6392\u7248\u548c\u53c2\u8003\u94fe\u63a5\uff1b\u8865\u56de\u56fe\u8868\u6570\u636e\u548c {audit["checkboxes_restored"]} \u4e2a\u590d\u9009\u6846\u3002', '',
            '\u6700\u7ec8\u590d\u6838\u65e0\u7f3a\u9879\u6216\u8f6c\u6362\u5dee\u5f02\u3002\u539f\u8bfe\u7a0b\u6982\u89c8\u7f13\u5b58\u5df2\u5907\u4efd\u5230 `.crawl/diagnostics-before-correction/audit-20261002/`\u3002', '',
            md_link('\u590d\u6838\u6c47\u603b', '.crawl/audit-summary.json') + ' \u00b7 ' +
            md_link('\u6b63\u6587\u590d\u6838', '.crawl/content-verification.json') + ' \u00b7 ' +
            md_link('\u7ebf\u4e0a\u590d\u6838', '.crawl/audit-live/verification.json'), '',
        ])
    (ROOT / 'CRAWL_STATUS.md').write_text('\n'.join(status), encoding='utf-8')
    save_json(ROOT / '.crawl' / 'report.json', report)


def crawl_catalog(browser):
    catalog_path = CACHE / 'catalog.json'
    if catalog_path.exists():
        return load_json(catalog_path)
    first = browser.fetch([f'{BASE}/api/courses/?page=1&sort=popularity'])[0]
    if first['status'] != 200:
        raise RuntimeError(f'Catalog HTTP {first["status"]}')
    payload = json.loads(first['body'])
    expected = payload['count']
    catalog = payload['results']
    pages = (expected + len(catalog) - 1) // len(catalog)
    print(f'Catalog: {expected} courses / {pages} pages', flush=True)
    for page in range(2, pages + 1):
        response = browser.fetch([f'{BASE}/api/courses/?page={page}&sort=popularity'])[0]
        if response['status'] != 200:
            raise RuntimeError(f'Catalog page {page}: HTTP {response["status"]}')
        catalog.extend(json.loads(response['body'])['results'])
    unique = {course['slug']: course for course in catalog}
    if len(unique) != expected:
        raise RuntimeError(f'Catalog count mismatch: expected {expected}, got {len(unique)}')
    # Keep only curriculum metadata; expiring signed image URLs are unnecessary.
    catalog = [{k: v for k, v in c.items() if k not in {'svg_icon', 'cover_icon', 'cover_image', 'added_to_roadmap'}} for c in unique.values()]
    save_json(catalog_path, catalog)
    return catalog


def extract_section(html, slug, expected_id=None):
    records = flight_records(html)
    for value in records.values():
        for item in objects(value):
            if item.get('slug') == slug and 'content' in item and 'references' in item:
                if expected_id is not None and item.get('id') != expected_id:
                    continue
                section = resolve(item, records)
                if not isinstance(section['content'], str):
                    raise ValueError('Section body is not a string')
                if re.fullmatch(r'\$[0-9a-f]+', section['content']):
                    raise ValueError('Section body has an unresolved source reference')
                return section
    raise ValueError('Section body missing from public page')


def render_section(task, section):
    path = task['path']
    text = [f'# {section["title"]}', '', f'\u6765\u6e90\uff1a[\u539f\u6587]({task["url"]})', '',
            md_link('\u8fd4\u56de\u7ae0\u8282\u76ee\u5f55', 'README.md') + ' \u00b7 ' + md_link('\u8fd4\u56de\u8bfe\u7a0b\u76ee\u5f55', '../README.md'), '',
            render_html(section['content'], task['url']), '']
    references = section.get('references') or []
    if references:
        text.extend(['## \u53c2\u8003\u8d44\u6599', ''])
        for reference in sorted(references, key=lambda r: r.get('order', 0)):
            title = reference['title']
            urls = reference_urls(reference.get('url') or '')
            title = markdown_label(title)
            if len(urls) == 1:
                title = f'[{title}]({markdown_url(urls[0])})'
            elif urls:
                title += ' ' + ' '.join(f'[{number}]({markdown_url(url)})' for number, url in enumerate(urls, 1))
            author = reference.get('author', '')
            year = reference.get('year', '')
            text.append(f'- {title} \u2014 {author} ({year})')
            publication = []
            for field, label in [('journal', 'Journal'), ('publisher', 'Publisher'),
                                 ('volume', 'Volume'), ('pages', 'Pages')]:
                if reference.get(field):
                    publication.append(f'{label}: {reference[field]}')
            doi = str(reference.get('doi') or '')
            if doi:
                doi_url = doi if doi.startswith(('https://', 'http://')) else 'https://doi.org/' + quote(doi.removeprefix('doi:'), safe='/')
                publication.append(f'DOI: [{markdown_label(doi)}]({markdown_url(doi_url)})')
            if publication:
                text.append('  ' + '; '.join(publication))
            if reference.get('note'):
                text.append(f'  {reference["note"]}')
        text.append('')
    navigation = []
    for label, other in [('\u4e0a\u4e00\u8282', task.get('previous')), ('\u4e0b\u4e00\u8282', task.get('next'))]:
        if other:
            navigation.append(md_link(label, os.path.relpath(other, path.parent)))
    if navigation:
        text.extend(['---', '', ' \u00b7 '.join(navigation), ''])
    path.write_text('\n'.join(text).strip() + '\n', encoding='utf-8')


def crawl_sections(browser, catalog, details, report, fetch_missing=True):
    tasks = []
    for index, summary in enumerate(catalog, 1):
        course = details.get(summary['slug'])
        if not course:
            continue
        course_tasks = []
        for chapter in sorted(course['chapters'], key=lambda c: c['number']):
            cp = chapter_dir(course_dir(index, course), chapter)
            for section in sorted(chapter['sections'], key=lambda s: s['order']):
                course_tasks.append({
                    'url': chapter_url(course, chapter) + '/' + section['slug'],
                    'id': section['id'], 'slug': section['slug'], 'path': section_path(cp, section),
                    'cache': CACHE / 'sections' / f'{section["id"]}.json',
                })
        for number, task in enumerate(course_tasks):
            task['previous'] = course_tasks[number - 1]['path'] if number else None
            task['next'] = course_tasks[number + 1]['path'] if number + 1 < len(course_tasks) else None
        tasks.extend(course_tasks)
    report['total_sections'] = len(tasks)
    pending = []
    for task in tasks:
        if task['cache'].exists():
            render_section(task, load_json(task['cache']))
            report['completed_sections'] += 1
        else:
            pending.append(task)
    print(f'Sections: {len(tasks)} total, {len(pending)} pending', flush=True)
    if not fetch_missing:
        return
    for start in range(0, len(pending), 4):
        batch = pending[start:start + 4]
        try:
            responses = browser.fetch([task['url'] for task in batch])
        except Exception as error:
            responses = [{'status': 0, 'error': str(error)} for task in batch]
        for task, response in zip(batch, responses):
            try:
                if response['status'] != 200:
                    raise ValueError(f'HTTP {response["status"]}: {response.get("error", "")}')
                section = extract_section(response['body'], task['slug'], task['id'])
                section = {k: v for k, v in section.items() if k not in {'has_completed', 'has_bookmarked', 'like_count'}}
                render_section(task, section)
                save_json(task['cache'], section)
                report['completed_sections'] += 1
            except Exception as error:
                report['errors'].append({'url': task['url'], 'error': str(error)})
                print(f'ERROR {task["url"]}: {error}', flush=True)
        write_index(catalog, details, report)
        print(f'Sections {report["completed_sections"]}/{len(tasks)}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', default=DEFAULT_CLI)
    parser.add_argument('--session', default='apxml')
    parser.add_argument('--delay', type=float, default=0.75)
    parser.add_argument('--phase', choices=['curricula', 'sections', 'all', 'render'], default='all')
    args = parser.parse_args()
    if args.phase != 'render':
        save_json(ROOT / '.crawl' / 'worker.json', {'pid': os.getpid(), 'session': args.session, 'phase': args.phase})
    browser = Browser(args.cli, args.session, args.delay) if args.phase != 'render' else None
    catalog = crawl_catalog(browser)
    details = {}
    report = {'source': BASE + '/zh/courses', 'course_count': len(catalog), 'errors': [], 'completed_sections': 0}
    for index, course in enumerate(catalog, 1):
        cache_path = CACHE / 'courses' / f'{course["slug"]}.json'
        try:
            if cache_path.exists():
                detail = load_json(cache_path)
            else:
                if args.phase == 'render':
                    raise FileNotFoundError(f'Missing curriculum cache: {course["slug"]}')
                response = browser.fetch([f'{BASE}/zh/courses/{course["slug"]}'])[0]
                if response['status'] != 200:
                    raise RuntimeError(f'HTTP {response["status"]}: {response.get("error", "")}')
                detail = extract_course(response['body'])
                detail = {k: v for k, v in detail.items() if k not in {'svg_icon', 'cover_icon', 'cover_image'}}
                save_json(cache_path, detail)
            details[course['slug']] = detail
            render_curriculum(index, detail)
            print(f'Curriculum {index}/{len(catalog)}: {detail["title"]} ({len(detail["chapters"])} chapters)', flush=True)
        except Exception as error:
            report['errors'].append({'url': f'{BASE}/zh/courses/{course["slug"]}', 'error': str(error)})
            render_curriculum(index, course)
            print(f'ERROR {course["slug"]}: {error}', flush=True)
        report['total_sections'] = sum(len(ch['sections']) for c in details.values() for ch in c['chapters'])
        write_index(catalog, details, report)
    if args.phase in {'sections', 'all', 'render'}:
        crawl_sections(browser, catalog, details, report, fetch_missing=args.phase != 'render')
    write_index(catalog, details, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'errors'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
