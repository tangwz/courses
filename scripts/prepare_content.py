#!/usr/bin/env python3
"""Create course-owned Markdown and plot data from the verified archive."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

from crawl_apxml import (
    BASE, CACHE, ROOT, chapter_dir, chapter_url, course_dir, load_json,
    render_html, section_path,
)

DESTINATION = ROOT / 'src' / 'content' / 'courses'


def frontmatter(metadata, body):
    rows = [f'{key}: {json.dumps(value, ensure_ascii=False)}' for key, value in metadata.items()]
    return '---\n' + '\n'.join(rows) + '\n---\n\n' + body.strip() + '\n'


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding='utf-8') != text:
        path.write_text(text, encoding='utf-8')


def lesson_body(path):
    text = path.read_text(encoding='utf-8')
    parts = text.split('\n\n', 3)
    if len(parts) != 4:
        raise ValueError(f'Invalid archive wrapper: {path}')
    body = parts[3]
    return re.sub(r'\n---\n\n\[(?:\u4e0a\u4e00\u8282|\u4e0b\u4e00\u8282)\][^\n]*\n?$', '', body)


def protect_table_pipes(body, corrections=None):
    # Absolute-value expressions must not become GFM column separators.
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if re.match(r'^[ \t>]*\|', line):
            def currency(match):
                if corrections is not None:
                    corrections.append(dict(kind='currency-table-boundary', line=index + 1))
                return '\\$'
            line = re.sub(r'(?<!\\)\$(?=\d[\d,.]*[kKMB]?\s*\|)', currency, line)
            lines[index] = re.sub(
                r'(?<!\\)\$(?!\$)[^\n$]+\$(?!\$)|(`+).*?\1',
                lambda match: re.sub(r'(?<!\\)\|', r'\\vert ' if match[0].startswith('$') else r'\\|', match[0]), line,
            )
    return ''.join(lines)


def protect_math_dollars(body, source_html, corrections):
    # Inline math delimiters cannot contain a literal dollar, even when escaped.
    for annotation in BeautifulSoup(source_html, 'html.parser').select('annotation[encoding="application/x-tex"]'):
        expression = annotation.get_text().strip()
        if '\\$' not in expression:
            continue
        original = '$' + expression + '$'
        if original in body:
            position = body.index(original)
            corrections.append(dict(kind='escaped-dollar-math', line=body[:position].count('\n') + 1))
            body = body.replace(original, '$' + expression.replace('\\$', '\\text{\\textdollar}') + '$')
    return body


def normalize_inline_math(body, corrections):
    from markdown_it import MarkdownIt
    lines = body.splitlines(keepends=True)
    protected = set()
    for token in MarkdownIt('commonmark').parse(body):
        if token.type in {'fence', 'code_block'} and token.map:
            protected.update(range(*token.map))
    pattern = re.compile(r'(?<![\\$])\$(?!\$)((?:\\.|[^$\\\n])*)\$(?!\$)')
    for index, line in enumerate(lines):
        if index in protected:
            continue
        def currency_bold(match):
            corrections.append(dict(kind='currency-bold-encoding', line=index + 1))
            return '**\\$' + match[1] + '**'
        line = re.sub(r'\\textbf\{\$([\d,.]+)\}\$', currency_bold, line)
        def replace(match):
            expression = match[1].replace('\\$', '\\text{\\textdollar}').replace('\x08', '\\b').replace('\x0c', '\\f')
            if expression != match[1]:
                corrections.append(dict(kind='inline-math-encoding', line=index + 1))
            return '$' + expression + '$'
        line = pattern.sub(replace, line)
        if '\x08eta' in line:
            corrections.append(dict(kind='plain-math-control-character', line=index + 1))
            line = line.replace('\x08eta', '$\\beta$')
        lines[index] = line
    return ''.join(lines)


def extract_plots(body, source_html, owner, folder):
    soup = BeautifulSoup(source_html, 'html.parser')
    plots = soup.select('[data-plot-definition]')
    pattern = re.compile(
        r'(?P<prefix>[ \t>]*)(?:\[\u4ea4\u4e92\u56fe\u8868\uff1a[^\n]*\]\([^\n]*\))\n(?:[ \t>]*\n)*'
        r'[ \t>]*<details>\n[ \t>]*<summary>[^\n]*</summary>\n(?:[ \t>]*\n)*'
        r'[ \t>]*(`{3,})json\n.*?\n[ \t>]*\2\n(?:[ \t>]*\n)*[ \t>]*</details>', re.S,
    )
    matches = list(pattern.finditer(body))
    if len(matches) != len(plots):
        raise ValueError(f'Plot coverage mismatch: {owner}: {len(matches)} != {len(plots)}')
    references = []
    replacements = []
    for index, (match, element) in enumerate(zip(matches, plots)):
        definition = json.loads(element['data-plot-definition'])
        reference, alt = save_plot(definition, owner, index, folder)
        references.append(reference)
        replacements.append((match.start(), match.end(), match['prefix'] + f'![{alt}]({reference})'))
    for start, end, replacement in reversed(replacements):
        body = body[:start] + replacement + body[end:]
    return body, references


def save_plot(definition, owner, index, folder):
    if not isinstance(definition, dict) or not isinstance(definition.get('data'), list):
        raise ValueError(f'Invalid plot data: {owner}-{index}')
    filename = f'{owner}-{index}.json'
    title = definition.get('layout', {}).get('title') or 'Interactive chart'
    title = title.get('text', 'Interactive chart') if isinstance(title, dict) else title
    save(folder / 'plots' / filename, json.dumps(definition, ensure_ascii=False, indent=2) + '\n')
    alt = re.sub(r'<[^>]+>', '', str(title)).replace('[', '\\[').replace(']', '\\]')
    return 'plots/' + filename, alt


def extract_project_plots(body, owner, folder):
    from markdown_it import MarkdownIt
    lines = body.splitlines(keepends=True)
    references = []
    replacements = []
    for token in MarkdownIt('commonmark').parse(body):
        if token.type != 'fence' or token.info.strip() != 'plotly':
            continue
        reference, alt = save_plot(json.loads(token.content), owner, len(references), folder)
        references.append(reference)
        start, end = token.map
        prefix = re.match(r'^[ \t>]*', lines[start]).group()
        replacements.append((start, end, prefix + f'![{alt}]({reference})\n'))
    for start, end, replacement in reversed(replacements):
        lines[start:end] = [replacement]
    return ''.join(lines), references


def prune_generated_content(manifest):
    # A failed generation must not prune existing content.
    expected = {ROOT / entry['path'] for entry in manifest}
    for entry in manifest:
        folder = DESTINATION / (ROOT / entry['path']).relative_to(DESTINATION).parts[0]
        expected.update(folder / reference for reference in entry.get('plots', []))
    for path in sorted(DESTINATION.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
        elif path not in expected:
            path.unlink()


def rewrite_links(body, source_file, source_url, local_paths, source_urls):
    # Match only actual Markdown destinations, leaving code examples untouched.
    from markdown_it import MarkdownIt
    lines = body.splitlines(keepends=True)
    protected = set()
    for token in MarkdownIt('commonmark').parse(body):
        if token.type in {'fence', 'code_block'} and token.map:
            protected.update(range(*token.map))
    def replace(match):
        target = match[1]
        parsed = urlsplit(target)
        fragment = '#' + parsed.fragment if parsed.fragment else ''
        canonical = target.split('#')[0].split('?')[0].rstrip('/')
        resolved = None
        if canonical in source_urls:
            resolved = source_urls[canonical]
        elif parsed.scheme == '' and parsed.netloc == '' and not target.startswith(('#', 'plots/')):
            resolved = local_paths.get((source_file.parent / unquote(parsed.path)).resolve())
        return '](' + (resolved + fragment if resolved else target) + ')'
    for index, line in enumerate(lines):
        if index not in protected:
            lines[index] = re.sub(r'\]\(([^\s)]+)\)', replace, line)
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--course', help='Generate one course for the initial reader prototype')
    args = parser.parse_args()
    catalog = load_json(CACHE / 'catalog.json')
    details = {item['slug']: load_json(CACHE / 'courses' / f'{item["slug"]}.json') for item in catalog}
    local_paths = {}
    source_urls = {}
    for index, item in enumerate(catalog, 1):
        course = details[item['slug']]
        archive = course_dir(index, course)
        route = '/courses/' + course['slug'] + '/'
        local_paths[(archive / 'README.md').resolve()] = route
        source_urls[f'{BASE}/zh/courses/{course["slug"]}'] = route
        local_paths[(archive / 'PROJECT.md').resolve()] = route + 'project/'
        for chapter in course['chapters']:
            chapter_route = route + chapter['slug'] + '/'
            cp = chapter_dir(archive, chapter)
            local_paths[(cp / 'README.md').resolve()] = chapter_route
            source_urls[chapter_url(course, chapter)] = chapter_route
            for section in chapter['sections']:
                lesson_route = chapter_route + section['slug'] + '/'
                local_paths[section_path(cp, section).resolve()] = lesson_route
                source_urls[chapter_url(course, chapter) + '/' + section['slug']] = lesson_route
    counts = dict(courses=0, chapters=0, lessons=0, projects=0, plots=0)
    manifest = []
    for index, item in enumerate(catalog, 1):
        course = details[item['slug']]
        if args.course and course['slug'] != args.course:
            continue
        folder = DESTINATION / course['slug']
        archive = course_dir(index, course)
        source_url = f'{BASE}/zh/courses/{course["slug"]}'
        chapters = sorted(course['chapters'], key=lambda ch: ch['number'])
        total = sum(len(ch['sections']) for ch in chapters)
        common = dict(course=course['slug'], sourceUrl=source_url)
        metadata = dict(
            **common, sourceId=course['id'], title=course['title'],
            description=course.get('short_description') or '',
            category=course['category'], level=course['level'], duration=course['duration'],
            order=index, prerequisites=course.get('prerequisites') or '',
            color=course.get('cover_color') or 'purple', chapterCount=len(chapters), lessonCount=total,
            outcomes=course.get('learning_outcomes') or [],
            hasProject=bool((course.get('project') or {}).get('content')),
        )
        body = render_html(course.get('description') or course.get('short_description', ''), source_url)
        body = rewrite_links(body, archive / 'README.md', source_url, local_paths, source_urls)
        save(folder / 'index.md', frontmatter(metadata, body))
        manifest.append(dict(path=str((folder / 'index.md').relative_to(ROOT)), sourceId=course['id'], kind='course'))
        counts['courses'] += 1
        for chapter in chapters:
            cp = chapter_dir(archive, chapter)
            chapter_source = chapter_url(course, chapter)
            meta = dict(
                course=course['slug'], sourceUrl=chapter_source, sourceId=chapter['id'],
                chapter=chapter['slug'], title=chapter['title'], order=chapter['number'],
                description=chapter.get('meta_description') or '', hasQuiz=bool(chapter.get('has_quiz')),
            )
            body = render_html(chapter.get('content') or '', chapter_source)
            body = rewrite_links(body, cp / 'README.md', chapter_source, local_paths, source_urls)
            save(folder / chapter['slug'] / 'index.md', frontmatter(meta, body))
            manifest.append(dict(path=str((folder / chapter['slug'] / 'index.md').relative_to(ROOT)), sourceId=chapter['id'], kind='chapter'))
            counts['chapters'] += 1
            for section in sorted(chapter['sections'], key=lambda s: s['order']):
                saved = load_json(CACHE / 'sections' / f'{section["id"]}.json')
                path = section_path(cp, section)
                body, plots = extract_plots(lesson_body(path), saved['content'], section['id'], folder)
                url = chapter_source + '/' + section['slug']
                corrections = []
                body = protect_math_dollars(body, saved['content'], corrections)
                body = protect_table_pipes(rewrite_links(body, path, url, local_paths, source_urls), corrections)
                body = normalize_inline_math(body, corrections)
                metadata = dict(
                    course=course['slug'], chapter=chapter['slug'], lesson=section['slug'],
                    sourceId=section['id'], sourceUrl=url, title=section['title'],
                    description=section.get('meta_description') or '', order=section['order'], plots=plots,
                    sourceHash=hashlib.sha256(saved['content'].encode()).hexdigest(),
                    sourceCorrections=corrections,
                )
                destination = folder / chapter['slug'] / (section['slug'] + '.md')
                save(destination, frontmatter(metadata, body))
                manifest.append(dict(path=str(destination.relative_to(ROOT)), sourceId=section['id'], kind='lesson', plots=plots))
                counts['lessons'] += 1
                counts['plots'] += len(plots)
        project = course.get('project') or {}
        if project.get('content'):
            body, plots = extract_project_plots(project['content'], f'project-{course["id"]}', folder)
            body = rewrite_links(body, archive / 'PROJECT.md', source_url, local_paths, source_urls)
            save(folder / 'project.md', frontmatter(dict(
                **common, title=project.get('title') or course['title'], sourceId=course['id'],
                description='', order=1, plots=plots,
            ), body))
            manifest.append(dict(path=str((folder / 'project.md').relative_to(ROOT)), sourceId=course['id'], kind='project', plots=plots))
            counts['projects'] += 1
            counts['plots'] += len(plots)
    if args.course and not counts['courses']:
        raise ValueError(f'Unknown course: {args.course}')
    if not args.course:
        prune_generated_content(manifest)
    save(ROOT / 'reports' / 'content-manifest.json', json.dumps(dict(counts=counts, partial=bool(args.course), entries=manifest), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
