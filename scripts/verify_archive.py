#!/usr/bin/env python3
"""Reconcile saved curricula and Markdown navigation without network access."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from crawl_apxml import (
    BASE, CACHE, ROOT, chapter_dir, chapter_url, course_dir, load_json,
    render_html, section_path, reference_urls,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-full', action='store_true')
    args = parser.parse_args()
    catalog = load_json(CACHE / 'catalog.json')
    errors = []
    projects = 0
    chapters = 0
    sections = 0
    full_sections = 0
    pending_sections = []
    empty_source_sections = []
    code_blocks = 0
    markdown_parser = MarkdownIt('commonmark')
    expected_ids = set()
    expected_folders = set()
    if len({course['slug'] for course in catalog}) != len(catalog):
        errors.append('Duplicate course slugs in catalog')
    for index, course in enumerate(catalog, 1):
        path = CACHE / 'courses' / f'{course["slug"]}.json'
        if not path.exists():
            errors.append(f'Missing curriculum: {course["slug"]}')
            continue
        detail = load_json(path)
        if detail['slug'] != course['slug']:
            errors.append(f'Course slug mismatch: {course["slug"]}')
        folder = course_dir(index, detail)
        expected_folders.add(folder)
        if not (folder / 'README.md').is_file():
            errors.append(f'Missing course README: {folder.name}')
        else:
            overview = (folder / 'README.md').read_text(encoding='utf-8')
            description = detail.get('description') or detail.get('short_description', '')
            expected = render_html(description, f'{BASE}/zh/courses/{detail["slug"]}')
            if expected not in overview:
                errors.append(f'Course description mismatch: {folder.name}')
            for field in ['title', 'duration', 'prerequisites']:
                value = detail.get(field)
                if value and str(value) not in overview:
                    errors.append(f'Course field {field} missing: {folder.name}')
            for outcome in detail.get('learning_outcomes', []):
                for field in ['topic', 'description']:
                    if outcome[field] not in overview:
                        errors.append(f'Learning outcome {field} missing: {folder.name}')
        project = detail.get('project')
        if isinstance(project, dict) and project.get('content'):
            projects += 1
            project_file = folder / 'PROJECT.md'
            if not project_file.exists():
                errors.append(f'Missing project: {project_file}')
            elif project['content'].strip().replace('\r\n', '\n') not in project_file.read_text(encoding='utf-8'):
                errors.append(f'Project body mismatch: {project_file}')
        for chapter in detail['chapters']:
            chapters += 1
            cp = chapter_dir(folder, chapter)
            if not (cp / 'README.md').is_file():
                errors.append(f'Missing chapter README: {cp}')
            else:
                introduction = (cp / 'README.md').read_text(encoding='utf-8')
                expected = render_html(chapter.get('content', ''), chapter_url(detail, chapter))
                if expected not in introduction:
                    errors.append(f'Chapter introduction mismatch: {cp}')
            for section in chapter['sections']:
                sections += 1
                if section['id'] in expected_ids:
                    errors.append(f'Duplicate section identity: {section["id"]}')
                expected_ids.add(section['id'])
                sp = section_path(cp, section)
                if not sp.is_file():
                    errors.append(f'Missing section entry: {sp}')
                else:
                    text = sp.read_text(encoding='utf-8')
                    cache = CACHE / 'sections' / f'{section["id"]}.json'
                    if not cache.exists():
                        pending_sections.append(str(sp.relative_to(ROOT)))
                        continue
                    saved = load_json(cache)
                    if saved.get('id') != section['id'] or saved.get('slug') != section['slug']:
                        errors.append(f'Section cache identity mismatch: {sp}')
                        continue
                    if saved.get('title') != section['title']:
                        errors.append(f'Section title mismatch: {sp}')
                    body = saved.get('content')
                    if not isinstance(body, str):
                        errors.append(f'Section cache body missing: {sp}')
                        continue
                    if not body.strip():
                        empty_source_sections.append(str(sp.relative_to(ROOT)))
                    url = chapter_url(detail, chapter) + '/' + section['slug']
                    expected = render_html(body, url)
                    if expected not in text or '\u6b63\u6587\u5f85\u6293\u53d6\u3002' in text:
                        errors.append(f'Section Markdown body mismatch: {sp}')
                    else:
                        full_sections += 1
                    if args.require_full:
                        visible_text = ''.join(
                            child.content for token in markdown_parser.parse(text) if token.type == 'inline'
                            for child in token.children or [] if child.type in {'text', 'code_inline'})
                        for reference in saved.get('references', []):
                            for field in ['title', 'author', 'year', 'journal', 'publisher', 'volume', 'pages', 'doi', 'note']:
                                value = reference.get(field)
                                if value and str(value).replace('\r\n', '\n') not in text and str(value) not in visible_text:
                                    errors.append(f'Reference {reference.get("id")} field {field} missing: {sp}')
                        hrefs = {unquote(child.attrGet('href'))
                                 for token in markdown_parser.parse(text) if token.type == 'inline'
                                 for child in token.children or [] if child.type == 'link_open'}
                        for reference in saved.get('references', []):
                            for href in reference_urls(reference.get('url') or ''):
                                if unquote(href) not in hrefs:
                                    errors.append(f'Reference {reference.get("id")} URL missing: {sp}')
                        soup = BeautifulSoup(body.replace('\r\n', '\n').replace('\r', '\n'), 'html.parser')
                        source_code = Counter(pre.get_text().strip() for pre in soup.find_all('pre') if pre.get_text().strip())
                        saved_code = Counter(token.content.strip() for token in markdown_parser.parse(text) if token.type == 'fence')
                        code_blocks += sum(source_code.values())
                        missing_code = source_code - saved_code
                        if missing_code:
                            errors.append(f'Code block mismatch ({sum(missing_code.values())}): {sp}')
    actual_folders = {folder for folder in ROOT.glob('[0-9][0-9][0-9]-*') if folder.is_dir()}
    if actual_folders != expected_folders:
        errors.append('Course folders differ from catalog')
    actual_ids = {int(path.stem) for path in (CACHE / 'sections').glob('*.json')}
    if actual_ids - expected_ids:
        errors.append(f'Unexpected cached sections: {sorted(actual_ids - expected_ids)}')
    files = list(ROOT.glob('*.md')) + list(ROOT.glob('[0-9][0-9][0-9]-*/**/*.md'))
    local_links = 0
    for file in files:
        text = file.read_text(encoding='utf-8')
        lines = text.splitlines(keepends=True)
        for token in markdown_parser.parse(text):
            if token.type in {'fence', 'code_block'} and token.map:
                start, end = token.map
                lines[start:end] = ['\n'] * (end - start)
        text = ''.join(lines)
        text = re.sub(
            r'(?<!\\)\$\$.*?\$\$|(?<![\\$])\$(?!\$)[^\n$]*?(?<!\\)\$(?!\$)|\\\[.*?\\\]|\\\(.*?\\\)',
            '', text, flags=re.S,
        )
        hrefs = [child.attrGet('href') for token in markdown_parser.parse(text)
                 if token.type == 'inline' for child in token.children or []
                 if child.type == 'link_open']
        for href in hrefs:
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            local_links += 1
            target = (file.parent / unquote(parsed.path)).resolve()
            if not target.exists():
                errors.append(f'Broken local link: {file.relative_to(ROOT)} -> {href}')
    result = {
        'courses': len(catalog),
        'chapters': chapters,
        'projects': projects,
        'sections': sections,
        'full_sections': full_sections,
        'pending_sections': len(pending_sections),
        'empty_source_sections': empty_source_sections,
        'markdown_files': len(files),
        'local_links': local_links,
        'code_blocks_checked': code_blocks,
        'errors': errors,
    }
    if args.require_full and pending_sections:
        errors.append(f'Incomplete section bodies: {len(pending_sections)}')
    (ROOT / '.crawl' / 'verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
