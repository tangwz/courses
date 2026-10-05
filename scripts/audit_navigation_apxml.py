#!/usr/bin/env python3
"""Check ordered indexes and adjacent lesson navigation independently."""

import json
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from crawl_apxml import CACHE, ROOT, chapter_dir, course_dir, load_json, section_path

PARSER = MarkdownIt('commonmark')


def local_targets(file):
    targets = []
    for token in PARSER.parse(file.read_text(encoding='utf-8')):
        if token.type != 'inline':
            continue
        for child in token.children or []:
            if child.type != 'link_open':
                continue
            href = urlsplit(child.attrGet('href'))
            if href.scheme or href.netloc or not href.path:
                continue
            targets.append((file.parent / unquote(href.path)).resolve())
    return targets


def main():
    errors = []
    lesson_count = 0
    catalog = load_json(CACHE / 'catalog.json')
    course_indexes = []
    for index, summary in enumerate(catalog, 1):
        course = load_json(CACHE / 'courses' / f'{summary["slug"]}.json')
        folder = course_dir(index, course)
        course_indexes.append(folder / 'README.md')
        expected_index = []
        lessons = []
        for chapter in sorted(course['chapters'], key=lambda value: value['number']):
            chapter_folder = chapter_dir(folder, chapter)
            chapter_index = chapter_folder / 'README.md'
            chapter_lessons = [section_path(chapter_folder, section)
                               for section in sorted(chapter['sections'], key=lambda value: value['order'])]
            expected_index.extend([chapter_index] + chapter_lessons)
            if local_targets(chapter_index) != [folder / 'README.md'] + chapter_lessons:
                errors.append(f'Chapter navigation/order mismatch: {chapter_index.relative_to(ROOT)}')
            lessons.extend(chapter_lessons)
        project = course.get('project')
        if isinstance(project, dict) and project.get('content'):
            expected_index.append(folder / 'PROJECT.md')
        if local_targets(folder / 'README.md') != expected_index:
            errors.append(f'Course navigation/order mismatch: {folder.name}')
        for number, file in enumerate(lessons):
            expected = [file.parent / 'README.md', folder / 'README.md']
            if number:
                expected.append(lessons[number - 1])
            if number + 1 < len(lessons):
                expected.append(lessons[number + 1])
            # Formula subscripts can resemble Markdown links; navigation lives
            # in the fixed header and final footer rather than the lesson body.
            text = file.read_text(encoding='utf-8')
            header = '\n'.join(text.splitlines()[:6])
            footer = text.rsplit('\n---\n', 1)[1] if '\n---\n' in text else ''
            targets = []
            for token in PARSER.parse(header + '\n' + footer):
                if token.type == 'inline':
                    for child in token.children or []:
                        if child.type == 'link_open':
                            href = urlsplit(child.attrGet('href'))
                            if not href.scheme and not href.netloc and href.path:
                                targets.append((file.parent / unquote(href.path)).resolve())
            if targets != expected:
                errors.append(f'Lesson adjacency mismatch: {file.relative_to(ROOT)}')
            lesson_count += 1
    root_indexes = course_indexes + [ROOT / 'CRAWL_STATUS.md']
    if (ROOT / 'WEBSITE.md').is_file():
        root_indexes.insert(0, ROOT / 'WEBSITE.md')
    if local_targets(ROOT / 'README.md') != root_indexes:
        errors.append('Root course index/order mismatch')
    result = {'verified_at': datetime.now(timezone.utc).isoformat(),
              'courses_checked': len(catalog), 'lessons_checked': lesson_count, 'errors': errors}
    (ROOT / '.crawl' / 'navigation-verification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
