#!/usr/bin/env python3
"""Check all rendered lesson text against the independently verified archive."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

from audit_content_apxml import PARSER, letters, normalized_inline, semantic_lexer, without_markers
from crawl_apxml import CACHE, ROOT, load_json, chapter_dir, course_dir, section_path


def main():
    report = dict(verifiedAt=datetime.now(timezone.utc).isoformat(), lessons=0, textErrors=[])
    manifest = load_json(ROOT / 'reports' / 'content-manifest.json')
    selected = {entry['sourceId'] for entry in manifest['entries'] if entry['kind'] == 'lesson'}
    catalog = load_json(CACHE / 'catalog.json')
    for index, item in enumerate(catalog, 1):
        course = load_json(CACHE / 'courses' / f'{item["slug"]}.json')
        for chapter in course['chapters']:
            for lesson in chapter['sections']:
                if lesson['id'] not in selected:
                    continue
                archive = section_path(chapter_dir(course_dir(index, course), chapter), lesson)
                body = archive.read_text(encoding='utf-8').split('\n\n', 3)[3]
                body = body.rsplit('\n---\n', 1)[0]
                body = body.rsplit('\n## \u53c2\u8003\u8d44\u6599\n', 1)[0]
                body = re.sub(r'\\textbf\{\$([\d,.]+)\}\$', lambda m: '**\\$' + m[1] + '**', body)
                body = body.replace('\x08eta', '$\\beta$')
                page = ROOT / 'dist' / 'courses' / course['slug'] / chapter['slug'] / lesson['slug'] / 'index.html'
                soup = BeautifulSoup(page.read_text(encoding='utf-8'), 'html.parser')
                article = soup.select_one('#lesson-content')
                rendered_expressions = [a.get_text().strip().replace('\\text{\\textdollar}', '\\$') for a in article.select('annotation[encoding="application/x-tex"]')]
                source_html = BeautifulSoup(load_json(CACHE / 'sections' / f'{lesson["id"]}.json')['content'], 'html.parser')
                source_expressions = [a.get_text().strip() for a in source_html.select('annotation[encoding="application/x-tex"]') if not re.match(r'^\d[\d,.]*[kKMB]?\s+\|\s*$', a.get_text())]
                expressions = rendered_expressions + source_expressions + [re.sub(r'\\vert\s*', '|', expression) for expression in rendered_expressions]
                masked, _ = semantic_lexer(expressions)(body)
                tokens = PARSER.parse(masked)
                expected = letters(''.join(normalized_inline(t.children) for t in tokens if t.type == 'inline' and not t.content.startswith('[\u4ea4\u4e92\u56fe\u8868\uff1a')))
                references = article.find(['h2','h3'], string='\u53c2\u8003\u8d44\u6599')
                if references:
                    for sibling in list(references.next_siblings):
                        sibling.extract()
                    references.decompose()
                for tag in article.select('.katex, .course-plot, pre, script, style'):
                    tag.decompose()
                for block in article.find_all(['h1','h2','h3','h4','h5','h6','p','li']):
                    if block.find(['p','ul','ol','pre','table']):
                        continue
                    if not block.contents or getattr(block.contents[0], 'name', None) != 'code':
                        block.string = without_markers(block.get_text())
                actual = letters(article.get_text())
                if actual != expected:
                    report['textErrors'].append(dict(
                        id=lesson['id'], missing=dict(expected - actual), extra=dict(actual - expected),
                    ))
                report['lessons'] += 1
        print(f'Audit {index}/{len(catalog)}: {report["lessons"]} lessons', flush=True)
    (ROOT / 'reports' / 'rendered-content-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(lessons=report['lessons'], textErrors=len(report['textErrors']))))
    raise SystemExit(bool(report['textErrors']))


if __name__ == '__main__':
    main()
