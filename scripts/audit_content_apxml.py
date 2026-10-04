#!/usr/bin/env python3
"""Compare source semantics with independently parsed archived Markdown."""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import unquote

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from crawl_apxml import (
    CACHE, ROOT, chapter_dir, course_dir, load_json, section_path, source_link,
    chapter_url, reference_urls,
)

PARSER = MarkdownIt('commonmark').enable('table')


def expression_pattern(expression):
    return r'\n(?:[ \t]*>[ \t]*)*[ \t]*'.join(re.escape(line) for line in expression.split('\n'))


def formula_pattern(expression):
    return r'\${1,2}[\s>]*' + expression_pattern(expression) + r'[\s>]*\${1,2}'


def semantic_lexer(expressions):
    expressions = sorted(set(expressions), key=len, reverse=True)
    code = r'(?P<code>(?<![\\`])(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`))'
    alternatives = [f'(?P<math{index}>' + formula_pattern(expression) + ')'
                    for index, expression in enumerate(expressions)]
    pattern = re.compile('|'.join(alternatives + [code]), re.S)

    def mask(text):
        saved = {}
        formulas = Counter()

        def replace(match):
            if match.group('code') is None:
                formulas[expressions[int(match.lastgroup[4:])]] += 1
                return ''
            token = f'APXMLAUDITCODETOKEN{len(saved)}END'
            saved[token] = match.group(0)
            return token

        text = pattern.sub(replace, text)
        for token, code in saved.items():
            text = text.replace(token, code)
        return text, formulas

    return mask


def letters(text):
    return Counter(character for character in text if character.isalnum())


def without_markers(text):
    return re.sub(r'(?m)^[ \t]*\d+\.\s*(?=[^\d\s])', '', text)


def plain_inline(tokens):
    parts = []
    for token in tokens or []:
        if token.type in {'text', 'code_inline'}:
            parts.append(token.content)
        elif token.type in {'softbreak', 'hardbreak'}:
            parts.append('\n')
        elif token.type == 'html_inline':
            parts.append(BeautifulSoup(token.content, 'html.parser').get_text())
        elif token.type == 'image':
            parts.append(token.content)
    return ''.join(parts)


def normalized_inline(tokens):
    text = plain_inline(tokens)
    first = next((token for token in tokens or [] if token.type in {'text', 'code_inline'} and token.content), None)
    return text if first and first.type == 'code_inline' else without_markers(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--section-id', action='append', type=int, default=[])
    args = parser.parse_args()
    selected = set(args.section_id)
    result = {'verified_at': datetime.now(timezone.utc).isoformat(),
              'lessons_checked': 0, 'formulas_checked': 0, 'tables_checked': 0,
              'body_links_checked': 0, 'reference_links_checked': 0,
              'plots_checked': 0, 'plot_errors': [],
              'text_differences': [], 'formula_errors': [], 'table_errors': [],
              'link_errors': [], 'suspicious_source_fields': []}
    catalog = load_json(CACHE / 'catalog.json')
    for index, summary in enumerate(catalog, 1):
        course = load_json(CACHE / 'courses' / f'{summary["slug"]}.json')
        for chapter in course['chapters']:
            for section in chapter['sections']:
                if selected and section['id'] not in selected:
                    continue
                saved = load_json(CACHE / 'sections' / f'{section["id"]}.json')
                file = section_path(chapter_dir(course_dir(index, course), chapter), section)
                text = file.read_text(encoding='utf-8')
                body = text.split('\n', 6)[6]
                if '\n---\n' in body:
                    body = body.rsplit('\n---\n', 1)[0]
                if saved.get('references'):
                    body = body.rsplit('\n## \u53c2\u8003\u8d44\u6599\n', 1)[0]
                source = saved['content'].replace('\r\n', '\n').replace('\r', '\n')
                soup = BeautifulSoup(source, 'html.parser')
                expected_plots = Counter(
                    json.dumps(json.loads(plot['data-plot-definition']), sort_keys=True, ensure_ascii=False)
                    for plot in soup.select('[data-plot-definition]'))
                actual_plots = Counter()
                for token in PARSER.parse(body):
                    if token.type == 'fence' and token.info.strip() == 'json':
                        try:
                            definition = json.loads(token.content)
                        except json.JSONDecodeError:
                            continue
                        actual_plots[json.dumps(definition, sort_keys=True, ensure_ascii=False)] += 1
                result['plots_checked'] += sum(expected_plots.values())
                missing_plots = expected_plots - actual_plots
                if missing_plots:
                    result['plot_errors'].append({'id': saved['id'], 'missing': sum(missing_plots.values())})
                formulas = []
                for katex in soup.select('.katex-display, .katex'):
                    if katex.parent is None or any(
                            {'katex-display', 'katex'} & set(parent.get('class', []))
                            for parent in katex.parents):
                        continue
                    annotation = katex.find('annotation', attrs={'encoding': 'application/x-tex'})
                    if annotation:
                        formulas.append(annotation.get_text().strip())
                        katex.replace_with('')
                mask = semantic_lexer(formulas)
                masked_body, actual_formulas = mask(body)
                for expression, count in Counter(formulas).items():
                    if actual_formulas[expression] < count:
                        result['formula_errors'].append({'id': saved['id'], 'expression': expression,
                                                         'expected': count, 'actual': actual_formulas[expression]})
                result['formulas_checked'] += len(formulas)
                source_tables = []
                for table in soup.find_all('table'):
                    source_tables.append([letters(cell.get_text())
                                          for cell in table.find_all(['th', 'td'])])
                for tag in soup.select('pre, script, style, button'):
                    tag.decompose()
                for image in soup.find_all('img'):
                    image.replace_with(image.get('alt') or '\u56fe\u7247')
                for code in soup.find_all('code'):
                    raw = code.get_text()
                    fence = '`' * (max((len(run) for run in re.findall(r'`+', raw)), default=0) + 1)
                    code.replace_with(fence + raw + fence)
                source_text, _ = mask(soup.get_text())
                expected = letters(without_markers(source_text))
                tokens = PARSER.parse(masked_body)
                actual_text = ''.join(normalized_inline(token.children) for token in tokens if token.type == 'inline'
                                      and not token.content.startswith('[\u4ea4\u4e92\u56fe\u8868\uff1a'))
                actual = letters(actual_text)
                if expected != actual:
                    result['text_differences'].append({
                        'id': saved['id'], 'path': str(file.relative_to(ROOT)),
                        'missing': dict(expected - actual), 'extra': dict(actual - expected),
                    })
                actual_tables = []
                cells = None
                for token in tokens:
                    if token.type == 'table_open':
                        cells = []
                    elif token.type == 'inline' and cells is not None:
                        cells.append(letters(plain_inline(token.children)))
                    elif token.type == 'table_close':
                        actual_tables.append(cells)
                        cells = None
                result['tables_checked'] += len(source_tables)
                remaining = list(actual_tables)
                missing_tables = []
                for table in source_tables:
                    if table in remaining:
                        remaining.remove(table)
                    else:
                        missing_tables.append(table)
                if missing_tables:
                    result['table_errors'].append({'id': saved['id'],
                                                  'source_cells': [len(t) for t in source_tables],
                                                  'saved_cells': [len(t) for t in actual_tables]})
                hrefs = Counter(unquote(child.attrGet('href')) for token in tokens if token.type == 'inline'
                                for child in token.children or [] if child.type == 'link_open')
                url = chapter_url(course, chapter) + '/' + section['slug']
                for anchor in soup.select('a[href]'):
                    href = unquote(source_link(anchor['href'], url))
                    result['body_links_checked'] += 1
                    if hrefs[href] <= 0:
                        result['link_errors'].append({'id': saved['id'], 'kind': 'body', 'url': href})
                    else:
                        hrefs[href] -= 1
                all_hrefs = {unquote(child.attrGet('href')) for token in PARSER.parse(text) if token.type == 'inline'
                             for child in token.children or [] if child.type == 'link_open'}
                for reference in saved.get('references', []):
                    for href in reference_urls(reference.get('url') or ''):
                        result['reference_links_checked'] += 1
                        href = unquote(href)
                        if href not in all_hrefs:
                            result['link_errors'].append({'id': saved['id'], 'kind': 'reference', 'url': href})
                for field, value in saved.items():
                    if isinstance(value, str) and re.fullmatch(r'\$[0-9a-f]+', value):
                        result['suspicious_source_fields'].append({'id': saved['id'], 'field': field, 'value': value})
                result['lessons_checked'] += 1
        print(f'Content {index}/{len(catalog)}: {result["lessons_checked"]} lessons', flush=True)
    result['finished_at'] = datetime.now(timezone.utc).isoformat()
    report_name = 'content-verification-subset.json' if selected else 'content-verification.json'
    (ROOT / '.crawl' / report_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: len(value) if isinstance(value, list) else value
                      for key, value in result.items()}, ensure_ascii=False, indent=2))
    raise SystemExit(any(result[key] for key in [
        'text_differences', 'formula_errors', 'table_errors', 'plot_errors',
        'link_errors', 'suspicious_source_fields',
    ]))


if __name__ == '__main__':
    main()
