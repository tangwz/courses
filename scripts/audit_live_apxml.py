#!/usr/bin/env python3
"""Refresh public curricula and compare one substantive lesson per course."""

import argparse
import json
from datetime import datetime, timezone

from crawl_apxml import (
    BASE, CACHE, DEFAULT_CLI, ROOT, Browser, chapter_url, extract_course,
    extract_section, load_json, save_json,
)

AUDIT = ROOT / '.crawl' / 'audit-live'
VOLATILE = {
    'svg_icon', 'cover_icon', 'cover_image', 'added_to_roadmap',
    'has_completed', 'has_bookmarked', 'like_count', 'enrollment_count',
}


def stable(value):
    if isinstance(value, dict):
        return {key: stable(child) for key, child in value.items()
                if key not in VOLATILE}
    if isinstance(value, list):
        return [stable(child) for child in value]
    return value


def changed_paths(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        return [path for key in sorted(before.keys() | after.keys())
                for path in changed_paths(before.get(key), after.get(key),
                                          f'{prefix}.{key}'.strip('.'))]
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [prefix + '.length']
        return [path for index, (old, new) in enumerate(zip(before, after))
                for path in changed_paths(old, new, f'{prefix}[{index}]')]
    return [prefix] if before != after else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='apxml-audit')
    parser.add_argument('--cli', default=DEFAULT_CLI)
    args = parser.parse_args()
    browser = Browser(args.cli, args.session, 1)
    archived = load_json(CACHE / 'catalog.json')
    first = browser.fetch([BASE + '/api/courses/?page=1&sort=popularity'])[0]
    if first['status'] != 200:
        raise RuntimeError(f'Catalog HTTP {first["status"]}')
    payload = json.loads(first['body'])
    live = payload['results']
    pages = (payload['count'] + len(live) - 1) // len(live)
    for page in range(2, pages + 1):
        response = browser.fetch([f'{BASE}/api/courses/?page={page}&sort=popularity'])[0]
        if response['status'] != 200:
            raise RuntimeError(f'Catalog page {page}: HTTP {response["status"]}')
        live.extend(json.loads(response['body'])['results'])
    if len({course['slug'] for course in live}) != payload['count']:
        raise ValueError('Live catalog count mismatch')
    save_json(AUDIT / 'catalog.json', live)
    result = {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'source_course_count': payload['count'],
        'missing_courses': sorted({c['slug'] for c in live} - {c['slug'] for c in archived}),
        'archived_courses_not_listed': sorted({c['slug'] for c in archived} - {c['slug'] for c in live}),
        'curricula_checked': 0, 'curriculum_changes': [],
        'lessons_sampled': 0, 'lesson_changes': [], 'errors': [],
    }
    for start in range(0, len(live), 4):
        batch = live[start:start + 4]
        responses = browser.fetch([f'{BASE}/zh/courses/{c["slug"]}' for c in batch])
        for course, response in zip(batch, responses):
            slug = course['slug']
            try:
                snapshot = AUDIT / 'courses' / f'{slug}.json'
                if response['status'] != 200:
                    raise ValueError(f'HTTP {response["status"]}')
                detail = extract_course(response['body'])
                if detail['slug'] != slug:
                    raise ValueError('Curriculum identity mismatch')
                save_json(snapshot, detail)
                old_path = CACHE / 'courses' / f'{slug}.json'
                paths = changed_paths(stable(load_json(old_path)), stable(detail)) if old_path.exists() else ['new_course']
                if paths:
                    result['curriculum_changes'].append({'slug': slug, 'fields': paths})
                result['curricula_checked'] += 1
            except Exception as error:
                result['errors'].append({'url': response['url'], 'error': str(error)})
        save_json(AUDIT / 'verification.json', result)
        print(f'Curricula {result["curricula_checked"]}/{len(live)}', flush=True)
    samples = []
    for course in live:
        path = AUDIT / 'courses' / f'{course["slug"]}.json'
        if not path.exists():
            continue
        detail = load_json(path)
        candidates = []
        for chapter in detail['chapters']:
            for section in chapter['sections']:
                source = CACHE / 'sections' / f'{section["id"]}.json'
                if source.exists():
                    body = load_json(source)['content']
                    score = len(body) + 1500 * body.count('<table') + 1000 * body.count('<pre')
                    candidates.append((score, chapter, section))
        if candidates:
            _, chapter, section = max(candidates, key=lambda item: item[0])
            samples.append({'url': chapter_url(detail, chapter) + '/' + section['slug'],
                            'slug': section['slug'], 'id': section['id']})
    for start in range(0, len(samples), 4):
        batch = samples[start:start + 4]
        responses = browser.fetch([sample['url'] for sample in batch])
        for sample, response in zip(batch, responses):
            try:
                if response['status'] != 200:
                    raise ValueError(f'HTTP {response["status"]}')
                section = extract_section(response['body'], sample['slug'], sample['id'])
                save_json(AUDIT / 'sections' / f'{sample["id"]}.json', section)
                paths = changed_paths(stable(load_json(CACHE / 'sections' / f'{sample["id"]}.json')), stable(section))
                if paths:
                    result['lesson_changes'].append({'id': sample['id'], 'url': sample['url'], 'fields': paths})
                result['lessons_sampled'] += 1
            except Exception as error:
                result['errors'].append({'url': sample['url'], 'error': str(error)})
        save_json(AUDIT / 'verification.json', result)
        print(f'Lessons {result["lessons_sampled"]}/{len(samples)}', flush=True)
    result['finished_at'] = datetime.now(timezone.utc).isoformat()
    save_json(AUDIT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(1 if result['errors'] else 0)


if __name__ == '__main__':
    main()
