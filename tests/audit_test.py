import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import crawl_apxml as crawler

with patch.dict(sys.modules, {'crawl_apxml': crawler}):
    from scripts import audit_content_apxml as content_audit

with patch.dict(sys.modules, {
    'crawl_apxml': crawler, 'audit_content_apxml': content_audit,
}):
    from scripts import audit_rendered_content as rendered_audit


class AuditScopeTests(unittest.TestCase):
    def test_partial_manifest_is_rejected_before_reading_cached_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'reports').mkdir()
            (root / 'reports' / 'content-manifest.json').write_text(
                json.dumps({'partial': True, 'entries': []}), encoding='utf-8',
            )
            with patch.object(rendered_audit, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Partial content manifests'):
                    rendered_audit.main()
            report_path = root / 'reports' / 'rendered-content-verification.json'
            self.assertFalse(report_path.exists())

    def run_catalog_audit(self, archive_body, source_html, rendered_html, status):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            lesson = dict(id=1, order=1, slug='lesson', title='Lesson')
            chapter = dict(
                number=1, slug='chapter', title='Chapter', sections=[lesson],
            )
            course = dict(slug='course', title='Course', chapters=[chapter])
            crawler.save_json(root / 'reports' / 'content-manifest.json', {
                'partial': False, 'entries': [],
            })
            crawler.save_json(cache / 'catalog.json', [{'slug': 'course'}])
            crawler.save_json(cache / 'courses' / 'course.json', course)
            crawler.save_json(cache / 'sections' / '1.json', {
                'content': source_html,
            })
            page = root / 'dist/courses/course/chapter/lesson/index.html'
            page.parent.mkdir(parents=True)
            page.write_text(
                f'<article id="lesson-content">{rendered_html}</article>',
                encoding='utf-8',
            )
            with (
                patch.object(rendered_audit, 'ROOT', root),
                patch.object(rendered_audit, 'CACHE', cache),
                patch.object(crawler, 'ROOT', root),
            ):
                archive = crawler.section_path(
                    crawler.chapter_dir(crawler.course_dir(1, course), chapter),
                    lesson,
                )
                archive.parent.mkdir(parents=True)
                archive.write_text(
                    '# Lesson\n\nSource\n\nChapter\n\n' + archive_body,
                    encoding='utf-8',
                )
                with self.assertRaises(SystemExit) as result:
                    rendered_audit.main()
            self.assertEqual(result.exception.code, status)
            report_path = root / 'reports' / 'rendered-content-verification.json'
            report = json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(report['lessons'], 1)
            return report

    def test_full_audit_checks_catalog_lessons_missing_from_manifest(self):
        report = self.run_catalog_audit(
            'Expected content', '<p>Expected content</p>',
            '<p>Wrong content</p>', 1,
        )
        self.assertEqual(report['textErrors'][0]['id'], 1)

    def test_rendered_image_alt_text_is_included_in_content_comparison(self):
        source = ('<p>Expected content</p>'
                  '<img src="https://example.com/diagram.png" alt="Diagram">')
        report = self.run_catalog_audit(
            'Expected content\n\n![Diagram](https://example.com/diagram.png)',
            source, source, 0,
        )
        self.assertEqual(report['textErrors'], [])


if __name__ == '__main__':
    unittest.main()
