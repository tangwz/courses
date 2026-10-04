import contextlib
import copy
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from markdown_it import MarkdownIt

from scripts import crawl_apxml as crawler


class CurriculumFailureTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = self.root / '.crawl' / 'cache'
        self.enterContext(patch.object(crawler, 'ROOT', self.root))
        self.enterContext(patch.object(crawler, 'CACHE', self.cache))
        self.enterContext(patch('sys.argv', ['crawl_apxml.py', '--phase', 'all']))
        self.output = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.catalog = [
            {'slug': 'first', 'title': 'First course', 'short_description': 'Summary'},
            {
                'slug': 'second',
                'title': 'Second course',
                'short_description': 'Summary',
            },
        ]
        self.detail = {
            **self.catalog[1],
            'chapters': [{
                'number': 1,
                'slug': 'introduction',
                'title': 'Introduction',
                'content': '<p>Chapter content</p>',
                'sections': [{
                    'id': 101, 'order': 1, 'slug': 'lesson', 'title': 'Lesson',
                }],
            }],
        }
        crawler.save_json(self.cache / 'catalog.json', self.catalog)
        crawler.save_json(self.cache / 'sections' / '101.json', {
            'title': 'Lesson', 'content': '<p>Verified lesson</p>', 'references': [],
        })
        self.browser = Mock()
        self.enterContext(patch.object(crawler, 'Browser', return_value=self.browser))

    def preserve_existing_archive(self):
        folder = crawler.course_dir(1, self.catalog[0])
        folder.mkdir(parents=True)
        readme = folder / 'README.md'
        readme.write_text('# Existing complete curriculum\n', encoding='utf-8')
        return readme

    def assert_remaining_course_completed(self):
        report = json.loads((self.root / '.crawl' / 'report.json').read_text())
        self.assertEqual(len(report['errors']), 1)
        self.assertEqual(report['errors'][0]['url'], crawler.BASE + '/zh/courses/first')
        self.assertEqual(report['total_sections'], 1)
        self.assertEqual(report['completed_sections'], 1)
        folder = crawler.course_dir(2, self.detail)
        self.assertTrue((folder / 'README.md').is_file())
        lesson = folder / '01-Introduction' / '01-Lesson.md'
        self.assertIn('Verified lesson', lesson.read_text())
        return report

    def test_http_failure_preserves_existing_archive_and_continues(self):
        readme = self.preserve_existing_archive()
        original = readme.read_bytes()
        self.browser.fetch.side_effect = [
            [{'status': 503, 'error': 'Temporarily unavailable'}],
            [{'status': 200, 'body': 'second'}],
        ]
        with patch.object(crawler, 'extract_course', return_value=self.detail):
            self.assertEqual(crawler.main(), 1)
        report = self.assert_remaining_course_completed()
        self.assertIn('HTTP 503', report['errors'][0]['error'])
        self.assertEqual(readme.read_bytes(), original)
        self.assertFalse((self.cache / 'courses' / 'first.json').exists())
        self.assertEqual(self.browser.fetch.call_count, 2)

    def test_extraction_failure_links_to_source_and_continues(self):
        self.browser.fetch.side_effect = [
            [{'status': 200, 'body': 'invalid'}],
            [{'status': 200, 'body': 'second'}],
        ]
        with patch.object(crawler, 'extract_course', side_effect=[
            ValueError('Curriculum missing'), self.detail,
        ]):
            self.assertEqual(crawler.main(), 1)
        report = self.assert_remaining_course_completed()
        self.assertEqual(report['errors'][0]['error'], 'Curriculum missing')
        first_readme = crawler.course_dir(1, self.catalog[0]) / 'README.md'
        self.assertFalse(first_readme.exists())
        index = (self.root / 'README.md').read_text()
        self.assertIn('[First course](https://apxml.com/zh/courses/first)', index)

    def test_malformed_cache_does_not_enter_successful_details(self):
        readme = self.preserve_existing_archive()
        original = readme.read_bytes()
        crawler.save_json(self.cache / 'courses' / 'first.json', self.catalog[0])
        self.browser.fetch.return_value = [{'status': 200, 'body': 'second'}]
        with patch.object(crawler, 'extract_course', return_value=self.detail):
            self.assertEqual(crawler.main(), 1)
        report = self.assert_remaining_course_completed()
        self.assertIn('chapters', report['errors'][0]['error'])
        self.assertEqual(readme.read_bytes(), original)
        self.assertEqual(self.browser.fetch.call_count, 1)

    def test_render_failure_is_recorded_without_a_second_render_attempt(self):
        first = copy.deepcopy(self.detail)
        first.update(self.catalog[0])
        self.browser.fetch.side_effect = [
            [{'status': 200, 'body': 'first'}],
            [{'status': 200, 'body': 'second'}],
        ]
        render = crawler.render_curriculum

        def render_or_fail(index, course):
            if index == 1:
                raise PermissionError('Cannot write first curriculum')
            return render(index, course)

        with (
            patch.object(crawler, 'extract_course', side_effect=[first, self.detail]),
            patch.object(crawler, 'render_curriculum', side_effect=render_or_fail),
        ):
            self.assertEqual(crawler.main(), 1)
        report = self.assert_remaining_course_completed()
        self.assertIn('Cannot write first curriculum', report['errors'][0]['error'])

    def test_missing_sections_are_rejected_before_rendering(self):
        readme = self.preserve_existing_archive()
        original = readme.read_bytes()
        malformed = copy.deepcopy(self.detail)
        malformed.update(self.catalog[0])
        del malformed['chapters'][0]['sections']
        crawler.save_json(self.cache / 'courses' / 'first.json', malformed)
        self.browser.fetch.return_value = [{'status': 200, 'body': 'second'}]
        with patch.object(crawler, 'extract_course', return_value=self.detail):
            self.assertEqual(crawler.main(), 1)
        report = self.assert_remaining_course_completed()
        self.assertIn('sections', report['errors'][0]['error'])
        self.assertEqual(readme.read_bytes(), original)

    def prepare_cached_sections(self, failure):
        first = {**self.catalog[0], 'chapters': []}
        detail = copy.deepcopy(self.detail)
        detail['chapters'][0]['sections'].insert(0, {
            'id': 100, 'order': 0, 'slug': 'broken', 'title': 'Broken lesson',
        })
        crawler.save_json(self.cache / 'courses' / 'first.json', first)
        crawler.save_json(self.cache / 'courses' / 'second.json', detail)
        broken_cache = self.cache / 'sections' / '100.json'
        if failure == 'json':
            broken_cache.write_text('{broken', encoding='utf-8')
        else:
            crawler.save_json(broken_cache, {})
        return detail, broken_cache

    def verify_cached_section_failure(self, failure):
        detail, _ = self.prepare_cached_sections(failure)
        render = crawler.render_section

        def render_or_fail(task, section):
            if failure == 'write' and task['id'] == 100:
                raise PermissionError('Cannot write cached lesson')
            return render(task, section)

        with (
            patch('sys.argv', ['crawl_apxml.py', '--phase', 'render']),
            patch.object(crawler, 'render_section', side_effect=render_or_fail),
        ):
            self.assertEqual(crawler.main(), 1)
        report = json.loads((self.root / '.crawl' / 'report.json').read_text())
        self.assertEqual(report['total_sections'], 2)
        self.assertEqual(report['completed_sections'], 1)
        self.assertEqual(len(report['errors']), 1)
        self.assertTrue(report['errors'][0]['url'].endswith('/broken'))
        folder = crawler.course_dir(2, detail) / '01-Introduction'
        self.assertIn('Verified lesson', (folder / '01-Lesson.md').read_text())
        self.browser.fetch.assert_not_called()

    def test_malformed_cached_json_does_not_abort_later_sections(self):
        self.verify_cached_section_failure('json')

    def test_invalid_cached_data_does_not_abort_later_sections(self):
        self.verify_cached_section_failure('data')

    def test_cached_render_failure_does_not_abort_later_sections(self):
        self.verify_cached_section_failure('write')

    def verify_online_cache_recovery(self, failure):
        detail, broken_cache = self.prepare_cached_sections(failure)
        recovered = {
            'id': 100, 'slug': 'broken', 'title': 'Broken lesson',
            'content': '<p>Recovered lesson</p>', 'references': [],
        }
        self.browser.fetch.return_value = [{'status': 200, 'body': 'fresh'}]
        with (
            patch('sys.argv', ['crawl_apxml.py', '--phase', 'sections']),
            patch.object(crawler, 'extract_section', return_value=recovered),
        ):
            self.assertEqual(crawler.main(), 0)
        report = json.loads((self.root / '.crawl' / 'report.json').read_text())
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['completed_sections'], 2)
        self.assertEqual(json.loads(broken_cache.read_text()), recovered)
        folder = crawler.course_dir(2, detail) / '01-Introduction'
        recovered_lesson = folder / '00-Broken lesson.md'
        self.assertIn('Recovered lesson', recovered_lesson.read_text())
        self.assertIn('Verified lesson', (folder / '01-Lesson.md').read_text())
        self.assertEqual(self.browser.fetch.call_count, 1)
        fetched_url = self.browser.fetch.call_args.args[0][0]
        self.assertTrue(fetched_url.endswith('/broken'))

    def test_malformed_cache_is_refetched_and_successfully_repaired(self):
        self.verify_online_cache_recovery('json')

    def test_invalid_cache_data_is_refetched_and_successfully_repaired(self):
        self.verify_online_cache_recovery('data')

    def test_failed_cache_recovery_remains_an_error_and_preserves_the_cache(self):
        _, broken_cache = self.prepare_cached_sections('json')
        original = broken_cache.read_bytes()
        self.browser.fetch.return_value = [{'status': 503}]
        with patch('sys.argv', ['crawl_apxml.py', '--phase', 'all']):
            self.assertEqual(crawler.main(), 1)
        report = json.loads((self.root / '.crawl' / 'report.json').read_text())
        self.assertEqual(report['completed_sections'], 1)
        self.assertEqual(len(report['errors']), 2)
        self.assertIn('HTTP 503', report['errors'][-1]['error'])
        self.assertEqual(broken_cache.read_bytes(), original)

    def verify_cli_status(self, invalid):
        first = self.catalog[0] if invalid else {**self.catalog[0], 'chapters': []}
        crawler.save_json(self.cache / 'courses' / 'first.json', first)
        crawler.save_json(self.cache / 'courses' / 'second.json', self.detail)
        script = self.root / 'scripts' / 'crawl_apxml.py'
        script.parent.mkdir()
        shutil.copyfile(crawler.__file__, script)
        result = subprocess.run(
            [sys.executable, str(script), '--phase', 'render'],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 1 if invalid else 0, result.stderr)
        report = json.loads((self.root / '.crawl' / 'report.json').read_text())
        self.assertEqual(bool(report['errors']), invalid)
        self.assertEqual(report['completed_sections'], 1)

    def test_cli_exits_nonzero_after_writing_a_failed_crawl_report(self):
        self.verify_cli_status(True)

    def test_cli_exits_zero_after_a_successful_crawl(self):
        self.verify_cli_status(False)


class ImageConversionTests(unittest.TestCase):
    def test_source_images_preserve_inline_markdown_and_resolved_urls(self):
        body = crawler.render_html(
            '<p><img src="/images/diagram.png" alt="Diagram [example]"></p>',
            'https://apxml.com/zh/courses/first/lesson',
        )
        images = [
            child for token in MarkdownIt('commonmark').parse(body)
            for child in token.children or [] if child.type == 'image'
        ]
        self.assertEqual(len(images), 1)
        alt = ''.join(child.content for child in images[0].children or [])
        self.assertEqual(alt, 'Diagram [example]')
        self.assertEqual(images[0].attrGet('src'),
                         'https://apxml.com/images/diagram.png')

    def test_images_without_sources_do_not_create_broken_markdown(self):
        self.assertEqual(crawler.render_html('<img alt="Missing">'), '')


if __name__ == '__main__':
    unittest.main()
