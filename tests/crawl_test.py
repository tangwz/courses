import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
            crawler.main()
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
            crawler.main()
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
            crawler.main()
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
            crawler.main()
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
            crawler.main()
        report = self.assert_remaining_course_completed()
        self.assertIn('sections', report['errors'][0]['error'])
        self.assertEqual(readme.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
