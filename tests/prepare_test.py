import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import crawl_apxml as crawler

with patch.dict(sys.modules, {'crawl_apxml': crawler}):
    from scripts import prepare_content as preparer


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cache = self.root / '.crawl' / 'cache'
        self.destination = self.root / 'src' / 'content' / 'courses'
        for module in (crawler, preparer):
            self.enterContext(patch.object(module, 'ROOT', self.root))
            self.enterContext(patch.object(module, 'CACHE', self.cache))
        self.enterContext(patch.object(preparer, 'DESTINATION', self.destination))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.plot = {'data': [{'type': 'scatter', 'x': [1], 'y': [2]}],
                     'layout': {'title': 'Project chart'}}
        self.course = {
            'id': 42, 'slug': 'first', 'title': 'First', 'category': 'Programming',
            'level': 1, 'duration': 2, 'chapters': [],
            'project': {'content': '```plotly\n' + json.dumps(self.plot) + '\n```'},
        }
        second = {**self.course, 'id': 43, 'slug': 'second', 'project': {}}
        crawler.save_json(self.cache / 'catalog.json', [self.course, second])
        for course in (self.course, second):
            path = self.cache / 'courses' / (course['slug'] + '.json')
            crawler.save_json(path, course)

    def run_preparation(self, *args):
        with patch('sys.argv', ['prepare_content.py', *args]):
            preparer.main()
        return json.loads((self.root / 'reports' / 'content-manifest.json').read_text())

    def stale_files(self):
        files = [
            'retired/index.md', 'first/old/index.md', 'first/old/lesson.md',
            'first/plots/retired.json', 'second/old.md',
        ]
        for name in files:
            preparer.save(self.destination / name, 'Old content')
        return [self.destination / name for name in files]

    def test_full_generation_prunes_stale_courses_chapters_lessons_and_plots(self):
        stale = self.stale_files()
        manifest = self.run_preparation()
        self.assertTrue(all(not path.exists() for path in stale))
        self.assertFalse((self.destination / 'retired').exists())
        self.assertTrue((self.destination / 'second' / 'index.md').exists())
        self.assertEqual(manifest['counts']['plots'], 1)
        project = next(e for e in manifest['entries'] if e['kind'] == 'project')
        self.assertEqual(project['plots'], ['plots/project-42-0.json'])
        plot_path = self.destination / 'first' / project['plots'][0]
        self.assertEqual(json.loads(plot_path.read_text()), self.plot)
        body = (self.destination / 'first' / 'project.md').read_text()
        self.assertIn('![Project chart](plots/project-42-0.json)', body)
        self.assertNotIn('```plotly', body)

    def test_partial_generation_preserves_other_courses_and_existing_files(self):
        stale = self.stale_files()
        manifest = self.run_preparation('--course', 'first')
        self.assertTrue(manifest['partial'])
        self.assertTrue(all(path.read_text() == 'Old content' for path in stale))
        self.assertFalse((self.destination / 'second' / 'index.md').exists())

    def test_failed_generation_does_not_prune_existing_files(self):
        stale = self.stale_files()
        malformed = copy.deepcopy(self.course)
        malformed['project']['content'] = '```plotly\n{"data": null}\n```'
        crawler.save_json(self.cache / 'courses' / 'first.json', malformed)
        with self.assertRaisesRegex(ValueError, 'Invalid plot data'):
            self.run_preparation()
        self.assertTrue(all(path.read_text() == 'Old content' for path in stale))

    def test_project_fences_preserve_quotes_examples_and_lesson_plot_names(self):
        folder = self.destination / 'first'
        preparer.save(folder / 'plots' / '42-0.json', 'Lesson plot')
        definition = json.dumps(self.plot)
        example = '````markdown\n```plotly\n{}\n```\n````\n'
        body = '> ~~~plotly\r\n> ' + definition + '\r\n> ~~~\r\n\n' + example
        rendered, references = preparer.extract_project_plots(
            body, 'project-42', folder,
        )
        self.assertEqual(references, ['plots/project-42-0.json'])
        self.assertIn('> ![Project chart](plots/project-42-0.json)', rendered)
        self.assertIn(example, rendered)
        self.assertEqual((folder / 'plots' / '42-0.json').read_text(), 'Lesson plot')


if __name__ == '__main__':
    unittest.main()
