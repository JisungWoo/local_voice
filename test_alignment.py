import unittest

from alignment import map_alignment, Word


class AlignmentTests(unittest.TestCase):
    def test_preserves_source_punctuation_and_measured_spans(self):
        result = map_alignment("Hello, world!", [
            {"text": " hello", "start": .12, "end": .4},
            {"text": " world.", "start": .5, "end": .9}], 1.2)
        self.assertTrue(result["all_words_match"])
        assert result["timing"] is not None
        self.assertEqual(result["timing"]["words"], [
            {"text": "Hello,", "start": .12, "end": .4},
            {"text": "world!", "start": .5, "end": .9}])

    def test_substitution_requires_listening_review(self):
        result = map_alignment("in groundwater", [
            {"text": "and", "start": .1, "end": .3},
            {"text": "groundwater", "start": .3, "end": 1.1}], 1.2)
        self.assertTrue(result["alignment_usable"])
        self.assertTrue(result["requires_review"])
        self.assertFalse(result["all_words_match"])
        self.assertEqual(result["differences"][0]["heard"], "and")

    def test_missing_or_repeated_word_never_receives_invented_timing(self):
        cases: list[list[Word]] = [[{"text": "hello", "start": .1, "end": .3}],
                      [{"text": word, "start": i*.2, "end": i*.2+.1}
                       for i, word in enumerate(["hello", "hello", "world"])]]
        for heard in cases:
            result = map_alignment("hello world", heard, 2)
            self.assertFalse(result["alignment_usable"])
            self.assertIsNone(result["timing"])

    def test_invalid_boundaries_are_not_repaired_by_guessing(self):
        for start, end in ((.5,.5), (.5,.4), (-.1,.2), (.5,2), (float('nan'),.5)):
            result = map_alignment("hello", [{"text":"hello","start":start,"end":end}], 1)
            self.assertFalse(result["alignment_usable"])

    def test_hyphenated_source_word_joins_measured_boundaries(self):
        result = map_alignment("sand-filled", [
            {"text":"sand","start":.1,"end":.3},
            {"text":"filled","start":.3,"end":.6}], 1)
        self.assertTrue(result["all_words_match"])
        assert result["timing"] is not None
        self.assertEqual(result["timing"]["words"], [{"text":"sand-filled","start":.1,"end":.6}])


if __name__ == "__main__":
    unittest.main()
