import json
import os
import tempfile
import unittest

from RealtimeSTT.core.text_formatting import preprocess_output
from RealtimeSTT.core.vocabulary import Vocabulary


class VocabularyTests(unittest.TestCase):
    def test_aliases_become_the_term(self):
        vocab = Vocabulary({"SQL": ["sequel"], "Node.js": ["node js", "no js"]})
        self.assertEqual(
            vocab.apply("We use sequel and node js, no JS too."),
            "We use SQL and Node.js, Node.js too.",
        )

    def test_spelled_out_acronyms(self):
        vocab = Vocabulary(["JSON", "API", "SQL"])
        self.assertEqual(vocab.apply("parse the j s o n via the a.p.i."), "parse the JSON via the API.")
        self.assertEqual(vocab.apply("so is sql and s-q-l. Use S.Q.L."), "so is SQL and SQL. Use SQL.")
        self.assertEqual(vocab.apply("is quite a plan"), "is quite a plan")

    def test_fuzzy_misspellings_and_case(self):
        vocab = Vocabulary(["Kubernetes", "Python", "machine learning"])
        self.assertEqual(
            vocab.apply("Deploy on kubernetis, kubernetes or Kubernetis. Pythn too."),
            "Deploy on Kubernetes, Kubernetes or Kubernetes. Python too.",
        )
        self.assertEqual(vocab.apply("machin lerning rocks"), "machine learning rocks")
        self.assertEqual(vocab.apply("a cube of butter"), "a cube of butter")

    def test_acronyms_are_not_fuzzy_matched(self):
        self.assertEqual(Vocabulary(["JSON"]).apply("Jason called"), "Jason called")
        self.assertEqual(Vocabulary({"JSON": ["jason"]}).apply("Jason called"), "JSON called")

    def test_empty_and_coerce(self):
        self.assertEqual(Vocabulary([]).apply("unchanged"), "unchanged")
        self.assertIsNone(Vocabulary.coerce(None))
        vocab = Vocabulary(["SQL"])
        self.assertIs(Vocabulary.coerce(vocab), vocab)
        self.assertEqual(Vocabulary.coerce(["SQL"]).terms, {"SQL": []})
        self.assertEqual(Vocabulary.coerce({"SQL": ["sequel"]}).terms, {"SQL": ["sequel"]})

    def test_json_round_trip(self):
        vocab = Vocabulary({"SQL": ["sequel"], "Kubernetes": []}, fuzzy_threshold=0.85, min_fuzzy_length=6)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "vocab.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(vocab.to_dict(), handle)
            loaded = Vocabulary.coerce(path)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(["SQL", "JSON"], handle)
            as_list = Vocabulary.from_json(path)
        self.assertEqual(loaded.to_dict(), vocab.to_dict())
        self.assertEqual(as_list.terms, {"SQL": [], "JSON": []})

    def test_preprocess_output_applies_vocabulary(self):
        vocab = Vocabulary({"SQL": ["sequel"]})
        self.assertEqual(preprocess_output("sequel rocks", vocabulary=vocab), "SQL rocks.")
        self.assertEqual(preprocess_output("sequel rocks", preview=True, vocabulary=vocab), "SQL rocks")
        self.assertEqual(preprocess_output("sequel rocks"), "Sequel rocks.")


if __name__ == "__main__":
    unittest.main()
