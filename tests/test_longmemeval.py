#!/usr/bin/env python3
"""Tests for LongMemEval evaluation harness.

Tests the adapter (ingestion), judge (scoring), and runner (end-to-end).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make eval + hooks importable
_TEST_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TEST_DIR.parent
_EVAL_DIR = _PROJECT_ROOT / "eval"
_HOOKS_DIR = _PROJECT_ROOT / "hooks"
for p in [str(_EVAL_DIR), str(_HOOKS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


class TestParseTimestamp(unittest.TestCase):
    """Test timestamp parsing from LongMemEval date strings."""

    def test_simple_format(self):
        from longmemeval_adapter import parse_timestamp
        ts = parse_timestamp("2023/01/15 10:00")
        self.assertGreater(ts, 0)

    def test_with_day_of_week(self):
        from longmemeval_adapter import parse_timestamp
        ts = parse_timestamp("2023/04/10 (Mon) 23:07")
        self.assertGreater(ts, 0)

    def test_date_only(self):
        from longmemeval_adapter import parse_timestamp
        ts = parse_timestamp("2023/01/15")
        self.assertGreater(ts, 0)

    def test_invalid_fallback(self):
        from longmemeval_adapter import parse_timestamp
        ts = parse_timestamp("not a date")
        self.assertGreater(ts, 0)  # Should return default, not crash


class TestIngestQuestion(unittest.TestCase):
    """Test single-question ingestion into isolated SQLite."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        # Initialize schema
        import db
        db._DB_PATH_OVERRIDE = self.db_path
        conn = db.get_db()
        conn.close()
        from db_memory import ensure_embedding_column
        ensure_embedding_column()

    def tearDown(self):
        import db
        db._DB_PATH_OVERRIDE = None
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_ingest_creates_memories(self):
        from longmemeval_adapter import ingest_question

        question_data = {
            "question_id": "q_001",
            "question_type": "single-session-user",
            "question": "What is the user's favorite color?",
            "answer": "blue",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "My favorite color is blue"},
                    {"role": "assistant", "content": "Got it, blue!", "has_answer": True},
                ]
            ],
            "haystack_session_ids": ["session_1"],
            "haystack_dates": ["2023/01/15 10:00"],
            "answer_session_ids": ["session_1"],
        }

        stats = ingest_question(question_data, self.db_path)
        self.assertEqual(stats["memories_created"], 2)
        self.assertEqual(stats["sessions_processed"], 1)

    def test_ingest_isolated_db(self):
        """Verify memories go to eval DB, not production."""
        from longmemeval_adapter import ingest_question
        import db

        question_data = {
            "question_id": "q_002",
            "question_type": "single-session-user",
            "question": "test",
            "answer": "test",
            "haystack_sessions": [
                [{"role": "user", "content": "test memory"}]
            ],
            "haystack_session_ids": ["session_test"],
            "haystack_dates": ["2023/01/15 10:00"],
            "answer_session_ids": [],
        }

        ingest_question(question_data, self.db_path)

        # Check eval DB has the memory
        db._DB_PATH_OVERRIDE = self.db_path
        conn = db.get_db()
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        db._DB_PATH_OVERRIDE = None

        self.assertEqual(count, 1)

    def test_importance_scoring(self):
        """Answer turns should get higher importance."""
        from longmemeval_adapter import ingest_question
        import db

        question_data = {
            "question_id": "q_003",
            "question_type": "single-session-user",
            "question": "test",
            "answer": "test",
            "haystack_sessions": [
                [
                    {"role": "user", "content": "normal turn"},
                    {"role": "assistant", "content": "answer turn", "has_answer": True},
                ]
            ],
            "haystack_session_ids": ["session_1"],
            "haystack_dates": ["2023/01/15 10:00"],
            "answer_session_ids": ["session_1"],
        }

        ingest_question(question_data, self.db_path)

        db._DB_PATH_OVERRIDE = self.db_path
        conn = db.get_db()
        rows = conn.execute(
            "SELECT content, importance FROM memories ORDER BY created_at"
        ).fetchall()
        conn.close()
        db._DB_PATH_OVERRIDE = None

        # First turn: answer session (importance 6), second: has_answer (importance 8)
        self.assertEqual(rows[0]["importance"], 6)  # answer session but not has_answer
        self.assertEqual(rows[1]["importance"], 8)  # has_answer = True

    def test_empty_session(self):
        """Empty sessions should not crash."""
        from longmemeval_adapter import ingest_question

        question_data = {
            "question_id": "q_004",
            "question_type": "single-session-user",
            "question": "test",
            "answer": "test",
            "haystack_sessions": [[]],
            "haystack_session_ids": ["session_empty"],
            "haystack_dates": ["2023/01/15 10:00"],
            "answer_session_ids": [],
        }

        stats = ingest_question(question_data, self.db_path)
        self.assertEqual(stats["memories_created"], 0)


class TestJudge(unittest.TestCase):
    """Test LLM judge scoring logic."""

    @patch("longmemeval_judge.urllib.request.urlopen")
    def test_judge_correct_answer(self, mock_urlopen):
        from longmemeval_judge import judge_answer

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "yes"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = judge_answer(
            question="What color?",
            answer="blue",
            hypothesis="The color is blue",
            question_type="single-session-user",
        )
        self.assertEqual(result["label"], 1)

    @patch("longmemeval_judge.urllib.request.urlopen")
    def test_judge_wrong_answer(self, mock_urlopen):
        from longmemeval_judge import judge_answer

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "no"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = judge_answer(
            question="What color?",
            answer="blue",
            hypothesis="The color is red",
            question_type="single-session-user",
        )
        self.assertEqual(result["label"], 0)

    @patch("longmemeval_judge.urllib.request.urlopen")
    def test_judge_abstention(self, mock_urlopen):
        from longmemeval_judge import judge_answer

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "yes"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = judge_answer(
            question="What is the user's pet name?",
            answer="",
            hypothesis="I don't have that information.",
            question_type="single-session-user_abs",
        )
        self.assertEqual(result["label"], 1)

    def test_judge_network_error(self):
        """Judge should return label=0 on network error, not crash."""
        from longmemeval_judge import judge_answer

        with patch("longmemeval_judge.urllib.request.urlopen", side_effect=Exception("timeout")):
            result = judge_answer(
                question="test", answer="test",
                hypothesis="test", question_type="single-session-user",
            )
            self.assertEqual(result["label"], 0)
            self.assertIn("ERROR", result["raw_response"])

    @patch("longmemeval_judge.urllib.request.urlopen")
    def test_judge_prompt_selection(self, mock_urlopen):
        """Verify correct judge prompt is selected per question type."""
        from longmemeval_judge import _get_judge_prompt
        from longmemeval_judge import (
            JUDGE_PROMPT_DEFAULT,
            JUDGE_PROMPT_TEMPORAL,
            JUDGE_PROMPT_KNOWLEDGE_UPDATE,
            JUDGE_PROMPT_ABSTENTION,
        )

        self.assertEqual(
            _get_judge_prompt("single-session-user"), JUDGE_PROMPT_DEFAULT
        )
        self.assertEqual(
            _get_judge_prompt("temporal-reasoning"), JUDGE_PROMPT_TEMPORAL
        )
        self.assertEqual(
            _get_judge_prompt("knowledge-update"), JUDGE_PROMPT_KNOWLEDGE_UPDATE
        )
        self.assertEqual(
            _get_judge_prompt("single-session-user_abs"), JUDGE_PROMPT_ABSTENTION
        )


class TestAbilityMapping(unittest.TestCase):
    """Test question type → ability mapping."""

    def test_information_extraction(self):
        from longmemeval_runner import get_ability
        self.assertEqual(get_ability("single-session-user"), "Information Extraction")
        self.assertEqual(get_ability("single-session-assistant"), "Information Extraction")
        self.assertEqual(get_ability("single-session-preference"), "Information Extraction")

    def test_multi_session(self):
        from longmemeval_runner import get_ability
        self.assertEqual(get_ability("multi-session"), "Multi-Session Reasoning")

    def test_knowledge_update(self):
        from longmemeval_runner import get_ability
        self.assertEqual(get_ability("knowledge-update"), "Knowledge Updates")

    def test_temporal(self):
        from longmemeval_runner import get_ability
        self.assertEqual(get_ability("temporal-reasoning"), "Temporal Reasoning")

    def test_abstention(self):
        from longmemeval_runner import get_ability
        self.assertEqual(get_ability("single-session-user_abs"), "Abstention")
        self.assertEqual(get_ability("multi-session_abs"), "Abstention")


class TestFormatContext(unittest.TestCase):
    """Test context formatting for generator."""

    def test_format_empty(self):
        from longmemeval_runner import format_context
        result = format_context([])
        self.assertIn("No relevant", result)

    def test_format_with_hits(self):
        from longmemeval_runner import format_context
        hits = [
            {"content": "My favorite color is blue"},
            {"content": "I like pizza"},
        ]
        result = format_context(hits)
        self.assertIn("blue", result)
        self.assertIn("pizza", result)

    def test_format_truncation(self):
        from longmemeval_runner import format_context
        hits = [{"content": "x" * 5000}]
        result = format_context(hits, max_chars=100)
        self.assertLess(len(result), 200)


class TestDatasetIngestion(unittest.TestCase):
    """Test full dataset ingestion pipeline."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        # Create a small test dataset
        self.test_data = [
            {
                "question_id": "q_test_1",
                "question_type": "single-session-user",
                "question": "What is the test value?",
                "answer": "42",
                "question_date": "2023/06/15 (Thu) 10:00",
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "The test value is 42"},
                        {"role": "assistant", "content": "Noted!", "has_answer": True},
                    ]
                ],
                "haystack_session_ids": ["test_session_1"],
                "haystack_dates": ["2023/06/15 10:00"],
                "answer_session_ids": ["test_session_1"],
            },
            {
                "question_id": "q_test_2",
                "question_type": "knowledge-update",
                "question": "What is the current status?",
                "answer": "completed",
                "question_date": "2023/07/20 (Thu) 14:00",
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "Status is now completed"},
                        {"role": "assistant", "content": "Updated."},
                    ]
                ],
                "haystack_session_ids": ["test_session_2"],
                "haystack_dates": ["2023/07/20 14:00"],
                "answer_session_ids": ["test_session_2"],
            },
        ]
        self.dataset_fd, self.dataset_path = tempfile.mkstemp(suffix=".json")
        with open(self.dataset_path, "w") as f:
            json.dump(self.test_data, f)

    def tearDown(self):
        import db
        db._DB_PATH_OVERRIDE = None
        os.close(self.db_fd)
        os.unlink(self.db_path)
        os.close(self.dataset_fd)
        os.unlink(self.dataset_path)

    def test_ingest_dataset(self):
        from longmemeval_adapter import ingest_dataset

        stats = ingest_dataset(self.dataset_path, self.db_path)
        self.assertEqual(stats["questions"], 2)
        self.assertGreater(stats["memories_created"], 0)

    def test_ingest_deduplicates_sessions(self):
        """Same session appearing in multiple questions should only be ingested once."""
        from longmemeval_adapter import ingest_dataset

        # Add a third question that reuses session_1
        self.test_data.append({
            "question_id": "q_test_3",
            "question_type": "multi-session",
            "question": "test",
            "answer": "test",
            "question_date": "2023/08/01 10:00",
            "haystack_sessions": [
                [{"role": "user", "content": "The test value is 42"}],
            ],
            "haystack_session_ids": ["test_session_1"],  # same as q_test_1
            "haystack_dates": ["2023/06/15 10:00"],
            "answer_session_ids": [],
        })
        with open(self.dataset_path, "w") as f:
            json.dump(self.test_data, f)

        stats = ingest_dataset(self.dataset_path, self.db_path)
        self.assertEqual(stats["unique_sessions"], 2)  # not 3


if __name__ == "__main__":
    unittest.main()
