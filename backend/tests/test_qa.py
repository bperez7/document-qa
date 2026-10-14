import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qa


class QATests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = self.root / "doc.txt"
        self.document.write_text("Vacation allowance: 20 days.")
        self.state = {"vector_store_id": None, "documents": {}}
        self.client = MagicMock()
        self.client.vector_stores.create.return_value = NS(id="vs_test")
        self.client.files.create.return_value = NS(id="file_test")
        self.client.vector_stores.files.retrieve.return_value = NS(status="completed")
        self.state_patch = patch.object(qa, "STATE", self.root / "state.json")
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def test_reuse_across_runs_and_changed_content(self):
        first = qa.prepare_document(self.client, self.state, self.document)
        loaded = qa.load_state()
        renamed = self.root / "renamed.txt"
        renamed.write_bytes(self.document.read_bytes())
        self.assertEqual(first, qa.prepare_document(self.client, loaded, renamed))
        self.client.files.create.assert_called_once()
        self.client.vector_stores.create.assert_called_once()
        renamed.write_text("New content")
        self.assertNotEqual(first, qa.prepare_document(self.client, loaded, renamed))
        self.assertEqual(self.client.files.create.call_count, 2)

    def test_attachment_poll_and_filter(self):
        self.client.vector_stores.files.retrieve.side_effect = [
            qa.NotFoundError("missing", response=httpx.Response(
                404, request=httpx.Request("GET", "https://example.test")), body=None),
            NS(status="completed")]
        self.client.vector_stores.files.create.return_value = NS(status="in_progress")
        with patch.object(qa.time, "sleep"):
            digest = qa.prepare_document(self.client, self.state, self.document)
        self.assertEqual(self.client.vector_stores.files.create.call_args.kwargs["attributes"],
                         {"sha256": digest})
        qa.ask(self.client, self.state, digest, "How many days?", "test-model")
        arguments = self.client.responses.create.call_args.kwargs
        self.assertEqual(arguments["tools"][0]["filters"]["value"], digest)
        self.assertEqual(arguments["tool_choice"], "required")
        self.assertFalse(arguments["store"])

    def test_failed_ingestion_retains_upload(self):
        self.client.vector_stores.files.retrieve.return_value = NS(status="failed")
        with self.assertRaises(ValueError):
            qa.prepare_document(self.client, self.state, self.document)
        self.assertEqual(len(qa.load_state()["documents"]), 1)
        self.client.vector_stores.files.retrieve.return_value = NS(status="completed")
        qa.prepare_document(self.client, qa.load_state(), self.document)
        self.client.files.create.assert_called_once()

    def test_citations_and_no_citations(self):
        citation = NS(type="file_citation", file_id="file_test", filename="doc.txt", index=12)
        response = NS(output_text="20 days.", output=[NS(type="message", content=[
            NS(type="output_text", annotations=[citation, citation])])])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            qa.display(response)
        self.assertEqual(output.getvalue().count("doc.txt"), 1)
        response.output = []
        with contextlib.redirect_stdout(output):
            qa.display(response)
        self.assertIn("No file citations returned", output.getvalue())

    def test_processing_timeout_can_resume(self):
        self.client.vector_stores.files.retrieve.return_value = NS(status="in_progress")
        with self.assertRaisesRegex(qa.CLIError, "timed out"):
            qa.prepare_document(self.client, self.state, self.document, timeout=0)
        self.client.vector_stores.files.retrieve.return_value = NS(status="completed")
        qa.prepare_document(self.client, qa.load_state(), self.document)
        self.client.files.create.assert_called_once()

    def test_missing_key_does_not_create_client(self):
        with patch.dict(qa.os.environ, {}, clear=True), patch.object(qa, "load_dotenv"), \
                patch.object(sys, "argv", ["qa.py", "cleanup"]), \
                patch.object(qa, "OpenAI") as client, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                qa.main()
        self.assertEqual(result.exception.code, 1)
        client.assert_not_called()

    def test_partial_cleanup_can_resume(self):
        qa.prepare_document(self.client, self.state, self.document)
        self.client.files.delete.side_effect = RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            qa.cleanup(self.client, self.state)
        loaded = qa.load_state()
        self.assertIsNone(loaded["vector_store_id"])
        self.assertEqual(len(loaded["documents"]), 1)
        self.client.files.delete.side_effect = None
        qa.cleanup(self.client, loaded)
        self.client.vector_stores.delete.assert_called_once()
        self.assertFalse(qa.STATE.exists())


if __name__ == "__main__":
    unittest.main()
