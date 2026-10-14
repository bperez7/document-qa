import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from backend import api, qa


class APITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        state_patch = patch.object(qa, "STATE", Path(self.temp.name) / "state.json")
        state_patch.start()
        self.addCleanup(state_patch.stop)
        self.http = TestClient(api.app)
        self.addCleanup(self.http.close)

    def test_list_only_ready_documents(self):
        qa.save_state({"vector_store_id": "vs_test", "documents": {
            "one": {"filename": "ready.txt", "ready": True},
            "two": {"filename": "pending.txt"}}})
        self.assertEqual(self.http.get("/api/documents").json(),
                         [{"id": "one", "filename": "ready.txt"}])

    def test_upload_validation(self):
        for filename, contents, status in [("file.exe", b"data", 400),
                                            ("file.txt", b"", 400),
                                            ("file.txt", b"12345", 413)]:
            with patch.object(api, "MAX_UPLOAD", 4), patch.object(api, "client") as sdk:
                response = self.http.post("/api/documents", files={"file": (filename, contents)})
                self.assertEqual(response.status_code, status)
                sdk.assert_not_called()

    def test_upload_uses_temporary_file_and_shared_state(self):
        paths = []
        def prepare(sdk, state, path):
            paths.append(path)
            self.assertEqual(path.read_bytes(), b"sample")
            self.assertEqual(path.name, "doc.txt")
            state["documents"]["digest"] = {"filename": "doc.txt", "ready": True}
            return "digest"
        with patch.object(api, "client", return_value=MagicMock()), \
                patch.object(qa, "prepare_document", side_effect=prepare):
            response = self.http.post("/api/documents", files={"file": ("../doc.txt", b"sample")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"id": "digest", "filename": "doc.txt"})
        self.assertFalse(paths[0].exists())

    def test_question_returns_answer_and_citations(self):
        qa.save_state({"vector_store_id": "vs_test", "documents": {
            "digest": {"filename": "doc.txt", "ready": True}}})
        citation = NS(type="file_citation", filename="doc.txt", file_id="file_test", index=5)
        response = NS(output_text="Answer", output=[NS(type="message", content=[
            NS(type="output_text", annotations=[citation])])])
        with patch.object(api, "client", return_value=MagicMock()), \
                patch.object(qa, "ask", return_value=response) as ask:
            result = self.http.post("/api/questions", json={"document_id": "digest", "question": "Why?"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["citations"][0]["filename"], "doc.txt")
        self.assertEqual(ask.call_args.args[2], "digest")

    def test_missing_key_is_actionable(self):
        with patch.dict(api.os.environ, {}, clear=True), patch.object(qa, "load_dotenv"):
            result = self.http.post("/api/documents", files={"file": ("doc.txt", b"text")})
        self.assertEqual(result.status_code, 503)
        self.assertIn("OPENAI_API_KEY", result.json()["detail"])

    def test_unknown_document_and_blank_question(self):
        with patch.object(api, "client", return_value=MagicMock()), patch.object(qa, "ask") as ask:
            result = self.http.post("/api/questions", json={"document_id": "missing", "question": "Why?"})
            self.assertEqual(result.status_code, 404)
            result = self.http.post("/api/questions", json={"document_id": "missing", "question": "   "})
            self.assertEqual(result.status_code, 400)
            ask.assert_not_called()

    def test_api_errors_do_not_expose_raw_details(self):
        import httpx
        from openai import BadRequestError
        error = BadRequestError("private upstream details", response=httpx.Response(
            400, request=httpx.Request("POST", "https://example.test")), body=None)
        with patch.object(api, "client", return_value=MagicMock()), \
                patch.object(qa, "prepare_document", side_effect=error):
            result = self.http.post("/api/documents", files={"file": ("doc.txt", b"sample")})
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("private upstream details", result.text)


if __name__ == "__main__":
    unittest.main()
