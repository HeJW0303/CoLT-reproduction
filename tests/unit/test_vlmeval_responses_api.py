from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[2]
VLMEVAL_ROOT = REPO_ROOT / "Evaluation" / "VLMEvalKit"
sys.path.insert(0, str(VLMEVAL_ROOT))

from vlmeval.api.gpt import OpenAIWrapper  # noqa: E402


class OpenAIResponsesTests(unittest.TestCase):
    def response(self, status_code: int, payload: dict) -> Mock:
        response = Mock()
        response.status_code = status_code
        response.text = json.dumps(payload)
        return response

    @patch("vlmeval.api.gpt.requests.post")
    def test_chat_completions_payload_is_unchanged(self, post: Mock) -> None:
        post.return_value = self.response(
            200, {"choices": [{"message": {"content": "chat answer"}}]}
        )
        wrapper = OpenAIWrapper(
            model="gpt-4o",
            key="sk-test",
            api_base="https://example.test/v1/chat/completions",
            verbose=False,
        )

        code, answer, _ = wrapper.generate_inner([{"type": "text", "value": "hello"}])

        self.assertEqual((code, answer), (0, "chat answer"))
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["messages"][0]["content"][0], {"type": "text", "text": "hello"})
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(payload["temperature"], 0)
        self.assertNotIn("input", payload)

    @patch("vlmeval.api.gpt.requests.post")
    def test_responses_payload_and_top_level_output_text(self, post: Mock) -> None:
        post.return_value = self.response(200, {"output_text": "response answer"})
        wrapper = OpenAIWrapper(
            model="gpt-5.6-luna",
            key="sk-test",
            api_base="https://example.test/v1/responses",
            wire_api="responses",
            reasoning_effort="medium",
            responses_max_output_tokens=4096,
            verbose=False,
        )

        code, answer, _ = wrapper.generate_inner(
            [{"type": "text", "value": "judge this"}], temperature=1.0
        )

        self.assertEqual((code, answer), (0, "response answer"))
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "gpt-5.6-luna")
        self.assertEqual(
            payload["input"],
            [{"role": "user", "content": [{"type": "input_text", "text": "judge this"}]}],
        )
        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertEqual(payload["max_output_tokens"], 4096)
        self.assertNotIn("messages", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("n", payload)

    @patch("vlmeval.api.gpt.requests.post")
    def test_deepseek_chat_completions_payload(self, post: Mock) -> None:
        post.return_value = self.response(
            200, {"choices": [{"message": {"content": "1"}}]}
        )
        wrapper = OpenAIWrapper(
            model="deepseek-v4-flash",
            key="sk-test",
            api_base="https://api.deepseek.com/chat/completions",
            wire_api="chat_completions",
            reasoning_effort="low",
            thinking={"type": "disabled"},
            verbose=False,
        )

        answer = wrapper.generate([{"type": "text", "value": "judge this"}])

        self.assertEqual(answer, "1")
        payload = json.loads(post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "deepseek-v4-flash")
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["messages"][0]["content"][0], {"type": "text", "text": "judge this"})
        self.assertNotIn("input", payload)

    @patch("vlmeval.api.gpt.requests.post")
    def test_responses_structured_output_is_aggregated(self, post: Mock) -> None:
        post.return_value = self.response(
            200,
            {
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "part one"},
                            {"type": "output_text", "text": " part two"},
                        ],
                    },
                ]
            },
        )
        wrapper = OpenAIWrapper(
            model="gpt-5.6-luna",
            key="sk-test",
            api_base="OFFICIAL",
            wire_api="responses",
            verbose=False,
        )

        code, answer, _ = wrapper.generate_inner([{"type": "text", "value": "hello"}])

        self.assertEqual((code, answer), (0, "part one part two"))
        self.assertEqual(wrapper.api_base, "https://api.openai.com/v1/responses")

    @patch("vlmeval.api.gpt.requests.post")
    def test_malformed_responses_payload_returns_failure_message(self, post: Mock) -> None:
        post.return_value = self.response(200, {"output": [{"type": "reasoning"}]})
        wrapper = OpenAIWrapper(
            model="gpt-5.6-luna",
            key="sk-test",
            api_base="https://example.test/responses",
            wire_api="responses",
            verbose=False,
        )

        code, answer, _ = wrapper.generate_inner([{"type": "text", "value": "hello"}])

        self.assertEqual(code, 0)
        self.assertEqual(answer, wrapper.fail_msg)

    def test_api_key_is_redacted_from_logs(self) -> None:
        secret = "sk-secret-that-must-not-be-logged"
        with self.assertLogs("ChatAPI", level="INFO") as logs:
            OpenAIWrapper(
                model="gpt-5.6-luna",
                key=secret,
                api_base="https://example.test/responses",
                wire_api="responses",
                verbose=False,
            )
        rendered = "\n".join(logs.output)
        self.assertIn("[REDACTED]", rendered)
        self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
