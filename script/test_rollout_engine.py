"""
test_rollout_engine.py
======================
对 trainer/rollout_engine.py 的单元测试 + 集成测试。

测试分三层：
  1. compute_per_token_logps  —— 纯函数，用 mock 模型验证数学正确性
  2. TorchRolloutEngine       —— 本地推理引擎，用真实 MiniMind 小模型跑
  3. SGLangRolloutEngine      —— 远程 HTTP 引擎，用 unittest.mock 拦截网络请求
  4. create_rollout_engine    —— 工厂函数，验证分发逻辑

运行方式（在项目根目录）：
  python -m pytest script/test_rollout_engine.py -v
  # 或者直接运行：
  python script/test_rollout_engine.py
"""

import sys
import os
import json
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import torch
import torch.nn as nn

# 把项目根目录加进 sys.path，这样才能 import trainer 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trainer.rollout_engine import (
    compute_per_token_logps,
    RolloutResult,
    TorchRolloutEngine,
    SGLangRolloutEngine,
    create_rollout_engine,
)


# ─────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────

def make_fake_tokenizer(vocab_size=100, pad_id=0, eos_id=2):
    """返回一个最小化的假 tokenizer，避免依赖真实词表文件。"""
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.eos_token_id = eos_id
    tok.batch_decode = lambda ids, **kw: ["fake completion"] * len(ids)
    tok.decode = lambda ids, **kw: "fake completion"
    return tok


class TinyLM(nn.Module):
    """
    极简语言模型 stub，足够让 compute_per_token_logps 和
    TorchRolloutEngine 能跑通，不需要真实权重文件。
    """

    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden)
        self.proj = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
        h = self.embed(input_ids)           # B, T, hidden
        logits = self.proj(h)               # B, T, vocab_size
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        # 返回一个像 HuggingFace 输出的简单对象
        out = types.SimpleNamespace(logits=logits)
        return out

    def generate(self, input_ids, attention_mask=None,
                 max_new_tokens=5, do_sample=True, temperature=1.0,
                 num_return_sequences=1, pad_token_id=0, eos_token_id=2):
        B = input_ids.shape[0]
        new_tokens = torch.randint(3, self.vocab_size,
                                   (B * num_return_sequences, max_new_tokens),
                                   device=input_ids.device)
        # 把 prompt 重复 num_return_sequences 次，再拼上新 token
        prompt_rep = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([prompt_rep, new_tokens], dim=1)


# ─────────────────────────────────────────────────────────────
# 测试 1：compute_per_token_logps
# ─────────────────────────────────────────────────────────────

class TestComputePerTokenLogps(unittest.TestCase):

    def setUp(self):
        self.model = TinyLM(vocab_size=50, hidden=16)
        self.B, self.T = 2, 10
        self.input_ids = torch.randint(0, 50, (self.B, self.T))

    def test_output_shape(self):
        """返回形状应该是 (B, logits_to_keep)。"""
        keep = 4
        logps = compute_per_token_logps(self.model, self.input_ids, logits_to_keep=keep)
        self.assertEqual(logps.shape, (self.B, keep),
                         f"期望 ({self.B}, {keep})，实际 {tuple(logps.shape)}")

    def test_values_are_negative(self):
        """log 概率恒 <= 0（概率 <= 1，取 log 为负）。"""
        logps = compute_per_token_logps(self.model, self.input_ids, logits_to_keep=4)
        self.assertTrue((logps <= 0).all(),
                        f"发现正的 log 概率，不合法：{logps}")

    def test_zero_logits_to_keep_returns_empty(self):
        """logits_to_keep=0 时应返回空张量，不报错。"""
        logps = compute_per_token_logps(self.model, self.input_ids, logits_to_keep=0)
        self.assertEqual(logps.shape[1], 0,
                         "logits_to_keep=0 应返回 seq_len=0 的空张量")

    def test_negative_logits_to_keep_returns_empty(self):
        """logits_to_keep<0 时同样返回空张量。"""
        logps = compute_per_token_logps(self.model, self.input_ids, logits_to_keep=-1)
        self.assertEqual(logps.shape[1], 0)

    def test_full_sequence_logits_to_keep(self):
        """logits_to_keep 等于序列长度 - 1 时仍然正常。"""
        keep = self.T - 1
        logps = compute_per_token_logps(self.model, self.input_ids, logits_to_keep=keep)
        self.assertEqual(logps.shape, (self.B, keep))

    def test_ddp_model_unwrapping(self):
        """
        传入 DistributedDataParallel 包装的模型时应自动 unwrap，不崩溃。
        （用 MagicMock 模拟 DDP，不需要真实多卡环境）
        """
        from torch.nn.parallel import DistributedDataParallel
        ddp_model = MagicMock(spec=DistributedDataParallel)
        ddp_model.module = self.model
        logps = compute_per_token_logps(ddp_model, self.input_ids, logits_to_keep=3)
        self.assertEqual(logps.shape, (self.B, 3))


# ─────────────────────────────────────────────────────────────
# 测试 2：TorchRolloutEngine
# ─────────────────────────────────────────────────────────────

class TestTorchRolloutEngine(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 50
        self.model = TinyLM(vocab_size=self.vocab_size, hidden=16)
        self.tokenizer = make_fake_tokenizer(vocab_size=self.vocab_size)
        self.engine = TorchRolloutEngine(self.model, self.tokenizer, device="cpu")

    def _make_batch(self, B=2, prompt_len=6):
        input_ids = torch.randint(3, self.vocab_size, (B, prompt_len))
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def test_rollout_returns_rollout_result(self):
        """rollout() 必须返回 RolloutResult 实例。"""
        ids, mask = self._make_batch()
        result = self.engine.rollout(ids, mask, num_generations=2, max_new_tokens=4)
        self.assertIsInstance(result, RolloutResult)

    def test_output_ids_shape(self):
        """output_ids 的 batch 维度应该是 B * num_generations。"""
        B, G = 2, 3
        ids, mask = self._make_batch(B=B)
        result = self.engine.rollout(ids, mask, num_generations=G, max_new_tokens=5)
        self.assertEqual(result.output_ids.shape[0], B * G,
                         f"期望 batch={B*G}，实际 {result.output_ids.shape[0]}")

    def test_completion_ids_length(self):
        """completion_ids 的 seq_len 应等于 max_new_tokens。"""
        ids, mask = self._make_batch()
        max_new = 7
        result = self.engine.rollout(ids, mask, num_generations=1, max_new_tokens=max_new)
        self.assertEqual(result.completion_ids.shape[1], max_new,
                         f"期望 completion 长度 {max_new}，实际 {result.completion_ids.shape[1]}")

    def test_per_token_logps_shape(self):
        """per_token_logps 应与 completion_ids 形状一致。"""
        ids, mask = self._make_batch()
        result = self.engine.rollout(ids, mask, num_generations=2, max_new_tokens=5)
        self.assertEqual(result.per_token_logps.shape, result.completion_ids.shape,
                         "logps 和 completion_ids 形状必须匹配")

    def test_completions_count(self):
        """completions 列表长度应等于 B * num_generations。"""
        B, G = 2, 3
        ids, mask = self._make_batch(B=B)
        result = self.engine.rollout(ids, mask, num_generations=G, max_new_tokens=4)
        self.assertEqual(len(result.completions), B * G)

    def test_update_policy_replaces_model(self):
        """update_policy() 应替换内部 policy_model。"""
        new_model = TinyLM(vocab_size=self.vocab_size, hidden=16)
        self.engine.update_policy(new_model)
        self.assertIs(self.engine.policy_model, new_model)

    def test_rollout_no_grad(self):
        """
        rollout 期间不应累积梯度（torch.no_grad 内部运行）。
        验证方式：生成前后模型权重的 grad 仍为 None。
        """
        ids, mask = self._make_batch()
        self.engine.rollout(ids, mask, num_generations=1, max_new_tokens=3)
        for p in self.model.parameters():
            self.assertIsNone(p.grad,
                              "rollout 内部不应产生梯度，但发现 grad 不为 None")


# ─────────────────────────────────────────────────────────────
# 测试 3：SGLangRolloutEngine（Mock HTTP）
# ─────────────────────────────────────────────────────────────

class TestSGLangRolloutEngine(unittest.TestCase):
    """
    不启动真实 SGLang 服务器，用 unittest.mock 拦截所有 HTTP 请求。
    这样测试可以在任何机器上离线运行。
    """

    BASE_URL = "http://fake-sglang:30000"
    # 模型路径指向不存在的目录，需要 mock tokenizer 加载
    MODEL_PATH = "/fake/model"

    def _make_mock_tokenizer(self):
        tok = MagicMock()
        tok.eos_token_id = 2
        tok.pad_token_id = 0
        tok.decode = MagicMock(return_value="这是生成的回答")
        tok.batch_decode = MagicMock(return_value=["这是生成的回答"])
        return tok

    def _make_engine(self):
        with patch("trainer.rollout_engine.AutoTokenizer.from_pretrained",
                   return_value=self._make_mock_tokenizer()):
            engine = SGLangRolloutEngine(
                base_url=self.BASE_URL,
                model_path=self.MODEL_PATH,
                shared_ckpt_path="/tmp/fake_ckpt",
                timeout=5,
            )
        return engine

    def _fake_sglang_response(self, prompt_ids, completion_ids):
        """构造一条 SGLang /generate 响应体。"""
        logprobs = [-0.5] * len(completion_ids)
        return {
            "meta_info": {
                "output_ids": completion_ids,
                "output_logprobs": logprobs,
            }
        }

    def test_rollout_parses_response_correctly(self):
        """rollout() 能正确解析 SGLang 的 JSON 响应并返回 RolloutResult。"""
        engine = self._make_engine()

        B, G = 2, 2
        prompt_len = 5
        comp_len = 4
        input_ids = torch.randint(3, 50, (B, prompt_len))
        attention_mask = torch.ones_like(input_ids)

        # 构造 B*G 条假响应（每条 prompt 生成 G 个 completion）
        fake_results = []
        for _ in range(B * G):
            comp_ids = [10, 20, 30, 40][:comp_len]
            fake_results.append(self._fake_sglang_response(
                prompt_ids=[1, 2, 3, 4, 5],
                completion_ids=comp_ids,
            ))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_results
        mock_resp.raise_for_status = MagicMock()

        with patch.object(engine.http, "post", return_value=mock_resp):
            result = engine.rollout(input_ids, attention_mask,
                                    num_generations=G, max_new_tokens=comp_len)

        self.assertIsInstance(result, RolloutResult)
        # batch 维度
        self.assertEqual(result.output_ids.shape[0], B * G)
        self.assertEqual(result.completion_ids.shape[0], B * G)
        self.assertEqual(result.per_token_logps.shape[0], B * G)
        # completion 长度
        self.assertEqual(result.completion_ids.shape[1], comp_len)

    def test_rollout_logprobs_values(self):
        """per_token_logps 的数值应与 SGLang 返回的 logprob 一致。"""
        engine = self._make_engine()

        input_ids = torch.randint(3, 50, (1, 4))
        attention_mask = torch.ones_like(input_ids)
        expected_logp = -1.23

        fake_results = [{
            "meta_info": {
                "output_ids": [5, 6, 7],
                "output_logprobs": [expected_logp, expected_logp, expected_logp],
            }
        }]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_results
        mock_resp.raise_for_status = MagicMock()

        with patch.object(engine.http, "post", return_value=mock_resp):
            result = engine.rollout(input_ids, attention_mask,
                                    num_generations=1, max_new_tokens=3)

        actual = result.per_token_logps[0].tolist()
        for v in actual:
            self.assertAlmostEqual(v, expected_logp, places=4)

    def test_update_policy_calls_correct_endpoint(self):
        """update_policy() 应调用 /update_weights_from_disk 端点。"""
        engine = self._make_engine()

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {}
        mock_model.save_pretrained = MagicMock()
        mock_model.lm_head.weight = nn.Parameter(torch.zeros(10, 10))

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(engine.http, "post", return_value=mock_resp) as mock_post:
            result = engine.update_policy(mock_model)

        self.assertTrue(result)
        called_url = mock_post.call_args[0][0]
        self.assertIn("update_weights_from_disk", called_url,
                      f"应调用 update_weights_from_disk，实际调用：{called_url}")

    def test_health_returns_true_on_200(self):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(engine.http, "get", return_value=mock_resp):
            self.assertTrue(engine.health())

    def test_health_returns_false_on_exception(self):
        engine = self._make_engine()
        with patch.object(engine.http, "get", side_effect=ConnectionError("拒绝连接")):
            self.assertFalse(engine.health())

    def test_flush_cache_calls_correct_endpoint(self):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.object(engine.http, "post", return_value=mock_resp) as mock_post:
            result = engine.flush_cache()
        self.assertTrue(result)
        self.assertIn("flush_cache", mock_post.call_args[0][0])


# ─────────────────────────────────────────────────────────────
# 测试 4：create_rollout_engine 工厂函数
# ─────────────────────────────────────────────────────────────

class TestCreateRolloutEngine(unittest.TestCase):

    def test_creates_torch_engine(self):
        model = TinyLM()
        tok = make_fake_tokenizer()
        engine = create_rollout_engine("torch", policy_model=model, tokenizer=tok)
        self.assertIsInstance(engine, TorchRolloutEngine)

    def test_creates_sglang_engine(self):
        with patch("trainer.rollout_engine.AutoTokenizer.from_pretrained",
                   return_value=make_fake_tokenizer()):
            engine = create_rollout_engine(
                "sglang",
                sglang_base_url="http://fake:30000",
                sglang_model_path="/fake/model",
                sglang_shared_path="/tmp/fake",
            )
        self.assertIsInstance(engine, SGLangRolloutEngine)

    def test_unknown_engine_raises_value_error(self):
        with self.assertRaises(ValueError):
            create_rollout_engine("unknown_engine")


# ─────────────────────────────────────────────────────────────
# 测试 5：RolloutResult 数据结构
# ─────────────────────────────────────────────────────────────

class TestRolloutResult(unittest.TestCase):

    def test_fields_accessible(self):
        """RolloutResult 的四个字段都应能正常访问。"""
        r = RolloutResult(
            output_ids=torch.zeros(2, 10, dtype=torch.long),
            completion_ids=torch.zeros(2, 5, dtype=torch.long),
            per_token_logps=torch.zeros(2, 5),
            completions=["hello", "world"],
        )
        self.assertEqual(r.output_ids.shape, (2, 10))
        self.assertEqual(r.completion_ids.shape, (2, 5))
        self.assertEqual(r.per_token_logps.shape, (2, 5))
        self.assertEqual(r.completions, ["hello", "world"])


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 按测试组顺序加载
    for cls in [
        TestComputePerTokenLogps,
        TestTorchRolloutEngine,
        TestSGLangRolloutEngine,
        TestCreateRolloutEngine,
        TestRolloutResult,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
