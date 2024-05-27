import unittest
from transformers import AutoTokenizer
from ..template import modelType2Template


class TestTemplates(unittest.TestCase):
    
    def setUp(self):
        self.access_token = "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"
        self.message = [
            {"role": "user", "content": "你好啊"},
            {"role": "assistant", "content": "我真的很开心"},
            {"role": "user", "content": "你好啊"},
            {"role": "assistant", "content": "我真的很开心"},
        ]

    def test_llama_template(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", token=self.access_token
        )
        
        g = modelType2Template["llama"](tokenizer)

        c = tokenizer.apply_chat_template(self.message, tokenize=True)
        a, b = g.apply(self.message)
        print('llama\n',a,'\n', c,'\n',)
        # print(tokenizer.eos_token_id)
        # print(tokenizer.convert_ids_to_tokens(c))
        # print(tokenizer.decode(a))
        if g.efficient_eos:
            self.assertEqual(a,c+[g.end_token_id])
        else:
            self.assertEqual(a, c)

    def test_gemma_template(self):
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-1.1-2b-it", token=self.access_token
        )
        g = modelType2Template["gemma"](tokenizer)

        c = tokenizer.apply_chat_template(self.message, tokenize=True)
        a, b = g.apply(self.message)
        print('gemma\n',a,'\n', c,'\n',)
        # print(tokenizer.eos_token_id)
        if g.efficient_eos:
            self.assertEqual(a,c+[g.end_token_id])
        else:
            self.assertEqual(a, c)


if __name__ == "__main__":
    unittest.main()
