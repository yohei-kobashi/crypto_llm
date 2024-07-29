# crypto_llm_gpt_1.3B_hyper_param_tune

事前学習を高速化するためには、以下の手順でハイパラ探索をすると良いと思います。

1. (micro) batch size は 1 で固定して、 global batch size を 384 -> 512 -> 768 -> 1024 -> 1536 と徐々に上げる。
2. 手順 1. で最も良かった global batch size で、 (micro) batch size を 1 -> 2 -> 4 と徐々に上げる。
3. もし余裕があれば、 sequence length を 4096 から 2048 に下げて、手順 1. ~ 2. を試す。
4. もし余裕があれば、 precision を fp32 から tf32 に変えて、手順 1. ~ 3. を試す。

※ tf32 を試すためには、事前に下記のコマンドで [matsuolab/Megatron-DeepSpeed](https://github.com/matsuolab/Megatron-DeepSpeed.git) レポジトリを `ucllm_nedo_v20240516.1.0` タグにチェックアウトしておく必要があります。

```sh
(.venv_train) $ cd ~/crypto_llm/train/Megatron-DeepSpeed/ && git fetch origin && git checkout refs/tags/ucllm_nedo_v20240516.1.0
```
