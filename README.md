# Crypto-LLM
本リポジトリは"Crypto-LLM: Two-Stage Language Model Pre-training with Ciphered and Natural Language Data"の実験に用いたコードです。

## 1. データ取得用の環境設定
```sh
$ cd ~/

# condaのインストール先ディレクトリを作成。
$ mkdir -p ~/miniconda3/ && cd ~/miniconda3/

# condaをインストール。
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh && bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh -b -u -p ~/miniconda3/

# インストールしたcondaを有効化。
$ source ~/miniconda3/etc/profile.d/conda.sh

# condaコマンドが使えることを確認。
$ which conda && echo "====" && conda --version

# Python仮想環境を作成。
$ conda create --name .venv_data python=3.11.7 -y

# 作成したPython仮想環境を有効化。
$ conda activate .venv_data

# データ取得、加工の作業ディレクトリへ移動
$ cd ~/crypto_llm/data_management

# pythonのライブラリ群をインストール
$ ./bin/setup
```

## 2. データダウンロード
Wikipedia dumpデータの取得（注：20240520は現在はダウンロード不可）
```sh
$ python -m preprocessing.download_dataset --dataset=wikipedia --split=20240520 --output_base=tmp/output
```

## 3. 学習用データの作成

LLM学習用データを収集加工する手順です。  
mC4(Japanese)のダウンロード、一連の加工処理を含みます。  
[こちら](data_management/README.md)

## LLM学習手順s

LLM学習手順です。  
トークナイザー学習、事前学習、事後学習（ファインチューニング）を含みます。  
[こちら](train/README.md)

## LLM評価手順

LLM評価手順です。  
本企画の評価指標であるNejumi Leaderboard Neoにおける評価手順となります。  
[こちら](eval/README.md)

## Contributors

```
@software{ucllm-nedo,
  author       = {Kawanishi, Hotsuyuki and
                  Shinozuka, Fumiya and
                  Taniguchi, Masachika and
                  Yamazaki, Yudai and
                  Yamagiwa, Manami and
                  Sekioka, Satoshi and
                  Harada, Keno and
                  Alfredo, Solano Martinez and
                  Noumi, Yoshihiro and
                  Yu, Zhenxuan and
                  Kobashi, Yohei and
                  Kojima, Takeshi},
  title        = {Standard Codes and Procedures for LLM Development},
  month        = 3,
  year         = 2024,
  url          = {https://github.com/matsuolab/ucllm_nedo_prod}
}
```
