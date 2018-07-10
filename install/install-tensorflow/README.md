# TensorFlowインストールガイド

## TL;DR:

![image](https://user-images.githubusercontent.com/28590220/29164815-c5cc9250-7dfb-11e7-81b7-e8514e995213.png)

TensorFlowは大規模な数値計算を行うライブラリである。
機械学習や深層学習で使われるので、そのためのツールのように思われることも多いが、様々な数値演算ができる汎用的なライブラリである。

名前の通り多次元配列計算を得意としている。Windowsが未対応なため、LinuxやMacを使う必要がある。

https://www.tensorflow.org/

## インストール方法

anyenv環境のanacondaでダウンロードしている。
ここは本当に環境ごとに多様なダウンロード方法があるので純粋にmacのインストールページでダウンロードしていたらはまった。

私の場合はここが一番参考になった

https://www.tensorflow.org/versions/r0.12/get_started/os_setup

```
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.1-py3-none-any.whl
$ pip install --upgrade $TF_BINARY_URL 
```

また、新陳代謝が激しいのですぐこのURLも廃れている可能性があるため自身で最新のURLを入手することが必要となる。

'tensorflow nightly build' で検索すると最新版が出てきたりするのでもしハマってしまったなら頼ってみるのもいいかもしれない。

## 動作確認

```rb
$ python
>>> import tensorflow as tf
>>> sess = tf.Session()
>>> hello = tf.constant('Hello')
>>> sess.run(hello)
b'Hello'
```

