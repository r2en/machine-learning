# Kerasインストールガイド

## TL;DR:

![image](https://user-images.githubusercontent.com/28590220/29164851-e2f0af74-7dfb-11e7-8121-f646712c529a.png)

KerasはバックエンドとしてTheanoとTensorflowの両方が使え、より高レイヤな表現で深層学習のさまざまなアルゴリズムが記述できる。またプログラムを修正せずに、バックエンドをTensorFlowからTheanoに記述一つで変更することもできる。

https://keras.io/

## インストール方法

※ 事前にTensorFlowを導入していること！

pipで導入する

```rb
$ pip install keras
```

root直下に.kerasフォルダを作成する

```rb
$ mkdir ~/.keras
$ vim ~/.keras/keras.json
```

その中のjsonファイルにバックエンドを何で動かすかなどを記述する

```rb
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

```

ネットにはthとtfで書かれているものがあるが画像集合を表す四次元テンソルの順番が変わってしまいハマる可能性があるので注意。デフォルトはtfなので変更しないでおく。

