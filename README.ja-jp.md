# CntkMnistPractice
C++とC#でCNTKを使ったMNIST分類プログラムです。Fully-connected layerだけを使いました。言語による2つのバージョンがありますが、完全に同じ機能を遂行します。

## ファイル
### Program.cpp / Program.cs
プログラムのエントリーポイントです。

### Mnist.cpp (Mnist.h) / Mnist.cs
Mnist、MnistItem、NormalizedMnistItemの3つのクラスの定義を含んでいます。
#### Mnist class
MNISTアイテムのリストです。
#### MnistItem class
一つのMNISTアイテムの画像データとラベル情報を含んでいます。
#### NormalizedMnistItem class
[0, 255]の範囲の値を[0.0, 1.0]の範囲に加工し、One-hot encodingを行います。

### MnistClassifier.cpp (MnistClassifier.h) / MnistClassifier.cs
学習とテストを行うMnistClassifierクラスの定義を含んでいます。

## その他
このレポジトリのオーナーは熟練したプログラマーでも、日本語が母国語である人でもありません。プログラム的、言語（日本語）的なエラーの可能性がありますので、エラーがあれば是非レポジトリのオーナーにお知らせてください。
