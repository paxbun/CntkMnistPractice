# CntkMnistPractice
C++과 C#으로 작성된 CNTK를 사용한 MNIST 분류 프로그램입니다. Fully-Connected Layer 만을 사용했습니다. 언어에 따른 두 가지 버전이 있는데, 완벽히 똑같은 작업을 수행합니다.

## 파일들
### Program.cpp / Program.cs
프로그램의 엔트리 포인트입니다.

### Mnist.cpp (Mnist.h) / Mnist.cs
이 파일은 Mnist, MnistItem, NormalizedMnistItem 클래스의 정의를 담고 있습니다.
#### Mnist
MNIST 요소들의 리스트입니다.
#### MnistItem
하나의 이미지와 그에 대응하는 레이블 하나의 정보를 담고 있습니다.
#### NormalizedMnistItem
이 클래스는 [0, 255]의 범위의 값을 [0.0, 1.0]의 범위로 바꾸고 원 핫 인코딩을 수행합니다.

### MnistClassifier.cpp (MnistClassifier.h) / MnistClassifier.cs
MNIST 분류 프로그램의 학습과 평가를 수행하는 MnistClassifier 클래스의 정의가 담겨있습니다.

## 기타
프로그램 저장소 내에 있는 모든 종류의 오류에 대해 알려주시면 감사하겠습니다.
