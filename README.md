# Word2Vec

## 1.Modules
```
.
|-- Embedding
|-- Model
|-- README.md
|-- Token
|   |-- kor_spm.model
|   |-- kor_spm.vocab
|   `-- korean.txt
|-- dataset.py
|-- model.py
|-- preprocessor.py
`-- train_kor.py
```

## 2. Model Specification
  1. Winodw Size : 11
  2. Token Size : 7000
  3. Model
     * CBOW
     * SkipGram
  4. Embedding Size : 512

## 3. Training
  1. Optimizer : Adam
  2. Learning Rate : 1e-4
  3. Batch Size : 1024
  
## 4. Source
  1. 모두의 말뭉치 : https://corpus.korean.go.kr
  2. 일상 대화 말뭉치

