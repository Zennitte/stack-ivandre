# stack-ivandre — classificação de clareza de respostas e-SIC

Este projeto classifica a clareza de respostas a pedidos de informação do e-SIC. A entrada é o texto da resposta e a saída é um rótulo entre **c1**, **c234** e **c5**. O modelo final usa BERTimbau, TF-IDF de palavras e TF-IDF de caracteres. Uma regressão logística combina as nove probabilidades dos três componentes por stacking.

No teste 1, o modelo acertou **462 de 900 respostas (51,33%)**, ante **443 de 900 (49,22%)** do EP1. A diferença foi de 2,11 pontos percentuais. O teste de McNemar exato bilateral resultou em **p = 0,1401**. [Relatório do EP2](reports/relatorio-ep2.pdf).

## Contexto geral

| Rótulo | Nota |
|---|---|
| c1 | 1 |
| c234 | 2, 3 ou 4 |
| c5 | 5 |

O classificador recebe apenas o conteúdo de resp_text. O pacote inclui os pesos e o tokenizer de BERTimbau, os dois pipelines TF-IDF, o combinador, as planilhas e os resultados agregados.

## Instalação

Use Python 3.12, Git e Git LFS. Os pesos de BERTimbau são armazenados no LFS.

~~~bash
git lfs install
git clone https://github.com/Zennitte/stack-ivandre.git
cd stack-ivandre
git lfs pull
python -m venv .venv
~~~

No Windows, use `.\.venv\Scripts\Activate.ps1`. No Linux, use `source .venv/bin/activate`. Instale as dependências de inferência em CPU.

~~~bash
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
~~~

Para reproduzir o ambiente CUDA 12.8, use um ambiente separado e instale `requirements-cuda.lock.txt` com `--extra-index-url https://download.pytorch.org/whl/cu128`. [Configuração do modelo](artifacts/inference_config.json).

## Como prever em uma planilha

~~~bash
python predict.py --input data/test.xlsx --sheet test2 --output predictions/nova_execucao.xlsx --device cpu
~~~

A planilha do teste 2 tem 897 respostas. Para aplicar o modelo a uma planilha própria, use uma aba com os cabeçalhos `resp_text` e `clarity`, deixe `clarity` vazia e informe a aba e a contagem com `--sheet` e `--expected-rows`. Escolha um caminho de saída novo. O programa preenche as células de rótulo e grava probabilidades e metadados em arquivos auxiliares.

Para inferência em GPU com PyTorch CUDA, use `--device cuda`. O lote padrão é 4 e pode ser ajustado por `--batch-size`.

## Como usar em Python

~~~python
from model import CLASSES, Inference

classificador = Inference(device="cpu")
probabilidades = classificador.predict_proba(
    ["A informação solicitada está disponível no portal do órgão."]
)
rotulos = [CLASSES[i] for i in probabilidades.argmax(axis=1)]
~~~

## Modelo e dados

Os três componentes foram ajustados em 15.035 respostas de treino após deduplicação por texto exato e rótulo. O combinador foi ajustado em 4.101 respostas de grupos distintos. O BERT usa até 512 tokens. Os modelos lexicais usam n-gramas de palavras (1 a 2) e de caracteres (3 a 5) com TF-IDF e regressão logística.

| Caminho | Conteúdo |
|---|---|
| [data/train.xlsx](data/train.xlsx) | Treino rotulado com 20.092 respostas |
| [data/test.xlsx](data/test.xlsx) | Teste 2 com 897 respostas |
| [predictions/test_predito.xlsx](predictions/test_predito.xlsx) | Rótulos entregues para o teste 2 |
| [artifacts/](artifacts/) | Pesos, tokenizer, pipelines e manifesto SHA-256 |
| [training/](training/) | Código e divisão do ajuste final |
| [reports/](reports/) | Relatório em PDF e LaTeX, métricas de avaliação |
| [provenance/](provenance/) | Versões e identificadores dos artefatos |

## Avaliação

A avaliação de desenvolvimento usou 12 rodadas de cinco folds estratificados por grupos. O stacking triplo atingiu 46,1830% de acurácia média, ante 45,9329% da média 50/50. No teste 1, o ganho sobre o EP1 foi de 2,11 pontos percentuais, com p = 0,1401. O teste 1 já havia sido usado em avaliações anteriores e contém duplicatas, conflitos e sobreposição com o treino. O teste 2 tem somente previsões disponíveis. [Métricas agregadas do teste 1](reports/test1_metricas.json) e [resultados de desenvolvimento](reports/desenvolvimento.json).

## Reconstruir o treinamento final

Use a GPU e as versões registradas em [provenance/runtime_versions.json](provenance/runtime_versions.json). O treinamento usa `data/train.xlsx` e o split em `training/final/final_split.json`. Obtenha o pré-treinado BERTimbau na revisão usada no ajuste.

~~~bash
python -m pip install -r requirements-cuda.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
huggingface-cli download neuralmind/bert-base-portuguese-cased config.json pytorch_model.bin vocab.txt tokenizer_config.json --revision 94d69c95f98f7d5b2a8700c420230ae10def0baa --local-dir .cache/bertimbau
python train.py
~~~

O treinamento verifica o hash dos dados e do pré-treinado, ajusta os três componentes e grava os novos modelos em `training/final/training/`. A inferência padrão usa os pesos publicados em `artifacts/`.

## Referência

O EP1 está documentado no repositório [ESIC-BERT-IVANDRE](https://github.com/Zennitte/ESIC-BERT-IVANDRE).
