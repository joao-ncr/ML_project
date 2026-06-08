# 📊 Semana 5 — Regressão e Séries Temporais
### Capacitação Ciência de Dados · Asimov Jr

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-blue)
![Status](https://img.shields.io/badge/status-concluído-brightgreen)

---

## 📌 Contexto

Este projeto é a entrega da **Semana 5** da trilha de Capacitação em Ciência de Dados da [Asimov Jr](https://asimov.academy/), com foco na aplicação prática de três técnicas de machine learning supervisionado e análise temporal:

- **Regressão Linear** — previsão de valores contínuos
- **Regressão Logística** — classificação binária
- **Séries Temporais com Prophet** — forecasting de dados sequenciais

O objetivo da atividade é aplicar essas três técnicas sobre um mesmo dataset, justificando as escolhas, avaliando os resultados e extraindo conclusões de negócio.

---

## 🗂️ Dataset

**Superstore Sales** — disponível publicamente no [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

| Atributo | Valor |
|----------|-------|
| Fonte | Kaggle |
| Arquivo | `Sample - Superstore.csv` |
| Registros | ~10.000 vendas |
| Período | Janeiro 2015 – Dezembro 2018 |
| Encoding | `latin-1` |

O dataset simula as operações de uma loja de varejo americana e contém colunas de pedidos, clientes, produtos, descontos, receita e lucro — cobrindo os três requisitos da atividade: variáveis contínuas, variável binária e coluna de data sequencial.

---

## 🎯 Problemas propostos

| Modelo | Pergunta de negócio | Variável alvo |
|--------|---------------------|---------------|
| Regressão Linear | Quanto uma venda vai gerar em receita? | `Sales` (contínua) |
| Regressão Logística | Essa venda vai dar lucro ou prejuízo? | `Profit_flag` (binária: `Profit > 0`) |
| Séries Temporais | Qual será a receita diária nos próximos 3 meses? | `Sales` agregado por dia |

---

## 🗃️ Estrutura do projeto

```
📁 semana-5-regressao-series-temporais/
│
├── 📓 ml.ipynb                     # Notebook principal com todo o desenvolvimento
├── 📄 README.md                    # Este arquivo
└── 📊 Sample - Superstore.csv      # Dataset (baixar do Kaggle — ver instruções)
```

> ⚠️ O arquivo CSV **não está incluído** no repositório. Veja as instruções de instalação abaixo para obtê-lo.

---

## 🔧 Instalação e configuração

### Pré-requisitos

- Python **3.10 ou superior**
- pip atualizado
- VSCode com a extensão **Jupyter** instalada (recomendado) — ou Jupyter Lab/Notebook

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/semana5-asimov.git
cd semana5-asimov
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install pandas numpy matplotlib seaborn scikit-learn prophet
```

Ou, se preferir instalar com versões fixas para garantir reprodutibilidade:

```bash
pip install pandas==2.2.0 numpy==1.26.4 matplotlib==3.8.2 seaborn==0.13.2 scikit-learn==1.4.0 prophet==1.1.5
```

> 💡 **Nota sobre o Prophet:** a biblioteca pode levar alguns minutos para instalar pois compila dependências em C++. Se encontrar erro no Windows, instale o [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) antes.

### 4. Baixe o dataset

1. Acesse [Superstore Dataset — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
2. Faça login ou crie uma conta gratuita
3. Clique em **Download**
4. Extraia e coloque o arquivo `Sample - Superstore.csv` na **mesma pasta** do notebook

### 5. Abra o notebook

No VSCode:
```bash
code ml.ipynb
```

Selecione o kernel Python do seu ambiente virtual no canto superior direito e clique em **Run All**.

---

## 📦 Bibliotecas utilizadas

| Biblioteca | Uso |
|------------|-----|
| `pandas` | Manipulação e transformação do dataset |
| `numpy` | Operações numéricas e transformações (log1p, expm1) |
| `matplotlib` | Geração de todos os gráficos |
| `seaborn` | Visualizações estatísticas (heatmap, scatter) |
| `scikit-learn` | Modelos de regressão linear e logística, métricas e pré-processamento |
| `prophet` | Modelagem e previsão de séries temporais |

---

## 🔬 Metodologia

### Análise Exploratória (EDA)

Antes de qualquer modelo, o notebook realiza uma análise exploratória com foco em:
- Distribuição de `Sales` (assimétrica → justifica uso de `log1p`) e `Profit` (com valores negativos → motiva a criação de `Profit_flag`)
- Relação entre desconto e lucro por categoria — desconto acima de 0.4 concentra a maioria dos prejuízos
- Matriz de correlação entre variáveis numéricas
- Detecção de outliers em `Sales` pelo método IQR — mantidos no dataset por serem vendas legítimas de alto valor

### Regressão Linear

- **Alvo:** `log1p(Sales)` — transformação logarítmica aplicada para estabilizar a variância e reduzir o impacto dos outliers
- **Features:** `Discount`, `Quantity` (numéricas padronizadas com `StandardScaler`) + `Category`, `Region`, `Segment` (categóricas com One-Hot Encoding)
- **Divisão:** 80% treino / 20% teste com `random_state=42`
- **Métricas:** R², RMSE e MAE no espaço log
- **Visualizações:** dispersão real vs. previsto, distribuição dos resíduos e coeficientes do modelo

### Regressão Logística

- **Alvo:** `Profit_flag` — variável binária criada a partir de `Profit > 0`
- **Features:** mesmas da regressão linear + `log1p(Sales)` como preditor adicional
- **Divisão:** 80% treino / 20% teste com `stratify=y_clf` para preservar proporção de classes
- **Parâmetros:** `class_weight='balanced'` para lidar com possível desbalanceamento; `max_iter=1000` para garantir convergência
- **Métricas:** acurácia, AUC-ROC e relatório de classificação (precisão, recall, F1)
- **Visualizações:** matriz de confusão, curva ROC e Odds Ratio dos coeficientes

### Séries Temporais — Prophet

- **Série:** `Sales` agregado por dia (`Order Date` → `ds`, soma de `Sales` → `y`) com dias sem venda preenchidos com zero
- **Configuração:** `yearly_seasonality=True`, `weekly_seasonality=True`, `daily_seasonality=False`, `changepoint_prior_scale=0.05`
- **Previsão:** +90 dias além do último registro histórico
- **Métricas:** MAE e RMSE in-sample
- **Visualizações:** série bruta histórica, gráfico de previsão com intervalo de incerteza e decomposição em componentes (tendência, sazonalidade semanal e anual)

---

## 📈 Resultados e conclusões

### O que os modelos revelam sobre o negócio

**Desconto é o principal destruidor de margem.** Tanto nos coeficientes da regressão linear quanto no Odds Ratio da regressão logística, `Discount` aparece como o preditor mais negativo. Descontos acima de 40% estão quase inteiramente associados a vendas com prejuízo — uma informação diretamente acionável para o time comercial.

**Categoria importa para receita, mas não necessariamente para margem.** `Furniture` gera receita expressiva, mas sua margem é consistentemente menor do que `Technology`. Os modelos capturam essa diferença através do One-Hot Encoding de `Category`.

**A loja tem crescimento estrutural e sazonalidade previsível.** A decomposição do Prophet revela tendência de crescimento ao longo dos 4 anos, picos de receita em novembro/dezembro (Black Friday + Natal) e padrão semanal estável. Isso torna o planejamento de estoque e equipe antecipado mais confiável.

**Transformações de escala são decisivas.** O uso de `log1p` no alvo da regressão linear foi essencial — sem ele, o modelo tentaria acomodar vendas de R$50 e de R$20.000 na mesma escala, prejudicando o ajuste para a maioria das transações.

### Pontos de melhoria identificados

- **Regressão Linear:** incluir `Sub-Category` e `Ship Mode` como features; testar modelos regularizados (Ridge, Lasso) para reduzir overfitting em categorias com poucos dados
- **Regressão Logística:** explorar limiares de decisão abaixo de 0.5 para aumentar o recall da classe `Prejuízo` — que é o erro de maior custo para o negócio
- **Séries Temporais:** aplicar `cross_validation()` do Prophet para estimativa de erro out-of-sample mais honesta; adicionar feriados americanos como regressores externos
- **Integração dos modelos:** usar a previsão de `Sales` do Prophet como feature de entrada na regressão logística, criando um pipeline que antecipa vendas de baixa margem antes dos pedidos serem fechados

---

## 📚 Referências

- [Documentação do scikit-learn](https://scikit-learn.org/stable/)
- [Documentação do Prophet](https://facebook.github.io/prophet/)
- [Superstore Dataset — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- [Asimov Jr — Capacitação Ciência de Dados](https://asimov.academy/)

---

## 👤 Autor

Desenvolvido como entrega da Semana 5 da Capacitação em Ciência de Dados da **Asimov Jr**.

---

> *"Um modelo não precisa ser perfeito para ser útil — ele precisa ser melhor do que não ter modelo nenhum."*