# unsloth fintune

用unsloth fintune 

## Getting Started

先用[marker](https://github.com/VikParuchuri/marker)把`PDF`論文轉成`markdown`純文字格式，將`markdown`轉換符合[huggingface](https://huggingface.co/) 的`dataset`格式上傳，最後再透過[unsloth](https://github.com/unslothai/unsloth)開使進行`fintune`。

### Prerequisites

The things you need before installing the software.

* 用[uv](https://github.com/astral-sh/uv)建立虛擬環境
* 申請[huggingface](https://huggingface.co/)帳號和`access token`
* 安裝`marker`、`unsloth`和其他python套件
* 要先確認`huggingface`的`access token`是否正確位於`.env`檔案中。

    ```bash
    $ echo "HF_TOKEN=xxxxxxxxxxxxxxxx" > .env # 設定huggingface token
    ``` 


### Installation

用[uv](https://github.com/astral-sh/uv)建立虛擬環境

1. 第一次使用專案

    ```bash
    $ pip install uv #安裝uv
    $ uv python install 3.10 3.11 3.12 # 準備環境
    $ uv venv --python 3.11 # 建立虛擬環境
    $ source .venv/bin/activate # 啟動虛擬環境 
    $ uv pip install marker-pdf # 安裝marker
    $ uv pip sync requirements.txt # 安裝其他套件
    ```

2. 重新啟動專案

    ```bash
    $ source .venv/bin/activate # 啟動虛擬環境
    $ deactivate # 關閉虛擬環境
    ```

- 如有新增安裝套件

    ```bash
    $ uv pip install -r requirements.txt
    ```

## Usage

1. 用`marker`將`PDF`轉換成`markdown`格式

    ```bash
    $ marker_single --output_dir docs/md docs/pdf/ijms-25-00574-v2.pdf # 專換單一檔案
    $ marker_chunk_convert ../pdf_in ../md_out # 批次轉換
    ```

2. 將`markdown`轉換成`dataset`格式，並上傳到`huggingface`，並建立`dataset`。

    ```bash
    $ uv run 01_create_dataset.py
    ```

3. 用`unsloth`進行`fintune`

    ```bash
    $ uv run 02_Fine_Tuning.py
    ```

4. 使用finetune後的模型

    ```bash
    $ uv run 03_using_the_mode.py
    ```

## Deployment

透過`unsloth`進行`fintune`有提供`4bit`模型，可以加快下載速度，並且避免`OOMs`。
這裡選用`unsloth/gemma-2-9b`模型進行`fintune`。
> unsloth huggingface 首頁有更多模型選擇 https://huggingface.co/unsloth

* max_seq_length: 每個模型都有不同的最大長度，比喻在閱讀教材時，每次最多讀幾個字(最大 token 數量)。
* model_name: 模型名稱，可以選擇不同的模型，這裡選用`unsloth/gemma-2-9b`模型。
* load_in_4bit: 窮人用的。有錢人都用`16bit`，這裡選用`4bit`模型，可以加快下載速度，並且避免`OOMs`。

```python
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally! # 
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False. 

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

print("Loading model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b", # 模型名稱
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
```

* target_modules: 模型的哪個部分要進行訓練，可以是模型的一部分，也可以是整個模型。這參數設定好可以讓模型更有針對性，也更加高效。
* lora_alpha: 設定的越大`Lora`影響就會越大，會影響模型的原始性能要做平衡拿捏。
* lora_dropout: 訓練過程中隨機比例丟棄，可以避免過度擬合。
```python
print("Loading Laura")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0, # Supports any, but = 0 is optimized
    bias="none", # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False, # We support rank stabilized LoRA
    loftq_config=None, # And LoftQ
)
```


### 專有名詞說明

* Lora / Qlora: 大模型比做一本百科全書，Lora 就比喻是便利貼把額外的訊息寫上再貼上。Qlora 則是在更小的便利貼上寫上更多的訊息。
* Dataset: 為了讓大模型更好的理解指令，做出更恰當的回應。用較少的資源訓練出更好的模型。
* SFTrainer: 用來訓練模型的工具，可以用來訓練大型模型。簡化了很多訓練的步驟，並且提供了很多優化訓練的參數。
* overfitting: 過度擬合，模型在訓練集上表現很好，但在測試集上表現不好。死讀書沒辦法面對新的問題，失去舉一反三的能力。
* etc...


## Additional Documentation and Acknowledgments

* [Creating and Uploading a Dataset with Unsloth: An Adventure in Wonderland](https://huggingface.co/blog/dimentox/unsloth-mistral-training)
* [Alpaca + Gemma2 9b Unsloth 2x faster finetuning](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing#scrollTo=95_Nn-89DhsL)
* [unsloth/huggingface](https://huggingface.co/unsloth)