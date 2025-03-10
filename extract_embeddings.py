import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, BertConfig
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# エンティティとリレーションの埋め込みを抽出する関数
def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            
            # モデルのエンコーダー層の出力を取得
            outputs = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            last_hidden_state = outputs.last_hidden_state
            embeddings.append(last_hidden_state[:, 0, :].cpu().numpy())  # [CLS]トークンの埋め込みを取得
    return np.concatenate(embeddings, axis=0)

# データローダーを作成する関数
def create_dataloader(texts, tokenizer, max_length, batch_size):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], encodings['token_type_ids'])
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

# テキストファイルをロードする関数
def load_text(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    texts = [line.strip().split('\t')[1] for line in lines]
    return texts

# メイン関数
def main():
    parser = argparse.ArgumentParser()

    # 引数の設定
    parser.add_argument("--data_dir", default="./data/WN18RR", type=str, help="The input data dir.")
    parser.add_argument("--model_dir", default="./output_WN18RR", type=str, help="The directory of the trained model.")
    parser.add_argument("--output_dir", default="./output_WN18RR/embedding", type=str, help="The directory to save embeddings.")
    parser.add_argument("--max_seq_length", default=50, type=int, help="The maximum total input sequence length.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for inference.")
    args = parser.parse_args()

    logger.info("Using device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザーのロード
    logger.info("Loading model and tokenizer from %s", args.model_dir)
    config = BertConfig.from_pretrained(args.model_dir)
    model = BertModel.from_pretrained(args.model_dir, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model.to(device)
    logger.info("Model and tokenizer loaded successfully")

    # エンティティとリレーションのテキストをロード
    logger.info("Loading entity and relation texts from %s", args.data_dir)
    entity_texts = load_text(os.path.join(args.data_dir, "entity2text.txt"))
    relation_texts = load_text(os.path.join(args.data_dir, "relation2text.txt"))
    logger.info("Loaded %d entity texts and %d relation texts", len(entity_texts), len(relation_texts))

    # データローダーを作成
    logger.info("Creating dataloaders")
    entity_dataloader = create_dataloader(entity_texts, tokenizer, args.max_seq_length, args.batch_size)
    relation_dataloader = create_dataloader(relation_texts, tokenizer, args.max_seq_length, args.batch_size)
    logger.info("Dataloaders created successfully")

    # エンティティとリレーションの埋め込みを取得
    logger.info("Extracting entity embeddings")
    entity_embeddings = get_embeddings(model, entity_dataloader, device)
    logger.info("Entity embeddings extracted successfully, shape: %s", entity_embeddings.shape)

    logger.info("Extracting relation embeddings")
    relation_embeddings = get_embeddings(model, relation_dataloader, device)
    logger.info("Relation embeddings extracted successfully, shape: %s", relation_embeddings.shape)

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logger.info("Created output directory %s", args.output_dir)

    # 埋め込みをファイルに保存
    logger.info("Saving embeddings to %s", args.output_dir)
    np.save(os.path.join(args.output_dir, "entity_embedding.npy"), entity_embeddings)
    np.save(os.path.join(args.output_dir, "relation_embedding.npy"), relation_embeddings)
    logger.info("Embeddings saved successfully")

if __name__ == "__main__":
    main()
