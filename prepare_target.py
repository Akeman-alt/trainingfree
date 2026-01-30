import torch
import esm
import requests
import os
from tqdm import tqdm

# --- 配置 ---
SAVE_PATH = ".cache/torch/checkpoints/target_membrane_embedding.pt"
MODEL_NAME = "esm2_t33_650M_UR50D" # 650M 参数版，4090 跑得飞快
NUM_SEQUENCES = 50                 # 拿 50 条做平均足够了
MAX_LENGTH = 1024                  # 截断过长的序列防止显存爆炸

def fetch_membrane_proteins(num=50):
    """
    从 UniProt API 自动下载膜蛋白序列
    关键词: "transmembrane" AND "reviewed:true" (高质量)
    """
    print(f"🌐 正在从 UniProt 下载 {num} 条膜蛋白序列...")
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": "(keyword:\"Transmembrane [KW-0812]\") AND (reviewed:true)",
        "format": "fasta",
        "size": num
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # 简单的 FASTA 解析
    sequences = []
    lines = response.text.strip().split('\n')
    current_seq = ""
    current_header = ""
    
    for line in lines:
        if line.startswith(">"):
            if current_seq and len(current_seq) < MAX_LENGTH:
                sequences.append((current_header, current_seq))
            current_header = line
            current_seq = ""
            if len(sequences) >= num:
                break
        else:
            current_seq += line.strip()
            
    if current_seq and len(sequences) < num and len(current_seq) < MAX_LENGTH:
        sequences.append((current_header, current_seq))
        
    print(f"✅ 成功获取 {len(sequences)} 条序列。")
    return sequences

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 1. 加载模型
    print(f"📦 正在加载模型 {MODEL_NAME}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    # 2. 获取数据
    data = fetch_membrane_proteins(NUM_SEQUENCES)
    
    # 3. 计算 Embeddings
    print("⚗️ 正在计算 Embeddings...")
    all_embeddings = []
    
    # 逐条处理 (Batch size = 1 比较稳，反正很快)
    with torch.no_grad():
        for header, seq in tqdm(data):
            batch_labels, batch_strs, batch_tokens = batch_converter([(header, seq)])
            batch_tokens = batch_tokens.to(device)
            
            # ESM 前向传播
            # repr_layers=[33] 表示取最后一层 (650M模型共33层)
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33] # [1, L+2, D]
            
            # --- 关键点: 选择 Pooling 策略 ---
            
            # 策略 A: [CLS] Token (分类任务常用)
            # embedding = token_representations[0, 0] 
            
            # 策略 B: Mean Pooling (语义相似度常用 <- 推荐!)
            # 注意: 排除 <cls> (index 0) 和 <eos> (index -1)
            # batch_tokens 中 padding 的位置也要排除，但这里 batch=1 所以不用管 padding
            seq_len = len(seq)
            # 取 1 到 seq_len+1 的范围，避开首尾特殊 token
            embedding = token_representations[0, 1 : seq_len + 1].mean(dim=0)
            
            all_embeddings.append(embedding)

    # 4. 计算平均向量
    if len(all_embeddings) == 0:
        print("❌ 错误：没有生成任何 Embedding")
        return

    # Stack: [N, D] -> Mean: [D]
    all_embeddings_tensor = torch.stack(all_embeddings)
    target_embedding = all_embeddings_tensor.mean(dim=0)
    
    # 归一化 (这一步很重要，因为之后我们要算 Cosine Similarity)
    target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    
    # 5. 保存
    # 增加一个维度变为 [1, D]，方便后续矩阵乘法
    target_embedding = target_embedding.unsqueeze(0) 
    
    torch.save(target_embedding, SAVE_PATH)
    print(f"💾 成功保存膜蛋白目标向量至: {os.path.abspath(SAVE_PATH)}")
    print(f"📊 向量维度: {target_embedding.shape}")

if __name__ == "__main__":
    main()