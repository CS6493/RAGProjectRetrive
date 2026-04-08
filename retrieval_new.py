import json
import dill as pickle  # 修改：使用 dill 替代 pickle，以处理自定义类反序列化
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

# 添加 SimpleBM25 类定义（从预处理代码复制）
class SimpleBM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.tokens = self.tokenized_corpus
        self.avgdl = sum(len(t) for t in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.k1 = 1.5
        self.b = 0.75
        self.dfs = []
        self.tf = {}
        for doc in self.tokenized_corpus:
            df = {}
            for term in doc:
                df[term] = df.get(term, 0) + 1
                self.tf[term] = self.tf.get(term, 0) + 1
            self.dfs.append(df)
    
    def get_scores(self, query):
        qtok = query.lower().split()
        scores = []
        n = len(self.corpus)
        for i, doc in enumerate(self.tokens):
            s = 0
            dl = len(doc)
            for t in qtok:
                tf = self.dfs[i].get(t, 0)
                idf = np.log((n - self.tf.get(t, 0) + 0.5) / (self.tf.get(t, 0) + 0.5) + 1)
                s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

class RetrievalSystem:
    def __init__(self, bm25_index_path, faiss_index_path, metadata_path, chunks_path=None, doc_embeddings_path=None):
        # 加载BM25索引
        with open(bm25_index_path, 'rb') as f:
            self.bm25_data = pickle.load(f)

        if isinstance(self.bm25_data, dict) and 'bm25_model' in self.bm25_data:
            self.bm25 = self.bm25_data['bm25_model']
            chunks = self.bm25_data.get('chunks')
            self.metadata = self.bm25_data.get('metadata', {})
        else:
            self.bm25 = self.bm25_data
            chunks = None
            self.metadata = {}

        # 如果 BM25 索引对象自带 corpus 文本，则提取 chunks
        if chunks is None:
            if hasattr(self.bm25, 'corpus'):
                chunks = self.bm25.corpus
            elif hasattr(self.bm25, 'docs'):
                chunks = self.bm25.docs
            # 注意：不要从 tokenized_corpus 中创建 chunks，那不是原始文本
            # elif hasattr(self.bm25, 'tokenized_corpus'):
            #     chunks = [" ".join(tokens) for tokens in self.bm25.tokenized_corpus]

        # 不要在这里直接初始化 chunks，应该在后续从文件中加载
        self.chunks = {}
        self.doc_ids = []

        # 加载FAISS索引（contriever_optimized.index，内积等价余弦相似度）
        self.faiss_index_ip = faiss.read_index(faiss_index_path)

        # 初始化 doc_embeddings
        self.doc_embeddings = None
        self.doc_embeddings_norm = None

        # 加载分块元数据
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        if isinstance(self.metadata, list):
            converted = {}
            for i, item in enumerate(self.metadata):
                key = item.get('id', f"chunk_{i}")
                converted[key] = item
            self.metadata = converted

        # 尝试从 metadata 中提取文本字段（只在有必要的字段时）
        if not self.chunks and isinstance(self.metadata, dict):
            has_text = False
            for key, meta in self.metadata.items():
                if isinstance(meta, dict):
                    text = meta.get('chunk') or meta.get('text') or meta.get('content', "")
                    if text:  # 只保存有实际内容的
                        self.chunks[key] = text
                        has_text = True
            if has_text:
                self.doc_ids = list(self.chunks.keys())

        # 如果还没有获得有效的 chunks，从文件中加载
        if len(self.chunks) == 0:
            # 优先级1：stage1_chunks.pkl（标准数据源）
            stage1_path = "rag_preprocess_output/stage1_chunks.pkl"
            try:
                with open(stage1_path, 'rb') as f:
                    stage1_chunks = pickle.load(f)
                if isinstance(stage1_chunks, list):
                    self.chunks = {f"chunk_{i}": chunk for i, chunk in enumerate(stage1_chunks)}
                    self.doc_ids = list(self.chunks.keys())
                    print(f"Loaded {len(self.chunks)} chunks from {stage1_path}")
            except FileNotFoundError:
                # 优先级2：chunking_results 中的分块文件
                chunks_candidates = [
                    "chunking_results/chunks_512_51.json",
                    "chunking_results/chunks_1024_25.json",
                    "indexes/chunks_1024_102.json"
                ]
                expected_size = self.bm25.corpus_size if hasattr(self.bm25, 'corpus_size') else 26167
                
                for chunks_file in chunks_candidates:
                    try:
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            chunks_data = json.load(f)
                        if isinstance(chunks_data, list):
                            chunks_to_load = chunks_data[:expected_size]
                            self.chunks = {f"chunk_{i}": item.get('chunk', '') for i, item in enumerate(chunks_to_load)}
                            self.doc_ids = list(self.chunks.keys())
                            print(f"Loaded {len(self.chunks)} chunks from {chunks_file}")
                            break
                    except FileNotFoundError:
                        continue

        # 加载编码器（用于查询编码）
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        self.model = AutoModel.from_pretrained("facebook/contriever-msmarco")
        self.model.eval()

    def encode_query(self, query):
        """编码查询为向量"""
        with torch.no_grad():
            inputs = self.tokenizer([query], padding=True, truncation=True, return_tensors="pt")
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0].cpu().numpy()
        return emb

    def bm25_retrieve(self, query, top_k=3):
        """BM25检索"""
        scores = self.bm25.get_scores(query)  # 修改：传入原始 query string，SimpleBM25 内部会 tokenize
        topk_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in topk_idx:
            doc_id = self.doc_ids[idx]
            score = scores[idx]
            meta = self.metadata.get(doc_id, {})
            chunk_text = self.chunks.get(doc_id, "")
            results.append({
                'doc_id': doc_id,
                'score': score,
                'embedding': None,  # BM25没有embedding
                'metadata': meta,
                'chunk_text': chunk_text
            })
        return results

    def dense_retrieve(self, query, top_k=3, distance_metric='cosine'):
        """稠密检索，支持不同距离度量"""
        query_vec = self.encode_query(query)

        if distance_metric == 'cosine':
            # 使用 cosine 相似度：提取并归一化 doc embeddings，计算相似度
            if self.doc_embeddings_norm is None:
                print("提取并归一化文档 embeddings...")
                self.doc_embeddings = np.array([self.faiss_index_ip.reconstruct(i) for i in range(self.faiss_index_ip.ntotal)])
                self.doc_embeddings_norm = self.doc_embeddings / np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)

            query_vec_norm = query_vec / np.linalg.norm(query_vec)
            similarities = np.dot(self.doc_embeddings_norm, query_vec_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[top_indices]
            indices = top_indices
            score_type = 'similarity'
        elif distance_metric == 'l2':
            # 使用 L2 距离
            scores, indices = self.faiss_index_ip.search(query_vec, top_k)
            score_type = 'distance'
        else:
            raise ValueError("Unsupported distance metric")

        results = []
        for i, (score, idx) in enumerate(zip(scores.flatten(), indices.flatten())):
            doc_id = self.doc_ids[idx]
            meta = self.metadata.get(doc_id, {})
            chunk_text = self.chunks.get(doc_id, "")
            
            # 获取 embedding
            if self.doc_embeddings is not None:
                embedding = self.doc_embeddings[idx]
            else:
                # 动态计算 chunk embedding
                with torch.no_grad():
                    inputs = self.tokenizer([chunk_text], padding=True, truncation=True, return_tensors="pt")
                    emb = self.model(**inputs).last_hidden_state[:, 0].cpu().numpy()
                embedding = emb
            
            results.append({
                'doc_id': doc_id,
                'score': score,
                'score_type': score_type,
                'embedding': embedding,
                'metadata': meta,
                'chunk_text': chunk_text
            })
        return results

    def retrieve(self, query, method='bm25', top_k=3, distance_metric='cosine', save_to_json=False, output_path=None):
        """统一检索接口"""
        if method == 'bm25':
            results = self.bm25_retrieve(query, top_k)
            metric = 'bm25_score'
        elif method == 'dense':
            results = self.dense_retrieve(query, top_k, distance_metric)
            metric = distance_metric
        else:
            raise ValueError("Unsupported method")

        # 格式化结果
        formatted_results = []
        for rank, res in enumerate(results, 1):
            chunk_id = int(res['doc_id'].split('_')[1]) if '_' in res['doc_id'] else 0
            formatted_results.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "score": float(res['score']),
                "text": res['chunk_text'],
                "embedding": res['embedding'].tolist() if res['embedding'] is not None else None,
                "metadata": res['metadata']
            })

        response = {
            "query": query,
            "method": method,
            "metric": metric,
            "top_k": top_k,
            "results": formatted_results
        }

        if save_to_json:
            if output_path is None:
                output_path = f"retrieval_results_{method}_{top_k}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=2)
            return output_path  # 返回文件路径
        
        return response

    def save_results(self, results, output_path):
        """保存检索结果到JSON文件"""
        json_results = []
        for res in results:
            json_res = {}
            for key, value in res.items():
                if isinstance(value, np.ndarray):
                    json_res[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    json_res[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    json_res[key] = int(value)
                else:
                    json_res[key] = value
            json_results.append(json_res)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        return output_path
if __name__ == "__main__":
    # 修改：使用新的数据文件结构
    bm25_path = "indexes_optimized_final/bm25_size_512_overlap_10.pkl"
    faiss_path = "indexes_optimized_final/contriever_optimized.index"
    metadata_path = "indexes_optimized_final/contriever_metadata.json"
    chunks_path = None
    embeddings_path = None
    queries_path = "data/queries.json"

    system = RetrievalSystem(bm25_path, faiss_path, metadata_path, chunks_path, embeddings_path)

    # 加载查询
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print("=== 检索系统 ===")
    print(f"加载了 {len(queries)} 个查询")

    # 创建 result 目录
    import os
    os.makedirs("result", exist_ok=True)

    # 选择运行前多少查询
    while True:
        try:
            num_queries = int(input(f"运行前多少个查询？(1-{len(queries)}): "))
            if 1 <= num_queries <= len(queries):
                break
            else:
                print("无效输入")
        except ValueError:
            print("请输入数字")

    # 选择检索方法
    while True:
        method_choice = input("选择检索方法 (1: BM25, 2: Dense): ")
        if method_choice == '1':
            method = 'bm25'
            distance_metric = None
            break
        elif method_choice == '2':
            method = 'dense'
            # 选择距离度量
            while True:
                dist_choice = input("选择距离度量 (1: Cosine, 2: L2): ")
                if dist_choice == '1':
                    distance_metric = 'cosine'
                    break
                elif dist_choice == '2':
                    distance_metric = 'l2'
                    break
                else:
                    print("无效选择")
            break
        else:
            print("无效选择")

    # 选择 top_k
    while True:
        try:
            top_k = int(input("选择 top_k (1-10): "))
            if 1 <= top_k <= 10:
                break
            else:
                print("无效输入")
        except ValueError:
            print("请输入数字")

    print(f"\n将运行前 {num_queries} 个查询，使用 {method.upper()} Top-{top_k}")

    # 批量运行
    for i in range(num_queries):
        selected_query = queries[i]['query']
        print(f"\n处理查询 {i+1}: {selected_query}")

        if method == 'bm25':
            output_filename = f"result/retrieval_results_bm25_{top_k}_{i+1}.json"
        else:
            output_filename = f"result/retrieval_results_dense_{distance_metric}_{top_k}_{i+1}.json"

        json_path = system.retrieve(
            query=selected_query,
            method=method,
            top_k=top_k,
            distance_metric=distance_metric,
            save_to_json=True,
            output_path=output_filename
        )

        print(f"结果保存到: {json_path}")

    print("\n所有查询处理完成！")
    print("感谢使用检索系统！")