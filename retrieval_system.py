import json
import pickle
import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel

class RetrievalSystem:
    def __init__(self, bm25_index_path, faiss_index_path, metadata_path, chunks_path, doc_embeddings_path=None):
        # 加载BM25索引
        with open(bm25_index_path, 'rb') as f:
            self.bm25_data = pickle.load(f)
        self.bm25 = self.bm25_data['bm25']
        self.tokenized_docs = self.bm25_data['tokenized_docs']
        self.doc_ids = self.bm25_data['doc_ids']

        # 加载FAISS索引（假设是IP索引，用于cosine）
        self.faiss_index_ip = faiss.read_index(faiss_index_path)

        # 如果有文档embeddings，创建L2索引
        self.doc_embeddings = None
        self.faiss_index_l2 = None
        if doc_embeddings_path:
            with open(doc_embeddings_path, 'rb') as f:
                self.doc_embeddings = pickle.load(f)
            # 创建L2索引
            dim = self.doc_embeddings.shape[1]
            self.faiss_index_l2 = faiss.IndexFlatL2(dim)
            self.faiss_index_l2.add(self.doc_embeddings)

        # 加载元数据
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # 加载分块文本
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

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
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
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
            # 使用内积索引（假设向量已归一化）
            faiss.normalize_L2(query_vec)
            scores, indices = self.faiss_index_ip.search(query_vec, top_k)
            score_type = 'similarity'
        elif distance_metric == 'l2':
            if self.faiss_index_l2 is None:
                raise ValueError("L2 index not available. Provide doc_embeddings_path.")
            # 使用L2索引
            scores, indices = self.faiss_index_l2.search(query_vec, top_k)
            score_type = 'distance'
        else:
            raise ValueError("Unsupported distance metric")

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc_id = self.doc_ids[idx]
            meta = self.metadata.get(doc_id, {})
            chunk_text = self.chunks.get(doc_id, "")
            embedding = self.doc_embeddings[idx] if self.doc_embeddings is not None else None
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
        elif method == 'dense':
            results = self.dense_retrieve(query, top_k, distance_metric)
        else:
            raise ValueError("Unsupported method")

        if save_to_json:
            if output_path is None:
                output_path = f"retrieval_results_{method}_{top_k}.json"
            # 转换numpy类型为Python基本类型（JSON兼容）
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
            return output_path  # 返回文件路径
        
        return results

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
    # 假设文件路径
    bm25_path = "data/bm25_index.pkl"
    faiss_path = "data/dense_retrieval_index.faiss"
    metadata_path = "data/chunk_metadata.pkl"
    chunks_path = "data/processed_chunks.json"
    embeddings_path = "data/doc_embeddings.pkl"  # 可选，用于L2
    queries_path = "data/queries.json"

    system = RetrievalSystem(bm25_path, faiss_path, metadata_path, chunks_path, embeddings_path)

    # 加载查询
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    print("=== 检索系统 ===")
    print(f"加载了 {len(queries)} 个查询")

    while True:
        # 显示查询列表（前5条）
        print("\n可用查询 (显示前5条):")
        for i, q in enumerate(queries[:5]):
            print(f"{i+1}. {q['query']}")
        print(f"6. 输入自定义查询")
        if len(queries) > 5:
            print(f"   (还有 {len(queries)-5} 个查询未显示)")

        # 选择查询
        while True:
            try:
                query_choice = input(f"\n请选择查询编号 (1-6): ")
                if query_choice == '6':
                    # 自定义查询
                    selected_query = input("请输入您的查询: ").strip()
                    if not selected_query:
                        print("查询不能为空，请重新输入")
                        continue
                    query_idx = -1  # 标记为自定义查询
                    break
                else:
                    query_idx = int(query_choice) - 1
                    if 0 <= query_idx < len(queries):
                        selected_query = queries[query_idx]['query']
                        break
                    else:
                        print("无效的选择，请重新输入")
            except ValueError:
                print("请输入有效的数字")

        if query_idx == -1:
            print(f"\n自定义查询: {selected_query}")
        else:
            print(f"\n选择的查询: {selected_query}")

        # 选择检索方法
        print("\n选择检索方法:")
        print("1. BM25 (稀疏检索)")
        print("2. Dense (稠密检索)")
        while True:
            method_choice = input("请选择 (1或2): ")
            if method_choice == '1':
                method = 'bm25'
                distance_metric = None
                break
            elif method_choice == '2':
                method = 'dense'
                # 选择距离度量
                print("\n选择距离度量:")
                print("1. Cosine 相似度")
                print("2. L2 距离")
                while True:
                    dist_choice = input("请选择 (1或2): ")
                    if dist_choice == '1':
                        distance_metric = 'cosine'
                        break
                    elif dist_choice == '2':
                        distance_metric = 'l2'
                        break
                    else:
                        print("无效的选择")
                break
            else:
                print("无效的选择")

        # 选择top-k
        print("\n选择返回结果数量:")
        print("1. Top-3")
        print("2. Top-5")
        while True:
            top_choice = input("请选择 (1或2): ")
            if top_choice == '1':
                top_k = 3
                break
            elif top_choice == '2':
                top_k = 5
                break
            else:
                print("无效的选择")

        # 执行检索
        print(f"\n执行检索: {method.upper()} Top-{top_k}")
        if method == 'dense':
            print(f"距离度量: {distance_metric}")

        # 检索并保存
        if method == 'bm25':
            if query_idx == -1:
                output_filename = f"retrieval_results_bm25_{top_k}_custom.json"
            else:
                output_filename = f"retrieval_results_bm25_{top_k}_{query_idx+1}.json"
        else:
            if query_idx == -1:
                output_filename = f"retrieval_results_dense_{distance_metric}_{top_k}_custom.json"
            else:
                output_filename = f"retrieval_results_dense_{distance_metric}_{top_k}_{query_idx+1}.json"

        json_path = system.retrieve(
            query=selected_query,
            method=method,
            top_k=top_k,
            distance_metric=distance_metric,
            save_to_json=True,
            output_path=output_filename
        )

        print(f"检索完成！结果已保存到: {json_path}")

        # 显示结果摘要
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"\n检索到 {len(results)} 个结果:")
        for i, res in enumerate(results, 1):
            print(f"{i}. Doc ID: {res['doc_id']}, Score: {res['score']:.4f}")
            if res['metadata']:
                print(f"   Metadata: {res['metadata']}")

        # 询问是否继续
        cont = input("\n是否继续检索其他查询？(y/n): ").lower()
        if cont != 'y':
            print("感谢使用检索系统！")
            break