import faiss


class Extract_Index(object):

    def __init__(self, embeds, gpu=False, **kwargs):
        super().__init__()
        self.index = faiss.IndexFlatIP(kwargs['dimension'])
        self.index.add(embeds)
        if gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    def search(self, topk, query_vectors):
        distances, indices = self.index.search(query_vectors, topk)
        return distances, indices

# TODO HNSW index, PQ index
