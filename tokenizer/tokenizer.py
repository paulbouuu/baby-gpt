import regex as re

from tqdm import tqdm


class Tokenizer():

    def __init__(self, data_path, pattern, num_merges=20, merges=None):
        self.pattern = re.compile(pattern)
        self.num_merges = num_merges
        self.itob = {i: bytes([i]) for i in range(256)} # special tokens will be added later
        if merges:
            for i, pair in enumerate(merges):
                self.itob[256 + i] = self.itob[pair[0]] + self.itob[pair[1]]
            self.merges, self.vocab_size = merges, 256 + len(merges)
        else:
            self.merges, self.vocab_size = self._train_bpe(data_path)
        
    def _get_stats(self, ids):
        counts = {}
        for i in range(len(ids)-1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        a, b = pair
        n = len(ids)
        newids = []
        i = 0
        while i < n:
            if i + 1 < n and ids[i] == a and ids[i + 1] == b:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def _train_bpe(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = self.pattern.findall(text)

        # convert each chunk to a list of byte ids
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = []

        for i in tqdm(range(self.num_merges)):

            next_idx = 256 + i

            # count pairs
            total_counts = {}
            for ids in chunk_ids:
                stats = self._get_stats(ids)
                for k, v in stats.items():
                    total_counts[k] = total_counts.get(k, 0) + v

            if not total_counts:
                break

            top_pair = max(total_counts, key=total_counts.get)

            # new id
            a, b = top_pair
            self.itob[next_idx] = self.itob[a] + self.itob[b]

            chunk_ids = [self._merge(ids, top_pair, next_idx) for ids in chunk_ids]

            merges.append(top_pair)

        vocab_size = len(merges) + 256

        return merges, vocab_size
    
    def encode(self, text):

        # split to chunks
        chunks = self.pattern.findall(text)
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        # apply merges
        for i, pair in enumerate(self.merges):
            new_idx = 256 + i
            chunk_ids = [self._merge(ids, pair, new_idx) for ids in chunk_ids]

        ids = [tid for ids in chunk_ids for tid in ids]
        return ids

    def decode(self, ids):
        tokens = b"".join(self.itob[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text