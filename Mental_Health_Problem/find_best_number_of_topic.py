from gensim.models import CoherenceModel, LdaModel
from tqdm import tqdm

def train_and_compute_coherence(
    dictionary, corpus, texts, coherence_type, start=2, limit=10, alpha='auto', eta='auto', passes=10):
    values = []
    for num_topics in tqdm(range(start, limit + 1), desc="Training LDA models"):
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=eta,
            passes=passes
        )
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence=coherence_type
        )
        values.append(coherence_model.get_coherence())
    return values

def score_coherence(
    dictionary, corpus, texts, coherence_type, start=2, limit=10, num_runs=5, alpha='auto', eta='auto', passes=10):
    all_runs = []
    for i in tqdm(range(num_runs), desc="Running coherence scores"):
        scores = train_and_compute_coherence(
            dictionary,
            corpus,
            texts,
            coherence_type=coherence_type,
            start=start,
            limit=limit,
            alpha=alpha,
            eta=eta,
            passes=passes
        )
        all_runs.append(scores)
    return all_runs
