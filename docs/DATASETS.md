# Dataset sources, citations, and license status

No raw data is included. The links below identify intended sources; users remain responsible for checking current terms and any constituent-dataset restrictions.

## English — CoMB

- Paper statistics: 45,488 texts; 62.2% humorous.
- Citation: Baranov, Kniazhevsky, and Braslavski, “You Told Me That Joke Twice: A Systematic Investigation of Transferability and Robustness of Humor Detection Models,” EMNLP 2023.
- Intended project source: `https://github.com/Humor-Research/Humor-detection` and its documented `hri_tools` preparation utilities.
- License: not confirmed for redistribution of the aggregated raw corpus. CoMB combines material from several sources, so a code-repository license alone does not establish data redistribution permission.
- Archive status: no raw CoMB file and no complete preparation/split manifest were present.

## Spanish — Martínez jokes

- Paper statistics: 2,419 texts; 63.1% humorous.
- Intended source: `https://github.com/liopic/chistes-nlp`.
- Paper mapping: rating 1 -> 0; ratings 2 or 3 -> 1.
- It is explicitly not HAHA/IberLEF 2018.
- License: not confirmed in the inspected source material.
- Archive status: scripts use an old Hugging Face dataset name, but no hash proves exact equivalence to the cited repository.

## Portuguese

- Paper statistics: 2,800 texts; 50.0% humorous.
- Citation: Gonçalo Oliveira, Clemêncio, and Alves, “Corpora and Baselines for Humour Recognition in Portuguese,” LREC 2020.
- Intended source: `https://github.com/NLP-CISUC/Recognizing-Humor-in-Portuguese`.
- License: not confirmed in the inspected source material.

## Chinese — HumorWB/Weibo

- Paper statistics: 5,291 texts; 38.0% humorous.
- Citation: Zeng et al., “Leveraging Social Context for Humor Recognition and Sense of Humor Evaluation in Social Media with a New Chinese Humor Corpus — HumorWB,” LREC-COLING 2024.
- Intended source: `https://github.com/zzyjerry/HumorWB`.
- License: not confirmed in the inspected source material.

## Hindi-English

- Paper statistics: 2,949 texts; 59.6% humorous.
- Citation: Khandelwal et al., “Humor Detection in English-Hindi Code-Mixed Social Media Content: Corpus and Baseline System,” LREC 2018.
- Intended source: `https://github.com/Ankh2295/humor-detection-corpus`.
- License: the repository advertises GPL-3.0, but whether that license unambiguously covers the corpus content must be confirmed before redistribution.

## Redistribution policy

The repository stores normalized data and split manifests only in ignored local paths. Publish a raw or normalized dataset only after confirming that its license and every upstream source permit redistribution. When a license cannot be confirmed, distribute preparation code and identifier/hash manifests rather than text.
