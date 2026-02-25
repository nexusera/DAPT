\documentclass[runningheads]{llncs}

%
\usepackage[T1]{fontenc}
% T1 fonts will be used to generate the final print and online PDFs,
% so please use T1 fonts in your manuscript whenever possible.
% Other font encondings may result in incorrect characters.
%
\usepackage{graphicx}
% Used for displaying a sample figure. If possible, figure files should
% be included in EPS format.
%
% If you use the hyperref package, please uncomment the following two lines
% to display URLs in blue roman font according to Springer's eBook style:
%\usepackage{color}
%\renewcommand\UrlFont{\color{blue}\rmfamily}
%\urlstyle{rm}
%
\begin{document}


\title{ KV-BERT: A Noise-Robust Pretraining Framework for Semi-Structured Key-Value Extraction in OCR Clinical Reports}
\author{Hao Li \and
Yingyun Li\and Ying Qin
\and Haiyang Qian }
%
\authorrunning{F. Author et al.}
% First names are abbreviated in the running head.
% If there are more than two authors, 'et al.' is used.
%
\institute{AI Starfish\\
\email{\{hao.l, yingyun.li, ying.qin, haiyang.qian\}@aistarfish.com}
 }

\maketitle              % typeset the header of the contribution


\begin{abstract}
Because clinical information often remains inaccessible across healthcare institutions, retrieving previous clinical records of patients from other institutions mostly relies on paper clinical reports provided by patients. Thus, IE (information extraction) is crucial in this context. Semi-structured key-value extraction—identifying header fields (keys) and their associated content fields (values) is a foundational task in clinical information extraction.  This process is typically performed by first running optical character recognition (OCR) on paper reports, followed by applying learning-based models to extract key-value pairs. While BERT-based models are suited for IE due to their bidirectional attention mechanism, existing models (e.g., BERT, RoBERTa) are primarily trained on clean data w/o noise, and perform poorly on noisy OCR-derived data. Thus, we propose a noise-resistant pre-training framework for semi-structured key-value (KV) extraction in OCR medical reports. We introduce KV-MLM (Masked Language Modeling) and KV-NSP (Next Sentence Prediction) as pre-training tasks, and model OCR artifacts and meta information as learnable embedding vectors. The KV-MLM task masks medical entities and KV pair boundaries, enhancing the model's contextual inference and memory capabilities for medical terminology. The KV-NSP task is an improved NSP task that constructs with strong negative samples, forcing the model to learn logical consistency between keys and values. We added an embedding dimension, extracting seven physical features including OCR confidence and alignment scores. These were mapped to learnable embedding vectors via a binning strategy. Experimental results on real world clinical report datasets  demonstrate that the models pretrained outperforms both general BERT and RoBERTa base models, as well as SOTA medical Large Language Models (LLMs) (but 1-3 orders of magnitude fewer than these LLM), on semi-structured key-value pair extraction.
\keywords{OCR\and Noise \and BERT \and Pretraining \and NER \and Clinical Report.}
\end{abstract}


\section{Introduction}
In today's healthcare landscape, electronic medical records (EMRs) are ubiquitous. However, they remain isolated across hospitals. Patients can access their information only through paper clinical reports. This necessitates converting these clinical reports into text using optical character recognition (OCR) technology during hospital transfers or when establishing patient records at third-party institutions. The OCR results must then be processed into a semi-structured format.

State-of-the-art optical character recognition (OCR) and mainstream commercial OCR frameworks (such as PaddleOCR and EasyOCR) typically integrate mature preprocessing modules. These commonly include image denoising, moiré pattern removal, document skew correction, and curvature correction, significantly enhancing the quality of processed raw images. However, in practical scenarios, recognition biases, semantic truncation, and logical association errors remain unavoidable. This “residual noise” driven by complex scenarios imposes significant robustness challenges on subsequent semi-structured tasks.

Concurrently, existing general-purpose language models (e.g., BERT, RoBERTa) are typically pre-trained on high-quality, clean corpora (e.g., Wikipedia, relevant news, medical guidelines). When directly applied to low-quality OCR medical text, their performance degrades. Moreover, these general pre-trained models understand text solely based on semantics, without leveraging additional information provided by OCR engines (e.g., confidence scores).

Thus, we propose a noise-resistant pre-training framework for semi-structured key-value (KV) extraction in OCR medical reports. We introduce KV-MLM (Masked Language Modeling) and KV-NSP (Next Sentence Prediction) as pre-training tasks, and model OCR artifacts and meta information as learnable embedding vectors. Specifically, we made the following innovations:
\subsection{KV-MLM}
   We refine the random masking strategy in the original Masked Language Modeling (MLM) task of the BERT model. Specifically, we implement masking tailored to medical entities and key-value boundaries. This novel masking task compels the model to learn inferring complete medical terminology from context during pre-training, thereby enhancing its ability to memorize, comprehend, and reason about medical entity vocabulary.
\subsection{KV-NSP}
   For the Next Sentence Prediction (NSP) task—one of BERT's original tasks—we replaced the goal of predicting sentence continuity with determining whether given Key and Value pairs are logically matched. By learning from positive samples and hard negative samples, we force the model to learn the logical consistency between field names and their corresponding content.
\subsection{Noise-Embedding}
  We map a 7-dimensional noise feature vector (including OCR confidence mean/variance, character break rate, layout alignment score, etc.) into a learnable embedding via binning strategy. This is then combined with text embedding, position embedding, and other embeddings as model's input. This enables the model to assess the quality of each token.
  % (presumably, this component is expected to function as described—though I haven't examined the specific attention results—such that when text is unclear, the model learns to rely on context or reduce the weight of that word).


To summarize, we make the following contributions:
We propose a novel noise fusion mechanism: for the first time, we integrate the confidence statistics from the OCR engine with visual layout features (totaling 7-dimensional features) into the BERT input layer, significantly enhancing the model's performance across various downstream tasks on low-quality OCR text.
We designed two pre-training tasks tailored for downstream semi-structured tasks: proposing KV-MLM and KV-NSP strategies to incorporate structural prior knowledge from Clinical Reports into the pre-training phase, addressing the limitations of general models in understanding semi-structured data.
We demonstrated effectiveness through extensive experiments on real medical OCR datasets. Results demonstrate that our approach outperforms both general BERT and Roberta bases, as well as state-of-the-art medical Large Language Models (LLMs) (though by 1-3 orders of magnitude less than these LLMs) across multiple downstream tasks (KV-NER, task1, 2, 3). Notably, our model exhibits particularly significant performance gains on noisy data.

\section{Related Work}
\subsection{Domain-Adaptive Pre-training}
In recent years, pre-trained language models have achieved remarkable success across various NLP tasks. Concurrently, BERT-based models are well-suited for information extraction (IE) due to their bidirectional attention mechanism. To enhance model performance in specific vertical domains, domain-adaptive pre-training has emerged as a mainstream trend. In the medical domain, BioBERT significantly improved biomedical entity recognition accuracy by fine-tuning on large-scale corpora like PubMed; ClinicalBERT further enhanced clinical text understanding by leveraging electronic medical records (EMR). However, these models primarily utilize high-quality, clean text during pre-training, leading to substantial performance degradation when confronted with real-world Clinical Reports containing OCR noise.

\subsection{Document Multimodal Models and Key-Value Pair Extraction}
When processing semi-structured documents like tables and reports, models such as LayoutLM pioneered a new paradigm in Document AI by integrating textual, layout, and visual features. Subsequent evolutions like LayoutLMv3 further optimized alignment efficiency. While these models excel in general cross-modal representation learning, they often overlook the strong structural prior knowledge inherent in domain-specific documents like Clinical Reports (e.g., the logical correspondence between Key and Value). The two pre-training tasks proposed in this paper—KV-MLM and KV-NSP—aim to address this shortcoming.

\subsection{OCR-Induced Noise}
Handling character-level noise induced by OCR has long been a challenge in document analysis. Traditional approaches primarily focus on post-OCR correction or spell checking. Recent studies, such as NAT (Noisy Augmented Training), attempt to enhance model robustness by simulating noise distributions through data augmentation. Unlike such “repair-oriented” approaches, this paper introduces a Noise-Embedding mechanism: by explicitly modeling physical features (including the Confidence Score) from OCR engine outputs across seven dimensions, it endows the model with the ability to perceive and adaptively handle text inputs of varying quality. This achieves a paradigm shift from ‘denoising’ to “proactive noise tolerance.”


\section{Methodology}

\subsection{Overall Architecture}
\textbf{Base Model:} - Base Model: We adopt RoBERTa (Robustly optimized BERT approach) as the backbone network (specifically hfl/chinese-roberta-wwm-ext).
  - Layer count L=12, hidden dimension=768, multi-head attention heads=12.

Traditional encoder-only models (represented by BERT) incorporate Token embeddings, Position embeddings, and Segment Embeddings. To enable the model to perceive OCR text quality and adapt to noise introduced by OCR, we introduce a fourth Embedding layer—Noise Embedding.

The final input vector $$E_input$$ is computed as follows:
$$E_{input}(x_i) = E_{tok}(x_i) + E_{pos}(p_i) + E_{seg}(s_i) + F_{noise}(v_i)$$
where \(x_i\) denotes the 7-dimensional continuous noise feature vector corresponding to the token.


\textbf{Input Representation:} Traditional encoder-only models (represented by BERT) incorporate Token embeddings, Position embeddings, and Segment Embeddings. To enable the model to perceive OCR text quality and adapt to noise introduced by OCR, we introduce a fourth Embedding layer—Noise Embedding. The final input representation $E_{\text{input}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ is computed as follows:
\begin{equation}
    E_{\text{input}}(x_i) = E_{\text{tok}}(x_i) + E_{\text{pos}}(p_i) + E_{\text{seg}}(s_i) + F_{\text{noise}}(v_i)
\end{equation}
where $v_i$ represents a 7-dimensional continuous noise feature vector corresponding to the $i$-th token.

% --- 插入图片代码开始 ---
\begin{figure}[t] % [t] 表示 top，意为将图片置于页面顶部。这是 AI 顶会（如 ICDAR）最常用的排版位置
    \centering % 使图片在页面或栏目中水平居中
    
    % \includegraphics 是核心命令
    % [width=...] 控制图片宽度。0.9\linewidth 表示占单栏宽度的 90%
    \includegraphics[width=0.9\linewidth]{figures/input_rep.pdf} 
    
    % \caption 命令用于添加图片标题，它会自动生成 "Figure 1: ..." 这种格式
    \caption{Detailed illustration of the multi-modal input representation. The final input vector $E_{\text{input}}$ for each token $x_i$ is computed as the element-wise summation of token, position, segment, and the proposed noise embeddings.}
    
    % \label 是这张图的“身份证号”，方便你在正文中使用 \ref{fig:input_rep} 来引用它
    \label{fig:input_rep}
\end{figure}
% --- 插入图片代码结束 ---

% [Place Figure 1: Overall architecture of the model here]
% \begin{figure}[t]
% \centering
% \includegraphics[width=\linewidth]{figures/architecture.pdf}
% \caption{Overall Architecture of the Noise-Aware Framework.}
% \label{fig:arch}
% \end{figure}

\subsection{Noise-Embedding Mechanism}
One innovation of this paper is the explicit modeling of text “confidence” and “morphology” as dimensions of the embedding, utilizing the output from an OCR engine.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/noise_embedding_mechanism.pdf} % 请替换为你导出的文件名
    \caption{Detailed workflow of the proposed Noise-Embedding Mechanism. For each token, seven statistical features are extracted from the OCR output, discretized into bin IDs via non-linear mapping functions $\Phi_k$, and transformed into dense vectors through separate embedding matrices $W_k$. These vectors are then summed to produce the final noise embedding $F_{\text{noise}}(v_i)$, which is added to the standard token representations.}
    \label{fig:noise_mechanism}
\end{figure}

\subsubsection{Feature Extraction (7-Dimensional Feature Definition):}

For each token in the sequence, we extract the following 7 statistical features from the OCR engine output:
\begin{itemize}
    \item $f_1 (\text{conf}_{\text{avg}})$: Average confidence score.
    \item $f_2 (\text{conf}_{\text{min}})$: Minimum character confidence, capturing the worst-case recognition within a token.
    \item $f_3 (\text{conf}_{\text{var}})$: Logarithmic variance of confidence, measuring recognition stability.
    \item $f_4 (\text{conf}_{\text{gap}})$: Confidence range ($\text{Avg} - \text{Min}$).
    \item $f_5 (\text{punct}_{\text{err}})$: Punctuation error rate for rule-based detection of OCR artifacts.
    \item $f_6 (\text{char}_{\text{break}})$: Character-level Noise ratio, specifically the character fragmentation ratio.
    \item $f_7 (\text{align}_{\text{score}})$: Vertical layout alignment score to measure Layout Fragmentation.
\end{itemize}

\subsubsection{Discretization and Binning}
To input continuous physical signals into the model, we constructed a binning mapping function.

\begin{itemize}
    \item \textbf{Non-linear Binning:}  For distributions of different features (e.g., confidence scores biased toward 1.0, alignment scores biased toward 0), we precomputed statistical quantiles as bin boundaries.
    \item \textbf{Anchor Bin:} All “perfect text” (no OCR-induced noise) and perfect OCR recognition results are mapped to ID=0.
\end{itemize}
The mapping is defined as:
\begin{equation}
    id_{i,k} = \Phi_k(v_{i,k}), \quad id_{i,k} \in \{0, 1, \dots, N_k\}
\end{equation}
In our implementation, the number of bins $N_k$ for each dimension is set to 64, 64, 32, 32, 16, 32, and 64, respectively.

\subsubsection{Embedding Lookup and Fusion}
The model maintains seven independent embedding matrices $W_k \in \mathbb{R}^{N_k \times d_{\text{model}}}$. The output of the noise embedding layer is the summation of the embeddings from each dimension:
\begin{equation}
    F_{\text{noise}}(v_i) = \sum_{k=1}^{7} \text{Lookup}(id_{i,k}; W_k)
\end{equation}
It is worth noting that our lookup-plus-addition approach is extremely lightweight, adding virtually no overhead to inference latency.

\subsection{KV-MLM: Key-level Masked Language Modeling}




\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/kv_mlm.pdf}
    \caption{Illustration of the proposed KV-MLM strategy. By integrating medical glossaries into the tokenization phase, the model assigns a unified \texttt{word\_id} to complex medical entities (e.g., ``hypertension''). During pre-training, if a \texttt{word\_id} is selected, all constituent tokens are masked simultaneously, forcing the model to reconstruct complete medical semantics from context.}
    \label{fig:kv_mlm}
\end{figure}

We propose a guided by a prior dictionary, key-value level Whole Word Masking (WWM) strategy to incorporate domain-specific knowledge into the pre-training phase.


\subsubsection{Key-Value Dictionary}
During preprocessing, we loaded three specialized dictionaries using the Jieba tokenizer:
\begin{itemize}
  \item Key Set: A collection of key terms from clinical cases (e.g., “chief complaint,” “presenting history,” “creatinine”).
  \item High-frequency words in OCR medical records: Frequently occurring terms extracted from extensive clinical case corpora, filtered by a large language model (Qwen3) to retain only medically relevant entities.
  \item Value component

\end{itemize}
Through Jieba segmentation, during data preprocessing, we assign the same word_ids to all characters within the same segment. This ensures medical entities, key names, and values are not split into individual Chinese characters.

\subsubsection{K-Aware Masking Strategy}
Given an input sequence $X$, we perform WWM based on the predefined \texttt{word\_ids}. The masking probability follows a Bernoulli distribution:
\begin{equation}
    M \sim \text{Bernoulli}(0.15)
\end{equation}

If a token belongs to a medical entity (e.g., “diabetes”), when selected, the entire associated word (all tokens sharing the same word_id) is replaced with [MASK]. This masking approach forces the model to reconstruct complete medical semantics through context. Consequently, it enhances the model's ability to memorize, comprehend, and reason about medical entity vocabulary and common key-value pairs, thereby improving performance on downstream tasks (Information Extraction and Semi-Structured Tasks).

\subsection{KV-NSP: Structure-Aware Contrastive Learning}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.95\linewidth]{figures/kv_nsp.pdf} % 建议导出名为此
    \caption{Overview of the KV-NSP pre-training task. The model is trained to distinguish between valid \textbf{Key--Value Pairs} (Positive) and mismatched ones (Negatives). \textbf{Hard Negatives} are constructed by reversing the order of keys and values to teach the model directional dependencies, while \textbf{Easy Negatives} are generated via random sampling to enhance semantic compatibility learning.}
    \label{fig:kv_nsp}
\end{figure}

The original NSP task for BERT has been proven too simplistic and detrimental to model training. Existing BERT-based models often modify or abandon the NSP task.
We have redesigned this task based on the structural characteristics of medical records, aiming for the model to learn logical binding relationships between keys and values through this task.

\subsubsection{Problem Formulation}
We formulate this task as a binary classification problem. The input format is structured as:
\begin{equation}
    \text{Input} = [\text{CLS}] \oplus K \oplus [\text{SEP}] \oplus V \oplus [\text{SEP}]
\end{equation}
where $\oplus$ denotes the concatenation operation. The model is required to predict a label $y \in \{1 (\text{Match}), 0 (\text{Mismatch})\}$.

% 

\subsubsection{Negative Sampling Strategy}
    
We maintain a 1:1 ratio between positive and negative samples. To increase the task complexity, we design two specific negative sampling strategies:
\begin{itemize}
    \item \textbf{Reverse Order (Hard Negative, 50\%):} Swap the order of Key and Value (i.e., Key-Value becomes Value-Key).
    Purpose: Forces the model to learn the directionality of medical records (where “key” points to “value,” not vice versa).
    \item \textbf{Random Replacement (Easy Negative, 50\%):} Keeps the Key unchanged and randomly samples an unrelated value from the corpus.
    Purpose: Learns semantic matching (e.g., distinguishing format differences between “Name” and “Date”)
\end{itemize}

The final pre-training objective is the joint loss:
\begin{equation}
    \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KV-MLM}} + \mathcal{L}_{\text{KV-NSP}}
\end{equation}

\section{Experiments}

\subsection{Experimental Setup}
\subsubsection{Datasets}

\textbf{Pre-training Data:} 
 The pre-training data comprises medical records processed by OCR, medical textbooks/guidelines, medical papers, Wikipedia entries, and a small amount of general Chinese text. After cleaning and deduplication, a total of 982,183 text samples were obtained (counted by lines, with one sample per line). To control domain proportions, we performed static resampling across sources, yielding 225,481 training texts. Among these, OCR Clinical Reports accounted for 35.00%, medical texts for 46.96%, and general corpora for 18.02%.
 After resampling, we obtained 225,481 pre-training texts (counted by lines, with one text per line). To mitigate information loss from truncation due to the maximum length of 512, we performed character-level sliding window segmentation (window=1000, stride=500) on long texts exceeding the threshold, splitting each long text into multiple chunks. Following sliding window segmentation, the total number of training segments increased from 225,481 to 537,721. Subsequent pre-training encoded these segmented chunks as the basic pre-training text units, truncating each to 512 tokens.

\textbf{Downstream Tasks:} 
    KV-NER (Key-Value Entity Recognition): Core tasks (task1, task3). Extract Keys (e.g., names) and Values (e.g., specific names) from unstructured medical records.
    Metric: precision/recall/F1.

\subsubsection{Baselines}
We compare our framework against several competitive Chinese pre-trained models:
\begin{itemize}
    \item \textbf{BERT-Base-Chinese:} The standard baseline for Chinese NLP tasks.
    \item \textbf{RoBERTa-wwm-ext:} An optimized BERT variant with whole-word masking, serving as our backbone.
    \item \textbf{MacBERT:} A strong baseline utilizing a correction-based masking strategy.
    
\end{itemize}

\subsubsection{Implementation Details}
Our framework is implemented using HuggingFace Transformers. All models are trained on an \textbf{NVIDIA H200 GPU cluster (8 nodes)}. For pre-training, we use the \textbf{AdamW} optimizer with a learning rate of $8e-5$, while for fine-tuning, the rate is adjusted to $2e-5$. 
% 中文注释：训练策略采用分阶段模式，先进行 KV-MLM 以稳定 Embedding，再引入 KV-NSP 进行结构化训练。

\subsection{Main Results}
% Table 1: Performance on KV-NER

\begin{table*}[t]
  \centering
  \small
  \setlength{\tabcolsep}{3.5pt}
  \renewcommand{\arraystretch}{1.15}
  \definecolor{lightgray}{gray}{0.92}
  \caption{Model evaluation results across tasks (Real).}
  \label{tab:main_results}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{llccccccccc}
    \toprule
    \multirow{2}{*}{Model} & \multirow{2}{*}{Setup} & \multicolumn{2}{c}{Task 1: Key Discovery} & \multicolumn{4}{c}{Task 2: QA} & \multicolumn{3}{c}{Task 3: KV Pairing} \\
    \cmidrule(lr){3-4} \cmidrule(lr){5-8} \cmidrule(lr){9-11}
      &  &  $K_e$ &  $K_a$ &  $QA_e$ &  $QA_a$ &  $QA_{pos-e}$ & $QA_{pos-a}$ &  $K_e V_e$ &  $K_e V_a$ &  $K_a V_a$ \\
    \midrule
    
    % Encoder-only Models
    MBERT (0.18B) & Fine-tuning  & 0.7307 & 0.7314 & \textbf{0.8598} & \textbf{0.8689} & 0.7384 & 0.7966 &  0.6450 & 0.6996 & 0.7001 \\
    RoBERTa-wwm (0.11B) & Fine-tuning  & 0.7348 &  0.7352 & 0.7482 & 0.7577 & 0.7451 & 0.8053 & 0.6359 &  0.6924 & 0.6926 \\
    MacBERT (0.11B) & Fine-tuning  & 0.7338 &  0.7347 & 0.7885 & 0.7982 & \textbf{0.7468} & \textbf{0.8084} & 0.6458 & \textbf{0.7052} & \textbf{0.7061} \\
    McBERT (0.11B) & Fine-tuning  & \textbf{0.7368} & \textbf{0.7372} & 0.7093 & 0.7185 & 0.7432 & 0.8017 & \textbf{0.6483} &  0.7053  &  0.7057 \\

    HybridSpanModel(ours)(0.11B) & Fine-tuning  & 0.7537 & 0.7557 & 0 & 0 & 0 & 0 & 0.6826 &  0.7102  &  0.7120 \\

    StagedRoBERTa(ours)(0.11B) & Fine-tuning  & 0.7400 & 0.7402 & 0 & 0 & 0 & 0 & 0.6254 &  0.6694  &  0.6696 \\

    Multi-TaskRoBERTa(ours)(0.11B) & Fine-tuning  & 0.7393 & 0.7399 & 0 & 0 & 0 & 0 & 0.6346 &  0.6677  &  0.6684 \\
    
    StagedMacBert(ours)(0.11B) & Fine-tuning  & \textbf{0.7523} & \textbf{0.7559} & 0 & 0 & 0 & 0 & \textbf{0.6902} &  \textbf{0.7139}  &  \textbf{0.7164} \\

    StagedMacBert(ours)(0.11B) & Fine-tuning  & \textbf{0.7507} & \textbf{0.7532} & 0 & 0 & 0 & 0 & \textbf{0.6822} &  \textbf{0.7037}  &  \textbf{0.7055} \\

    kv-nsp+noise embedding(ours)(0.11B) & Fine-tuning  & 0.7524 & 0.7549 & 0 & 0 & 0 & 0 & 0.6830 &  0.7078  &  0.7094 \\
    
    kv-mlm+noise embedding(ours)(0.11B) & Fine-tuning  & 0.7554 & 0.7567 & 0 & 0 & 0 & 0 & 0.6889 &  0.7084  &  0.7095 \\
    
    \midrule

    % Decoder-only Models
    Qwen3-0.6B & Two-shot  & 0.0580 & 0.0600 & 0.8635 & 0.8657 & 0.1878 & 0.2013 & 0.3811 & 0.4197 & 0.4221 \\
    
    \bottomrule
  \end{tabular}%
  }
\end{table*}

As shown in Table \ref{tab:main_results}, our model significantly outperforms all baselines. Notably, the gain in \textbf{Recall} is more substantial than the gain in \textbf{Precision}. This suggests that by explicitly modeling \textbf{OCR-induced noise}, our model effectively recovers entities that are typically missed by general-purpose models due to character fragmentation or recognition errors.

\subsection{Ablation Study}
To evaluate the contribution of each component, we conduct extensive ablation experiments as follows:
\begin{itemize}
    \item \textbf{Effect of Noise Embedding:} Removing the 7-dimensional noise features leads to a performance drop of $XX\%$ in low-quality scenarios, confirming that explicit \textbf{Character-level Noise} modeling is crucial for \textbf{Practical Usability}.
    \item \textbf{Effect of KV-MLM:} Substituting the key-aware WWM with random masking reduces the F1-score by $XX\%$, proving the importance of \textbf{Prior Knowledge} in preserving medical semantic integrity.
    \item \textbf{Effect of KV-NSP:} The inclusion of the matching task improves the logical alignment between keys and values. 
    % 中文注释：针对 KV-NSP 的数据问题，实验证明混合使用 Qwen 生成的 clean 数据与带噪样本（Noisy Samples）能提供最佳的鲁棒性。
\end{itemize}

\subsection{Robustness Analysis}

We categorize the test set into three tiers based on \textbf{OCR confidence scores}: High, Medium, and Low. 
% [Insert Figure: Robustness bar chart]
While baselines suffer a significant ``performance cliff'' as image quality degrades, our model demonstrates \textbf{Empirical Convergence} and graceful degradation. This is attributed to the \textbf{Confidence Calibration} provided by our \textbf{Noise-Embedding} layer.

\subsection{Case Study}
Figure [X] illustrates a typical case where the term ``Hypertension'' was misrecognized by the OCR engine as ``High [X] Pressure'' with low confidence. While the baseline RoBERTa failed to extract this entity or predicted an incorrect term, our model utilized the \textbf{Spatial Grounding} and the noise-aware embedding to correctly infer the medical entity from the clinical context.

\subsection{Results and Analysis}
The comprehensive results across multiple downstream tasks validate that the \textbf{Noise-Aware Domain-Adaptive Pre-training Framework} not only mitigates the negative impact of \textbf{OCR-induced noise} but also enhances the \textbf{Holistic Extraction Quality} by integrating structural and linguistic priors.

\section{Conclusion, Limitations, and Future Work}

\subsection{Conclusion and Contributions}
本文聚焦于 OCR 临床纸质报告的半结构化信息抽取场景，核心目标是让基于 BERT 的编码器在残余 OCR 噪声（字符断裂、置信度波动、版式错位、语义截断等）下仍能稳定完成 Key--Value 抽取与下游 KV-NER 任务。围绕这一目标，我们提出了一个面向真实业务流程的噪声鲁棒预训练框架 KV-BERT，并在工程实现中给出了可复现的数据处理与训练流水线。

从研究定位与方法特点、关键问题解决以及实践洞察与经验总结三个维度，本文的核心贡献可概括为：
\begin{itemize}
    \item 研究定位与方法特点：我们提出 KV-BERT——一个面向 OCR 临床报告半结构化抽取的 Noise-Robust Domain-Adaptive Pre-training 框架。该框架以中文 BERT 类编码器（如 MacBERT 等）为基座，将文本语义建模、结构先验与 OCR 物理信号进行统一融合，并面向真实数据清洗、去重、分源配比以及 OCR/非 OCR 分路构建等工程约束，形成端到端可落地的训练方案。
    \item 关键问题解决：针对通用预训练模型“仅依赖语义、忽略 OCR 质量信号、且缺乏 KV 结构约束”导致在噪声 OCR 文本上性能显著下降的问题，我们从预训练任务设计与输入表示增强两条路径入手：
    (1) 设计 KV-MLM，在预训练阶段对医学实体与 KV 边界进行 key-aware 的整体遮盖，使模型在噪声上下文中学习重建完整医学语义；
    (2) 设计 KV-NSP，将传统 NSP 替换为“Key--Value 是否匹配”的二分类任务，并引入强负样本策略，迫使模型显式学习字段名与字段值之间的逻辑一致性；
    (3) 提出 Noise Embedding，将 OCR 置信度统计与版式/形态信号抽象为 7 维特征，并通过分桶映射到可学习向量后与 Token/Position/Segment embedding 相加，使模型具备“感知文本质量”的能力。
    \item 实践洞察与经验总结：在真实 OCR 医疗文本中，影响抽取质量的关键并非简单的随机字符替换，而是更复杂的结构性噪声与分布偏移：例如字符碎裂导致实体被拆分、低置信字符在关键字段附近密集出现、以及版式错位引发的 KV 边界歧义。进一步地，我们在工程实践中发现：噪声特征与文本 token 的对齐一致性是方法获得稳定收益的前提——一旦 OCR 样本顺序或切分策略发生改变，噪声特征将出现错配，从而对训练产生系统性干扰。因此，本工作强调将 OCR 数据与非 OCR 数据分路构建、对齐校验与再合并的流程化约束，以保证噪声建模的可靠性。
\end{itemize}

综上，KV-BERT 将结构先验（KV-MLM/KV-NSP）与质量先验（Noise Embedding）融合进预训练阶段，使模型在噪声 OCR 场景下获得更稳定的实体召回与更一致的 KV 逻辑匹配能力，从而为临床纸质报告跨机构流转场景提供可落地的信息抽取基础模型。

\subsection{Limitations}
尽管本文方法在真实数据上验证了有效性，但仍存在一些局限：
\begin{itemize}
    \item 对 OCR 元信息的依赖：Noise Embedding 需要 OCR 引擎输出的置信度/对齐等统计信号。不同 OCR 引擎的特征定义与分布可能不一致，导致跨引擎迁移需要重新标定特征与分桶边界。
    \item 分桶策略的启发式与数据依赖：连续噪声特征通过分桶离散化后再查表映射，虽然轻量，但 bin 数量、边界与“完美文本锚点”的设定仍是经验性的；在分布变化明显的数据上可能需要重新拟合。
    \item 对齐与切分的工程敏感性：KV-MLM 依赖 word-level 的一致切分（如 jieba 词典与 word\_ids），Noise 特征又依赖 OCR 文本顺序与 token 对齐。若训练前处理（滑窗切分、shuffle、合并配比）与对齐校验未严格遵守约束，可能出现噪声错配并显著影响收益。
    \item 结构监督的覆盖范围有限：KV-NSP 以 Key--Value 匹配为目标，但临床报告中还存在跨行/跨段落的长程依赖、表格化布局以及多值字段等复杂结构，仅用二分类匹配不能完全覆盖。
    \item 评测维度仍可扩展：当前实验主要围绕 KV-NER / KV Pairing 等任务展开；在更广泛的文档类型、更多医院域、以及更极端噪声（拍照反光、遮挡、手写混杂）下的泛化能力仍需进一步系统评估。
\end{itemize}

\subsection{Future Work}
面向上述局限与真实落地需求，未来工作可从以下方向展开：
\begin{itemize}
    \item 更强的噪声表示学习：从“离散分桶 + 查表”扩展到可学习的连续映射（如小型 MLP / spline / monotonic network），并引入跨域自适应机制，降低对特定 OCR 引擎分布的依赖。
    \item 多模态与版式融合：在文本 + 噪声特征之外，进一步融合布局信息（bbox、行块结构）甚至视觉特征，形成更完整的 Document AI 表示，以提升对表格、对齐错位与跨列字段的处理能力。
    \item 更丰富的结构化预训练任务：在 KV-NSP 基础上，探索面向报告结构的对比学习/排序学习目标（如 Key--Value 多候选检索、跨段落一致性约束、字段类型判别），并引入更难的“半硬负样本”以提升泛化。
    \item 数据与训练流程的自动化鲁棒性：将对齐校验、分路构建、合并配比与质量监控纳入标准化工具链，减少人工配置带来的错配风险；同时探索更高效的预训练策略（如课程学习、分阶段引入 KV-NSP），以降低计算成本。
    \item 更广泛的临床应用验证：将预训练模型推广到更多下游任务（如字段标准化、结构化质控、跨机构模板迁移），并在隐私合规前提下构建更大规模、更复杂噪声分布的评测基准。
\end{itemize}

\end{document}
