\documentclass[runningheads]{llncs}

%
\usepackage[T1]{fontenc}
% T1 fonts will be used to generate the final print and online PDFs,
% so please use T1 fonts in your manuscript whenever possible.
% Other font encondings may result in incorrect characters.
%
% --- 在这里添加你的数学宏包 ---
\usepackage{amsmath}  % 提供 \text{} 和更强的公式环境
\usepackage{amssymb}  % 提供 \mathbb{R} 等数学符号

\usepackage{multirow}
\usepackage{booktabs}
\usepackage[table]{xcolor} % 注意必须带 table 参数
\usepackage{amsmath}

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
% ===== TODO toggle =====
\newif\iftodo
\todotrue   % 开：显示 TODO
% \todofalse % 关：隐藏 TODO
\newcommand{\TODO}[1]{\iftodo{\color{red}\textbf{[TODO: #1]}}\fi}
% ===== Anonymous submission toggle =====
\newif\ifanonymous
\anonymoustrue      % 匿名投稿阶段
% \anonymousfalse   % camera-ready 阶段
\ifanonymous
  \author{Anonymous Author(s)}
  \institute{Anonymous Institution}
\else
  \author{Hao Li \and Yingyun Li \and Ying Qin \and Haiyang Qian}
  \institute{AI Starfish\\
  \email{\{hao.l, yingyun.li, ying.qin, haiyang.qian\}@aistarfish.com}}
\fi

\begin{document}


\title{ KV-BERT: A Noise-Robust Pretraining Framework for Semi-Structured Key-Value Extraction in OCR Clinical Reports}
\author{Anonymous Authors}
%
%\authorrunning{ et al.}
% First names are abbreviated in the running head.
% If there are more than two authors, 'et al.' is used.
%
%\institute{AI Starfish\\
%\email{\{hao.l, yingyun.li, ying.qin, haiyang.qian\}@aistarfish.com}
 %}

\maketitle              % typeset the header of the contribution


\begin{abstract}
Because clinical information often remains inaccessible across healthcare institutions, retrieving previous clinical records of patients from other institutions mostly relies on paper clinical reports provided by patients. Thus, IE (information extraction) is crucial in this context. Semi-structured key-value extraction—identifying header fields (keys) and their associated content fields (values) is a foundational task in clinical information extraction.  This process is typically performed by first running optical character recognition (OCR) on paper reports, followed by applying learning-based models to extract key-value pairs. While BERT-based models are suited for IE due to their bidirectional attention mechanism, existing models (e.g., BERT, RoBERTa) are primarily trained on clean data w/o noise, and perform poorly on noisy OCR-derived data. Thus, we propose a noise-resistant pre-training framework for semi-structured key-value (KV) extraction in OCR medical reports. We introduce KV-MLM (Masked Language Modeling) and KV-NSP (Next Sentence Prediction) as pre-training tasks, and model OCR artifacts and meta information as learnable embedding vectors. The KV-MLM task masks medical entities and KV pair boundaries, enhancing the model's contextual inference and memory capabilities for medical terminology. The KV-NSP task is an improved NSP task that constructs with strong negative samples, forcing the model to learn logical consistency between keys and values. We added an embedding dimension, extracting seven physical features including OCR confidence and alignment scores. These were mapped to learnable embedding vectors via a binning strategy. Experimental results on real world clinical report datasets  demonstrate that the models pretrained outperforms both general BERT and RoBERTa base models, as well as SOTA medical Large Language Models (LLMs) (but 1-3 orders of magnitude fewer than these LLM), on semi-structured key-value pair extraction.
\keywords{OCR\and Noise \and BERT \and Pretraining \and NER \and Clinical Report.}
\end{abstract}


\section{Introduction}
In today's healthcare landscape, electronic medical records (EMRs) are ubiquitous. However, they remain isolated across hospitals. Patients can access their information only through paper clinical reports. This necessitates converting these clinical reports into text using optical character recognition (OCR) technology during hospital transfers or when establishing patient records at third-party institutions. The OCR results must then be processed into a semi-structured format.

State-of-the-art optical character recognition (OCR) and mainstream commercial OCR frameworks (such as PP-OCR~\cite{PPOCR2022}) typically integrate mature preprocessing modules. These commonly include image denoising, moiré pattern removal, document skew correction, and curvature correction, significantly enhancing the quality of processed raw images. However, in practical scenarios, recognition biases, semantic truncation, and logical association errors remain unavoidable. This “residual noise” driven by complex scenarios imposes significant robustness challenges on subsequent semi-structured tasks.

Concurrently, BERT-based models are widely used for IE because of their bidirectional attention mechanism, as introduced in ~\cite{bert}, these models are typically pre-trained by two pretraining tasks--- Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Existing language BERT-based models are typically pre-trained on corpora without OCR-induced noise (e.g., Wikipedia, relevant news, medical guidelines). When directly applied to OCR-derived clinical report with OCR-induced noise, their performance degrades. Moreover, these general pre-trained models understand text solely based on semantics, without leveraging additional information provided by OCR engines (e.g., confidence scores).

Thus, we propose KV-BERT, a noise-resistant pre-training framework for semi-structured key-value (KV) extraction in OCR medical reports. We introduce KV-MLM and KV-NSP as pre-training tasks, and model OCR artifacts and meta information as learnable embedding vectors. Specifically, we made the following innovations:
\begin{itemize}
\item \textbf{KV-MLM:}
   To better support semi-structured tasks, we refined the random masking strategy by implementing masking tailored to medical entities and key-value boundaries.
   % We refine the random masking strategy in the original Masked Language Modeling (MLM) task of the BERT model. Specifically, we implement masking tailored to medical entities and key-value boundaries. 
   This novel masking task compels the model to learn inferring complete medical terminology from context during pre-training, thereby enhancing its ability on downstream tasks.
\item \textbf{KV-NSP:}
   For the Next Sentence Prediction (NSP) task—one of BERT's original tasks—we replaced the goal of predicting sentence continuity with determining whether given Key and Value pairs are logically matched. By learning from positive samples and hard negative samples, we force the model to learn the logical consistency between field names and their corresponding content.
\item \textbf{Noise-Embedding:}
  Current OCR engines not only output the recognized text but also provide additional information regarding recognition quality. To enable the model to assess the quality of each token, We select seven pieces of this information and then map a 7-dimensional noise feature vector (including OCR confidence mean/variance, character break rate, layout alignment score, etc.) into a learnable embedding via binning strategy. This is then combined with text embedding, position embedding, and other embeddings as model's input.
  % (presumably, this component is expected to function as described—though I haven't examined the specific attention results—such that when text is unclear, the model learns to rely on context or reduce the weight of that word).
\end{itemize}


we make the following contributions:
We propose a novel noise fusion mechanism, we integrate the confidence statistics from the OCR engine with visual layout features (totaling 7-dimensional features) into the BERT input layer, significantly enhancing the model's performance across various downstream tasks on low-quality OCR text.
We designed two pre-training tasks tailored for downstream semi-structured tasks: proposing KV-MLM and KV-NSP strategies to incorporate structural prior knowledge from Clinical Reports into the pre-training phase, addressing the limitations of general models in understanding semi-structured data.
We demonstrated effectiveness through extensive experiments on real medical OCR datasets. Results demonstrate that our approach outperforms both general BERT and Roberta bases, as well as state-of-the-art medical Large Language Models (LLMs) (though by 1-3 orders of less than these LLMs) across multiple downstream tasks. Notably, our model exhibits particularly significant performance gains on noisy data.

\section{Related Work}
\subsection{Domain-Adaptive Pre-training}
In recent years, pre-trained language models have achieved remarkable success across various NLP tasks. Concurrently, BERT-based models are well-suited for information extraction (IE) due to their bidirectional attention mechanism. To enhance model performance in specific vertical domains, domain-adaptive pre-training has emerged as a mainstream trend. In the medical domain, BioBERT ~ \cite{BioBERT} significantly improved biomedical entity recognition accuracy by fine-tuning on large-scale corpora like PubMed; ClinicalBERT ~ \cite{ClinicBERT} further enhanced clinical text understanding by leveraging electronic medical records (EMR). However, these models primarily utilize high-quality, clean text during pre-training, leading to substantial performance degradation when confronted with real-world Clinical Reports containing OCR noise.

\subsection{Document Multimodal Models and Key-Value Pair Extraction}
When processing semi-structured documents like tables and reports, models such as LayoutLM ~ \cite{Layoutv1} pioneered a new paradigm in Document AI by integrating textual, layout, and visual features. Subsequent evolutions like LayoutLMv3 ~ \cite{Layoutv3} further optimized alignment efficiency. While these models excel in general cross-modal representation learning, they often overlook the strong structural prior knowledge inherent in domain-specific documents like Clinical Reports (e.g., the logical correspondence between Key and Value). The two pre-training tasks proposed in this paper—KV-MLM and KV-NSP—aim to address this shortcoming.

\subsection{OCR-Induced Noise}
Handling OCR-induced noise has long been a challenge in document analysis. Traditional approaches primarily focus on post-OCR correction or spell checking.
% Recent studies, such as NAT (Noisy Augmented Training) ~\cite{NAT}, attempt to enhance model robustness by simulating noise distributions through data augmentation. 
Unlike such “repair-oriented” approaches, this paper introduces a Noise-Embedding mechanism: by explicitly modeling physical features (including the Confidence Score) from OCR engine outputs across seven dimensions, it endows the model with the ability to perceive and adaptively handle text inputs of varying quality. This achieves a paradigm shift from ‘denoising’ to “proactive noise tolerance.”


\section{Methodology}

\subsection{Overall Architecture}
\begin{itemize}
\item \textbf{Base Model:} This study employs MacBERT (MLM as Correction BERT) ～\cite{MacBert} as its backbone network. The model utilizes a standard base architecture comprising a $L=12$ layer Transformer encoder with a hidden layer dimension of $H=768$ and configured with $A=12$ multi-head attention heads. Compared to traditional BERT models, MacBERT further optimizes the pre-training strategy, enabling it to perform better on complex semi-structured tasks in Chinese.

\item \textbf{Input Representation:} Traditional encoder-only models (represented by BERT) incorporate Token embeddings, Position embeddings, and Segment Embeddings. To enable the model to perceive OCR derived text quality and adapt to noise introduced by OCR, we introduce a fourth Embedding layer—Noise Embedding. The final input representation $E_{\text{input}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ is computed as follows:
\begin{equation}
    E_{\text{input}}(x_i) = E_{\text{tok}}(x_i) + E_{\text{pos}}(p_i) + E_{\text{seg}}(s_i) + F_{\text{noise}}(v_i)
\end{equation}
where $E_{\text{tok}}$, $E_{\text{pos}}$, and $E_{\text{seg}}$ denote the token, position, and segment embeddings, respectively. The $v_i \in \mathbb{R}^7$ represents a 7-dimensional continuous noise feature vector, which is mapped into the d-dimensional embedding space via a linear projection layer $F_{\text{noise}}$.
\end{itemize}

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
Beyond plain text, raw OCR outputs provide rich metadata that reflects the reliability of the recognition process. We model these information provided by the OCR engine as dimensions of the embedding.
    
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/noise_embedding_mechanism.pdf} % 请替换为你导出的文件名
    \caption{The workflow of the proposed noise-embedding Mechanism. For each token, seven statistical features are extracted from the OCR output, discretized into bin IDs via non-linear mapping functions $\Phi_k$, and transformed into dense vectors through separate embedding matrices $W_k$. These vectors are then summed to produce the final noise embedding $F_{\text{noise}}(v_i)$, which is added to the standard token representations.}
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
To input continuous physical signals into the model, we constructed a binning mapping function. Compared to traditional linear continuous mapping approaches, this discretization strategy offers significant advantages. First, it effectively captures nonlinear features in noisy signals by learning independent semantic vectors for different numerical intervals. For instance, when OCR confidence scores exhibit extreme skewness (clustering near 1.0), encrypted binning enhances the model's resolution for detecting minute quality variations. Second, combined with clipping processing, this approach enhances the model's robustness against OCR outliers. Finally, by setting fixed anchor bins, the strategy successfully achieves compatibility between noisy OCR samples and perfectly clean data within the same representation space, enabling the model to dynamically adjust its trust weights for contextual information across varying data quality levels.

\begin{itemize}
    \item \textbf{Non-linear Binning:}  For distributions of different features (e.g., confidence scores biased toward 1.0, alignment scores biased toward 0), we precomputed statistical quantiles as bin boundaries.
    \item \textbf{Anchor Bin:} All “perfect text” (no OCR-induced noise) and perfect OCR recognition results are mapped to Anchor Bin.
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

We propose a key-value level Whole Word Masking (WWM) strategy guided by a prior dictionary to incorporate domain-specific knowledge into the pre-training phase.


\subsubsection{Key-Value Dictionary}
During preprocessing, we loaded two specialized dictionaries using the Jieba tokenizer:
\begin{itemize}
  \item Key Set: A collection of key terms from clinical cases (e.g., “chief complaint,” “presenting history,” “creatinine”).
  \item WordPiece: We use WordPiece to derive a foundational vocabulary from OCR-derived clinical report. We find that directly using WordPiece results as part of the expanded vocabulary yielded poor performance, causing the model's performance to decline in downstream tasks. As noted by Balde et al. (2024) in their MEDVOC study ～\cite{medvoc}, blindly incorporating the large number of candidate subwords generated by WordPiece into the vocabulary leads to reduced model performance on downstream tasks in specific domains such as medicine, due to token sparsity and fragmented subwords. So，we filter the result by a large language model Qwen3～ \cite{Qwen3} to retain only medically relevant entities.
  
\end{itemize}
Through Jieba segmentation, during data preprocessing, we assign the same \texttt{word\_id} to all characters within the same segment. This ensures medical entities, key names, and values are not split into individual Chinese characters.

\subsubsection{KV-Aware Masking Strategy}
Given an input sequence $X$, we perform WWM based on the predefined \texttt{word\_ids}. The masking probability follows a Bernoulli distribution and standard masking probability:
\begin{equation}
    M \sim \text{Bernoulli}(0.15)
\end{equation}

If a token belongs to a medical entity (e.g., “adenocarcinoma”, or “腺癌” in our Chinese corpus), when selected, the entire associated word (all tokens sharing the same \texttt{word\_id}) is replaced with [MASK]. This masking approach forces the model to reconstruct complete medical semantics through context. Consequently, it enhances the model's ability to memorize, comprehend, and reason about medical entity vocabulary and common key-value pairs, thereby improving performance on downstream tasks (Information Extraction and Semi-Structured Tasks).

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
    \item \textbf{Reverse Order (Hard Negative, 50\%):} To forces the model to learn the directionality of medical records (where “key” points to “value,” not vice versa), we swap the order of Key and Value (i.e., Key-Value becomes Value-Key).
    Purpose: Forces the model to learn the directionality of medical records (where “key” points to “value,” not vice versa).
    \item \textbf{Random Replacement (Easy Negative, 50\%):} Keeps the Key unchanged and randomly samples an unrelated value from the corpus.
    Purpose: Learns semantic matching (e.g., distinguishing format differences between “Name” and “Date”)
\end{itemize}

\subsection{Data Composition and Pre-processing}
To ensure the model acquires both specialized medical knowledge and general language proficiency, we adopted a multi-source data integration strategy.
The data ratio has undergone multiple adjustments by our team. Through experimentation, we have learned that the proportion of clinic reports should not be excessively high. We maintain the proportion of OCR-drived clinical reports at approximately 35\%. The remaining corpus consists of specialized medical corpora and general language corpora.
In practice, the length of individual samples often exceeds the model's max length. We employ a sliding window segmentation strategy. This approach (whose parameters are determined by window size and stride) preserves the continuity of medical information, ensuring that the corpus is fully utilized within the model's max length constraint.


\section{Experiments}

\subsection{Experimental Setup}
\subsubsection{Pre-training Data}

 The pre-training data comprises medical records processed by OCR, medical textbooks/guidelines, medical papers, Wikipedia entries, and a small amount of general Chinese text. After cleaning and deduplication, a total of 982,183 text samples were obtained (counted by lines, with one sample per line). To control domain proportions, we performed static resampling across sources, yielding 225,481 training texts. Among these, OCR-drived clinical reports accounted for 35.00\%, medical corpora for 46.96\%, and general corpora for 18.02\%. 
 After resampling, we obtained 225,481 pre-training texts (counted by lines, with one text per line). To mitigate information loss from truncation due to the maximum length of 512, we performed character-level sliding window segmentation ($\mathrm{window}=1000$, $\mathrm{stride}=500$) on long texts exceeding the threshold, splitting each long text into multiple chunks. Following sliding window segmentation, the total number of training segments increased from 225,481 to 537,721. Subsequent pre-training encoded these segmented chunks as the basic pre-training text units, truncating each to 512 tokens.

\subsubsection{Fine-tuning（KV-NER）方法与数据}
为验证预训练模型在真实业务场景中的有效性，我们在下游半结构化抽取任务上采用统一的微调流程。我们将 OCR 页面文本视作序列标注问题，预测“键名（KEY）/键值（VALUE）”两类实体的边界，再通过轻量级规则将其组装为键值对集合，用于任务1的键名发现与任务2的键值对抽取评估。

数据来源与格式：下游数据来自真实医疗检验/检查报告的 OCR 结果，并在标注平台以字符级位置标注实体边界。每个样本以“单页/单条 OCR 文本”为单位，包含 OCR 文本、实体标注（起止位置与类别），以及与 OCR 质量相关的噪声特征（对应本文提出的噪声嵌入）。我们将按词或按行统计的噪声向量展开到字符级序列，使其与 OCR 文本严格对齐；若样本不含噪声元信息（或为干净文本），则使用“完美文本”的锚点向量作为缺省噪声输入。

数据规模与统计：在 Real 数据集上，训练集包含 3,224 页样本，测试集包含 358 页样本。训练集与测试集的文本长度（按字符数统计）分别为：均值 735.49 和 743.25，中位数 714 和 705；第 90 百分位为 1149 和 1122，第 99 百分位为 1673 和 1571。实体标注总量（按标注实例数统计）在训练集与测试集上分别为：KEY 45,684 和 5,102，VALUE 43,432 和 4,836；此外数据中还包含辅助标签 HOSPITAL 2,242 和 239。

标签体系与编码：训练时使用 BIO 序列标注体系，并通过标签映射将不同标注别名（中英文、大小写）归一到 KEY 和 VALUE。由于中文采用子词切分，我们启用“标签传播到所有子词”的策略，将同一字符跨度对应的标签同步到被切分后的所有子词 token，以减少边界信息在子词级的丢失。

输入构造（含长文本切片与噪声对齐）：模型最大输入长度设为 512。对于超过最大长度的 OCR 页面文本，我们采用滑窗切片方式构造重叠片段以覆盖跨窗口实体：片段长度为 500，重叠为 50。噪声特征为 7 维向量，先通过与预训练一致的分桶边界离散化为 bin id，再通过查表得到噪声嵌入并与 token 表示相加，从而保证下游阶段的噪声建模与预训练阶段一致。

微调模型与优化设置：我们在预训练骨干（MacBERT/对照模型）之上添加序列标注头，可选 BiLSTM（3 层，hidden size 为 384）用于增强局部序列建模，再接线性分类层与 CRF 解码层以建模 BIO 转移约束并输出最优标签序列。优化器为 AdamW，学习率 2e-5，warmup ratio 为 0.1，weight decay 为 0.03；训练 4 个 epoch，batch size 为 8，dropout 为 0.2。为保证对比公平，所有模型在完全一致的数据划分与超参下微调，仅替换初始化的预训练 checkpoint。

推理、键值对组装与评测：推理阶段将 CRF 解码得到的实体跨度还原为 KEY/VALUE 片段，并采用“最近邻顺序配对”的轻量规则将键与其后最近的值组成键值对集合。评测时，我们输出标准化的预测结果与由测试集转换得到的标准答案，并在任务1与任务2指标计算前进行一次 span 对齐处理，以消除分词或切片带来的边界抖动对严格匹配评测的影响。

\subsection{Downstream Task Definitions}
% The semi-structured extraction (Task 3 corresponds to Key-Value Pairing) is decomposed into three sub-tasks to accommodate different real-world scenarios.
To accommodate diverse practical scenarios, we decompose semi-structured information extraction into three tasks, where Task~3 is the end-to-end setting.

Let $d$ denote an input OCR-derived clinical report page, $K$ the ground truth key set, and $S$ the ground-truth set of key--value pairs. For Task~2, $k \in K$ denotes a queried key.

In \textbf{Task~1 (Open-World Key Discovery)}, the input is $d$ and the output is a predicted key set $\hat{K}$. The model identifies $\hat{K}$ from the OCR-derived text, without a predefined key set. 
% \textbf{Task~2 (Key-Conditioned Question Answering)} focuses on value extraction: the input is $(d, k)$ where $k$ is a queried key. The output is the corresponding value prediction $\hat{v}$. Tasks~1 and~2 together form a two-stage pipeline for semi-structured information extraction.
Finally, \textbf{Task~2 (End-to-End Key--Value Pairing)} takes $d$ as input and outputs a predicted set of key--value pairs $\hat{S}=\{(\hat{k},\hat{v})\}$. By jointly identifying keys and their corresponding values, the objective of the model is to learn how to extract $S$ from $d$.

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
为便于阅读表 1 中的指标符号，本文用下标 e 和 a 区分两种匹配判定方式：e 表示精确匹配（Exact Match），a 表示近似匹配（Approx Match）。精确匹配指在统一的文本归一化后（如去首尾空白、统一大小写）预测文本与真值文本完全一致；近似匹配则允许一定的字符扰动，采用归一化编辑距离相似度（NED，相似度越高越接近）并结合长度自适应阈值进行判定（短文本阈值更宽松、长文本更严格）。

在任务 1（键名发现）中，K_e 表示按精确匹配统计得到的键名 F1，K_a 表示按近似匹配统计得到的键名 F1。
在任务 2（键值配对）中，我们同时考察键名与键值两部分的匹配程度：K_eV_e 表示键名精确匹配且键值精确匹配；K_eV_a 表示键名精确匹配且键值近似匹配；K_aV_a 表示键名近似匹配且键值近似匹配。上述指标均基于精确/近似匹配下的 TP 数量计算精确率、召回率与 F1。
% Table 1: Performance on KV-NER

% \begin{table*}[t]
%   \centering
%   \small
%   \setlength{\tabcolsep}{3.5pt}
%   \renewcommand{\arraystretch}{1.15}
%   \definecolor{lightgray}{gray}{0.92}
%   \caption{Model evaluation results across tasks (Real).}
%   \label{tab:main_results}
%   \resizebox{\textwidth}{!}{%
%   \begin{tabular}{llccccccccc}
%     \toprule
%     \multirow{2}{*}{Model} & \multirow{2}{*}{Setup} & \multicolumn{2}{c}{Task 1: Key Discovery} & \multicolumn{4}{c}{Task 2: QA} & \multicolumn{3}{c}{Task 3: KV Pairing} \\
%     \cmidrule(lr){3-4} \cmidrule(lr){5-8} \cmidrule(lr){9-11}
%       &  &  $K_e$ &  $K_a$ &  $QA_e$ &  $QA_a$ &  $QA_{pos-e}$ & $QA_{pos-a}$ &  $K_e V_e$ &  $K_e V_a$ &  $K_a V_a$ \\
%     \midrule

%     % Decoder-only Models
%     Qwen3-0.6B & Two-shot  & 0.0580 & 0.0600 & 0.8635 & 0.8657 & 0.1878 & 0.2013 & 0.3811 & 0.4197 & 0.4221 \\
%     % Encoder-only Models
%     MBERT (0.18B) & Fine-tuning  & 0.7307 & 0.7314 & \textbf{0.8598} & \textbf{0.8689} & 0.7384 & 0.7966 &  0.6450 & 0.6996 & 0.7001 \\
%     RoBERTa-wwm (0.11B) & Fine-tuning  & 0.7348 &  0.7352 & 0.7482 & 0.7577 & 0.7451 & 0.8053 & 0.6359 &  0.6924 & 0.6926 \\
%     MacBERT (0.11B) & Fine-tuning  & 0.7338 &  0.7347 & 0.7885 & 0.7982 & \textbf{0.7468} & \textbf{0.8084} & 0.6458 & \textbf{0.7052} & \textbf{0.7061} \\
%     McBERT (0.11B) & Fine-tuning  & \textbf{0.7368} & \textbf{0.7372} & 0.7093 & 0.7185 & 0.7432 & 0.8017 & \textbf{0.6483} &  0.7053  &  0.7057 \\

%     % HybridSpanModel(ours)(0.11B) & Fine-tuning  & 0.7537 & 0.7557 & 0 & 0 & 0 & 0 & 0.6826 &  0.7102  &  0.7120 \\

%     % StagedRoBERTa(ours)(0.11B) & Fine-tuning  & 0.7400 & 0.7402 & 0 & 0 & 0 & 0 & 0.6254 &  0.6694  &  0.6696 \\

%     % Multi-TaskRoBERTa(ours)(0.11B) & Fine-tuning  & 0.7393 & 0.7399 & 0 & 0 & 0 & 0 & 0.6346 &  0.6677  &  0.6684 \\
    
%     % StagedMacBert(ours)(0.11B) & Fine-tuning  & \textbf{0.7523} & \textbf{0.7559} & 0 & 0 & 0 & 0 & \textbf{0.6902} &  \textbf{0.7139}  &  \textbf{0.7164} \\

%     %StagedMacBert(ours)(0.11B) & Fine-tuning  & \textbf{0.7507} & \textbf{0.7532} & 0 & 0 & 0 & 0 & \textbf{0.6822} &  \textbf{0.7037}  &  \textbf{0.7055} \\

%     KV-Bert(ours)(0.11B) & Fine-tuning  & \textbf{0.7565} & \textbf{0.7579} & 0.1665 & 0.1767 & 0.7226 & 0.7669 & \textbf{0.6900} &  \textbf{0.7125}  &  \textbf{0.7132} \\

%     kv-nsp+noise embedding(without kv-mlm)(0.11B) & Fine-tuning  & 0.7524 & 0.7549 & 0 & 0 & 0 & 0 & 0.6830 &  0.7078  &  0.7094 \\
    
%     kv-mlm+noise embedding(without kv-nsp)(0.11B) & Fine-tuning  & 0.7554 & 0.7567 & 0 & 0 & 0 & 0 & 0.6889 &  0.7084  &  0.7095 \\

%     kv-mlm+kv-nsp(without noise embedding)(0.11B) & Fine-tuning  & 0.7521 & 0.7540 & 0 & 0 & 0 & 0 & 0.6799 &  0.7020  &  0.7036 \\
%     \midrule

    
    
%     \bottomrule
%   \end{tabular}%
%   }
% \end{table*}

\begin{table*}[t]
  \centering
  \small
  \setlength{\tabcolsep}{6pt} % 适当增加列间距使表格更饱满
  \renewcommand{\arraystretch}{1.15}
  \definecolor{lightgray}{gray}{0.92}
  \caption{Performance evaluation on Key Discovery (Task 1) and KV Pairing (Task 2).}
  \label{tab:main_results_reduced}
  \resizebox{\textwidth}{!}{%
  \begin{tabular}{ll ccc cc}
    \toprule
    \multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Setup}} & \multicolumn{2}{c}{\textbf{Task 1: Key Discovery}} & \multicolumn{3}{c}{\textbf{Task 2: KV Pairing}} \\
    \cmidrule(lr){3-4} \cmidrule(lr){5-7}
      &  & $K_e$ & $K_a$ & $K_e V_e$ & $K_e V_a$ & $K_a V_a$ \\
    \midrule

    % Baselines & LLMs
    Qwen3-0.6B & Two-shot  & 0.0580 & 0.0600 & 0.3811 & 0.4197 & 0.4221 \\
    MBERT (0.18B) & Fine-tuning & 0.7307 & 0.7314 & 0.6450 & 0.6996 & 0.7001 \\
    RoBERTa-wwm (0.11B) & Fine-tuning & 0.7348 & 0.7352 & 0.6359 & 0.6924 & 0.6926 \\
    MacBERT (0.11B) & Fine-tuning & 0.7338 & 0.7347 & 0.6458 & 0.7052 & 0.7061 \\
    McBERT (0.11B) & Fine-tuning & 0.7368 & 0.7372 & 0.6483 & 0.7053 & 0.7057 \\
    \midrule

    % Our Main Model
    \textbf{KV-Bert (Ours)} & Fine-tuning & \textbf{0.7565} & \textbf{0.7579} & \textbf{0.6900} & \textbf{0.7125} & \textbf{0.7132} \\
    \midrule

    % Ablation Study (Shaded)
    \rowcolor{lightgray} \cellcolor{white} w/o KV-MLM & Fine-tuning & 0.7524 & 0.7549 & 0.6830 & 0.7078 & 0.7094 \\
    \rowcolor{lightgray} \cellcolor{white} \cellcolor{white} w/o KV-NSP & Fine-tuning & 0.7554 & 0.7567 & 0.6889 & 0.7084 & 0.7095 \\
    \rowcolor{lightgray} \cellcolor{white} \cellcolor{white} w/o Noise-Embedding & Fine-tuning & 0.7521 & 0.7540 & 0.6799 & 0.7020 & 0.7036 \\
    
    \bottomrule
  \end{tabular}%
  }
\end{table*}


As shown in Tab.~\ref{tab:main_results_reduced}, our model significantly outperforms all baselines. This suggests that by explicitly modeling \textbf{OCR-induced noise}, our model effectively recovers entities that are typically missed by general-purpose models due to character fragmentation or recognition errors.

\subsection{Ablation Study}
To evaluate the contribution of each component, we conduct extensive ablation experiments as follows:
\begin{itemize}
    \item \textbf{Effect of Noise Embedding:} Removing the 7-dimensional features and noise embeddding causes the model's F1-score to decrease by approximately 1.0\% in Task3, confirming that explicit OCR-induced noise modeling is crucial for the performence of the model.
    
    \item \textbf{Effect of KV-MLM:} Removing the KV-MLM task causes Task 3 ($K_e V_e$) to decrease by 0.7\% and causes Task 1 ($K_e$) to decrease by 0.41\%. This demonstrates that KV-MLM task is central to the model's understanding of the logical mapping between keys and values in medical reports.
    
    \item \textbf{Effect of KV-NSP:} Removing the KV-NSP task causes the model's F1-score to decrease by approximately 0.4\% in both $K_e V_a$ and $K_a V_a$.While its improvement in single-key recognition is less pronounced than MLM's, it enhances the model's ability to maintain key-value logical consistency under complex typography.

\end{itemize}


% \subsection{Robustness Analysis}

% We categorize the test set into three tiers based on \textbf{OCR confidence scores}: High, Medium, and Low. 
% % [Insert Figure: Robustness bar chart]
% While baselines suffer a significant ``performance cliff'' as image quality degrades, our model demonstrates \textbf{Empirical Convergence} and graceful degradation. This is attributed to the \textbf{Confidence Calibration} provided by our \textbf{Noise-Embedding} layer.

% \subsection{Case Study}
% Figure [5] illustrates a typical case where the term ``Hypertension'' was misrecognized by the OCR engine as ``High [X] Pressure'' with low confidence. While the baseline RoBERTa failed to extract this entity or predicted an incorrect term, our model utilized the \textbf{Spatial Grounding} and the noise-aware embedding to correctly infer the medical entity from the clinical context.

\section{Conclusion, Limitations, and Future Work}

\subsection{Conclusion and Contributions}
This paper focuses on the scenario of semi-structured information extraction from  Clinical Reports with OCR-induced noise. The core objective is to enable BERT-based models to reliably perform key-value extraction tasks despite residual OCR noise—including character fragmentation, confidence score fluctuations, layout misalignment, and semantic truncation. To achieve this goal, we propose KV-BERT, a noise-robust pre-training framework tailored for real-world workflows, and provide a fully reproducible end-to-end training pipeline in our engineering implementation.

From the perspectives of research positioning and methodological characteristics, resolution of key issues, and practical summaries, the core contributions of this paper can be summarized as follows:
\begin{itemize}
    \item Research Positioning and Methodological Features: We propose KV-BERT, a Noise-Robust Domain-Adaptive Pre-training framework for semi-structured extraction of OCR Clinical Reports. This framework leverages Chinese BERT models (e.g., MacBERT) as its foundation, integrating text semantic modeling, structural prior knowledge, and OCR physical signals into a unified system. It addresses practical engineering constraints—including real-world data cleaning, deduplication, source-based proportioning, and OCR/non-OCR data streaming—to deliver a reproducible, deployable training solution.
    \item Key Problem Solving: Addressing the significant performance degradation on noisy OCR text caused by existing general pre-training models (e.g., BERT, RoBERTa, MacBERT~\cite{Ma2023}) relying solely on semantic information, ignoring OCR quality signals, and lacking KV logical consistency in training, we designed and proposed the following innovations:
    (1) We introduced KV-MLM, which applies whole-word masking to medical entities and K-V boundaries during pretraining, enabling the model to learn reconstructing complete medical semantics in noisy contexts;
    (2) We introduced KV-NSP, replacing the traditional NSP task with a binary classification task of “determining whether Key-Value pairs match,” while introducing and constructing difficult negative samples. This task forces the model to explicitly learn logical consistency between field names and field values;
    (3) We propose Noise Embedding, abstracting OCR confidence statistics and layout signals into 7-dimensional features. These are bucketed and mapped to learnable vectors to endow the model with text quality perception capabilities.
    \item Practical Summary: In real-world scenarios, noise induced by OCR is often unavoidable. Furthermore, our engineering implementation reveals that whether noise embedding modeling yields stable gains depends on whether noise features are strictly aligned with training samples: If samples undergo shuffling, filtering, or sliding window segmentation after OCR text extraction, yet noise features are still read in the original OCR order or original line numbers, a mismatch occurs where text and noise features originate from different samples. This leads to systematic interference with the training objective. Based on this experience, our model adopts a strategy of constructing separate OCR and non-OCR data pipelines: The OCR pipeline preserves sample order and performs alignment verification before writing noise features; The non-OCR pipeline—clean text without noise (typically medical guidelines, medical data, Wikipedia, or general Chinese corpora)—is not bound to OCR metadata. Under the bucket allocation strategy, it is mapped to bucket whose ID=0. Finally, the two pipelines are merged, reducing the risk of noise feature mismatch through this workflow.
\end{itemize}

In summary, the proposed KV-BERT model integrates structural priors (KV-MLM/KV-NSP) with quality priors (Noise Embedding) into the pre-training phase. This enables the model to achieve stronger and more stable key-value recognition and extraction capabilities in scenarios with OCR-induced noise. Consequently, it provides a practical foundational model for information extraction in the cross-institutional circulation of clinical reports.

\subsection{Limitations}
Although the effectiveness of the method presented in this paper has been validated on real-world data, several limitations remain:
\begin{itemize}
    \item Dependence on OCR Metadata: Noise Embedding requires statistical signals such as confidence scores and alignment data output by OCR engines. Since different OCR engines may define and distribute features inconsistently, cross-engine migration necessitates recalibration of features and binning boundaries, followed by retraining according to the methodology.
    \item Data Dependencies of Binning Strategies: Continuous noise features are discretized through binning before lookup table mapping. While lightweight, the number of bins, their boundaries, and the definition of “perfect text anchors” are still designed based on the distribution and statistics of existing data. If the distribution of new data changes significantly, it may be necessary to re-statistic the distribution information and perform fitting.
    \item Engineering Sensitivity to Alignment and Segmentation: The proposed method demands high consistency in data preprocessing. KV-MLM relies on stable word segmentation results to generate word_ids (e.g., introducing custom dictionaries alters segmentation boundaries), while Noise Embedding requires accurately mapping statistical signals from OCR engines to the token sequences of corresponding training samples. If operations such as shuffling, filtering, deduping and reordering, or sliding window segmentation are performed on OCR text during dataset construction, but noise features are still read in the original order or with original indices, feature mismatch is likely to occur. Experiments demonstrate this reduces model performance. To mitigate this risk, practical deployments typically require fixing the sample order in the OCR pipeline, performing alignment verification before and after writing noise features, and maintaining consistent segmentation strategies when merging multiple data streams.
    \item Evaluation dimensions remain expandable: Current experiments primarily focus on tasks such as KV-NER; systematic assessment of generalization capabilities across broader document types, additional hospital domains, and more extreme noise conditions (e.g., glare, occlusion, mixed with handwriting) is still needed.
\end{itemize}

\subsection{Future Work}
In light of the aforementioned limitations and practical implementation requirements, future work can be pursued in the following directions:
\begin{itemize}
    \item Enhanced noise representation learning: Introduces cross-domain adaptive mechanisms to reduce dependency on specific OCR engine distributions.
    \item Multimodal and layout fusion: Beyond text + noise features, further integrates layout information (bbox, line block structure) and even visual features to train a multi-input-dimensional multimodal model, enhancing processing capabilities for tables, misaligned text, and cross-column fields.
    \item Automated robustness in data and training workflows: Incorporate alignment verification, lane construction, merge ratio adjustment, and quality monitoring into standardized toolchains to reduce misconfiguration risks from manual setup; concurrently explore more efficient pre-training strategies.
    \item Broader Downstream Task Validation: Extend pre-trained models to more downstream tasks (e.g., field standardization, structured quality control) and establish larger-scale evaluation benchmarks with more complex noise distributions while ensuring privacy compliance.
\end{itemize}



\bibliographystyle{splncs04}

\bibliography{reference}
\end{document}
