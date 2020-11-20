# 好未来开源教育领域首个在线教学中文预训练模型TAL-EduBERT

## 一、背景及下载地址

### 1. 背景

2020年初Covid-19疫情的爆发对各行各业产生了不可小觑的影响，也让以线下方式为主的传统教育在短期内受到了极大的冲击，更多人开始看到科技对教育市场的价值。在线教育成为了特殊时期教学的最佳选择，大规模地渗透至每一所学校、每一个家庭。在线教育的爆火使得教育行业产生了海量的在线教学语音识别（Automatic Speech Recognition，以下简称ASR）文本数据，极大地推动了教育领域技术的发展。

数据作为产业最为核心和宝贵的资源之一，更是自然语言处理技术（Natural Language Processing，以下简称NLP）在各个领域得以应用和发展的基础。在线教育文本数据有着区别于通用场景数据的特有属性，给在线教育领域NLP的研究、应用和发展带来了极大的挑战，一是从音视频转录出来的文本数据中，存在着较多的ASR错误，这些错误可能会对文本处理相关任务的效果造成较大的影响；二是数据中含有大量的教育领域特有的专有词汇，现有的通用领域的开源词向量和开源预训练语言模型（如Google BERT Base<sup>[1]</sup>，Roberta<sup>[2]</sup>等）对于这些词汇的语义表示能力有限，进而会影响后续任务的效果。

为了帮助解决这两个问题，好未来AI中台机器学习团队从多个来源收集了超过2000万条（约包含3.8亿Tokens）的教育领域中文ASR文本数据，基于此建立了教育领域首个在线教学中文预训练模型TAL-EduBERT，并把其推至开源。

从2018年谷歌发布预训练模型BERT以来，以BERT为代表的预训练语言模型， 在各个自然语言处理任务上都达到了SOTA的效果。并且作为通用的预训练语言模型，BERT的出现，使得NLP算法工程师不需要进行繁重的网络结构的修改，直接对于下游任务进行fine-tune，便可得到比以往的深度学习方法更好的效果，显著的减轻了NLP算法工程师的繁重的调整模型网络结构的工作，降低了算法应用的成本，预训练语言模型已经成为工作中不可或缺的一项基础技术。

但是，当前开源的各类中文领域的深度预训练模型，多是面向通用领域的应用需求，在包括教育在内的多个垂直领域均没有看到相关开源模型。相较于谷歌发布的Google BERT Base以及开源的中文Roberta模型，**好未来本次开源的TAL-EduBERT在多个教育领域的下游任务中得到了显著的效果提升**。好未来希望通过本次开源，助力推动 NLP技术在教育领域的应用发展，欢迎各位同仁下载使用。

### 2. 模型下载

下载地址：[https://ai.100tal.com/download/TAL-EduBERT.zip](https://ai.100tal.com/download/TAL-EduBERT.zip)

## 二、 模型结构及训练数据

### 1. 模型结构
TAL-EduBERT在网络结构上，采用与Google BERT Base相同的结构，包含12层的Transformer编码器、768个隐藏单元以及12个multi-head attention的head。之所以使用BERT Base的网络结构，是因为我们考虑到实际使用的便捷性和普遍性，后续会进一步开源其他教育领域ASR预训练语言模型。

### 2. 训练语料
TAL-EduBERT所采用的预训练语料，主要源于好未来内部积淀的海量教师教学语音经ASR转录而得到的文本，对于语料进行筛选、预处理后，选取了超过2000万条教育ASR文本，大约包含3.8亿Tokens。

### 3. 预训练方式
 
![Alt text](imgs/kjt.png?raw=true "")

如上图所示，TAL-EduBERT采取了与BERT相同的两种预训练任务来进行预训练学习，分别是教育领域字级别任务（Masked Language Modeling，简称MLM）和句子级别的训练任务（Next Sentence Prediction，简称NSP），通过这两个任务，使得TAL-EduBERT能够捕获教育ASR文本数据中的字、词和句子级别的语法和语义信息。

## 三、 下游任务实验结果
为了证明TAL-EduBERT在下游任务上的效果，我们从实际业务中抽取了4类典型的在线教育领域教学行为预测任务数据集，详见文献[3][4]。在此基础上，我们与Google BERT Base这一在中文领域应用最为广泛的模型以及效果较好的Roberta做了对比，实验结果表明，TAL-EduBERT在教育ASR下游任务上取得了较好的效果。

### 1. 实验简介：教师行为预测
此任务来源于我们对老师的教学行为进行智能化的评估，具体我们评估了四项教师行为，分别是引导学生进行课后总结（Conclude）、带着学生记笔记（Note）、表扬学生（Praise）和提问学生（QA）。通过对教师教学行为进行分类，给老师打上行为标签，从而更方便地分析老师教学行为，进而辅助老师更好地教学，提升教学质量。

### 2. 实验结果：
<table>
    <tr>
        <th colspan="2">Task\Model</th><th>Conclude</th><th>Note</th><th>Praise</th><th>QA</th>
    </tr>
    <tr>
        <td rowspan="2">Google BERT</td><td>Acc</td><td>0.7036</td><td>0.8436</td><td>0.8652</td><td>0.8948</td>
    </tr>
    <tr>
        <td>F1</td><td>0.6404</td><td>0.8356</td><td>0.8683</td><td>0.8469</td>
    </tr>
    <tr>
        <td rowspan="2">Roberta</td><td>Acc</td><td>0.7097</td><td>0.8558</td><td>0.8689</td><td>0.8979</td>
    </tr>
    <tr>
        <td>F1</td><td>0.6382</td><td>0.8464</td><td>0.8668</td><td>0.8433</td>
    </tr>
	<tr>
        <td rowspan="2">TAL-EduBERT</td><td>Acc</td><td>0.7270</td><td>0.8638</td><td>0.8731</td><td>0.9147</td>
    </tr>
    <tr>
        <td>F1</td><td>0.6486</td><td>0.8549</td><td>0.8688</td><td>0.8721</td>
    </tr>
</table>

## 四、 适用范围、使用方法及使用案例
### 1. 适用范围：
相较于Google BERT Base和Roberta，TAL-EduBERT基于大量教育ASR文本数据训练，因此对于ASR的识别错误具有较强的鲁棒性，并且在教育场景的下游任务上也具有较好的效果。鉴于此，我们推荐从事教育，并且工作内容与ASR文本相关的NLP算法工程师使用我们的模型，希望能通过本次的开源，推进自然语言处理在教育领域的应用和发展。

### 2. 使用方法：
与Google发布的原生BERT使用方式一致，支持transformers包，因此在使用时，直接进行模型路径替换即可。

### 3.使用案例：
```
from transformers import BertTokenizer, BertModel
import torch

path_to_TAL-EduBERT = "/YourPath/TAL-EduBERT/"

tokenizer = BertTokenizer.from_pretrained(path_to_TAL-EduBERT)
model = BertModel.from_pretrained(path_to_TAL-EduBERT)

sentence = "让我们来看一下这道题，这个题的也是一种比较经典类型的这个数列题目他呢，有个特点就是前面的是an+1，后面是一个an的式子加上一个根号下an的，一个二次的一个式子。"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state
```
## 五、 小结
为了证明TAL-EduBERT在教育领域下游任务的优势，我们从教育场景中的四类业务问题和数据入手进行了对比实验，对比Google BERT Base和Roberta这两种通用领域的预训练模型可知，TAL-EduBERT效果显著提升，在F1上最高提升大约3个百分点。因此，想要在教育领域进行NLP相关方向探索的技术伙伴可以直接使用TAL-EduBERT开展更专业地教育技术实践训练。

本文介绍了 TAL-EduBERT 的开源背景、数据背景、对比实验结果。后续，好未来AI中台也会持续进行理论创新和实践探索，进行更全面的开源开放，非常欢迎从事相关领域的伙伴们提供更多、更丰富的对比实验和实际应用案例，让我们共同推进自然语言处理技术在教育领域的应用和发展，为中国的教育事业注入新的动能。


## 参考文献：
    [1] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019.
    [2] Liu, Yinhan, et al. "Roberta: A robustly optimized BERT pretraining approach." arXiv preprint arXiv:1907.11692 (2019).
    [3] Huang, Gale Yan, et al. "Neural Multi-Task Learning for Teacher Question Detection in Online Classrooms." International Conference on Artificial Intelligence in Education. Springer, Cham, 2020. 
    [4] Xu, Shiting, Wenbiao Ding, and Zitao Liu. "Automatic Dialogic Instruction Detection for K-12 Online One-on-one Classes." International Conference on Artificial Intelligence in Education. Springer, Cham, 2020.

