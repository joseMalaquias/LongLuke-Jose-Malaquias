# LongLuke-Jose-Malaquias
Source code of the Masters Thesis - IST Lisboa



## This repository contains all the essential code to reproduce the experiments performed during the thesis implementation. 

### **Abstract**
Recent Pre-trained Language Models (PLM), based on self-attention and the Transformer neural architecture, produce a contextualized representation of words. Leveraging such representations in other neural networks enabled state-of-the-art results in the corresponding downstream Natural Language Processing (NLP) tasks. A new line of research focuses on studying how to incorporate named entity components within a PLM. The motivation is that named entities convey essential information that can further improve the performance of a model in a NLP task. However, current entity-focused models face limitations, such as the inability to process long inputs. This is especially noticeable in recent complex tasks which demand good understanding over long sequences of text. In this article, we propose Long-LUKE, a model based on the entity-aware LUKE PLM that is capable of processing long sequences of text. In particular, it obtains interesting results on four well-known datasets: Open Entity, FIGER and DocRED (entity typing), and TACRED and ReDocRED~(relation classification).


Some of the model's weights can be found in this
[link](https://drive.google.com/drive/folders/1QX_3tfyu8A0C6HpOSnhgODJFQHpIIKZb?usp=share_link).

List of datasets used:
1. [OpenEntity](http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz)
2. [FIGER](https://github.com/xiaoling/figer/zipball/master)
3. [TA-CRED](https://nlp.stanford.edu/projects/tacred/#access)
4. [DocRED](https://github.com/thunlp/DocRED)
5. [Revisiting DocRED](https://github.com/tonytan48/Re-DocRED/tree/main/data)

The finetuned baseline models of LUKE were obtained from [Huggingface/studio-ousia](https://huggingface.co/studio-ousia).

