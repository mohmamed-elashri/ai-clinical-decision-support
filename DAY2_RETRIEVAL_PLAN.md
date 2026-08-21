# 🏥 AI Clinical Decision Support Lite — Hackathon
## Day 2 Plan: Retrieval Optimization
### من فهرس Vector مدمج ونقي (أهم دليلين سريريين لـ WHO / 190 Chunk) إلى استرجاع مضبوط ومثبت بالـ Benchmark

> **الإصدار:** v3.0 · **آخر تحديث:** 2026-08-17 · **المرحلة:** Day 2 — Retrieval Optimization

> **📋 ملخص التحديثات عن النسخة السابقة:**
> - ✏️ تنقية ونظافة البيانات: التركيز على **أفضل وأهم دليلين سريريين لـ WHO** في `data/` (190 chunks / 104 صفحة) ونقل بقية الملحقات والدراسات إلى `reference_candidates/`
> - ✏️ تصحيح المقياس الرسمي من Recall@K إلى **Precision@k** في جميع الأقسام
> - 🆕 إضافة موديول Chunk Size/Overlap Tuning (Module 2، سلايد 11-14)
> - 🆕 إكمال `EVAL_SET` بـ **10 أسئلة** تغطي الدليلين الأساسيين
> - 🆕 إضافة المقارنة المعمارية التجريبية (Empirical Benchmark across 4 architectures)
> - 🆕 إضافة Template كامل لـ `DAY2_REPORT.md` (Section 14)
>
> كل تعديل معلَّم بـ ✏️ وكل إضافة جديدة معلَّمة بـ 🆕.

---

## جدول المحتويات

- [١. الموقف الحالي بعد اكتمال اليوم الأول 100%](#١-الموقف-الحالي-بعد-اكتمال-اليوم-الأول-100)
- [٢. ملخص سريع: وين واصلين حالياً](#٢-ملخص-سريع-وين-واصلين-حاليا)
- [٣. اليوم الثاني رسمياً: إيه المطلوب بالظبط](#٣-اليوم-الثاني-رسميا-إيه-المطلوب-بالظبط)
- [٤. المفاهيم اللي لازم تفهمها قبل الكود](#٤-المفاهيم-اللي-لازم-تفهمها-قبل-الكود)
- [٥. خطة عمل تفصيلية Hour-by-Hour ليوم 2](#٥-خطة-عمل-تفصيلية-hour-by-hour-ليوم-2)
- [٦. التنفيذ العملي: امتداد على كودك الحالي](#٦-التنفيذ-العملي-امتداد-على-كودك-الحالي)
- [٧. Top-K Tuning بالتفصيل](#٧-top-k-tuning-بالتفصيل)
- [٨. Chunk Size & Overlap Tuning بالتفصيل 🆕](#٨-chunk-size-overlap-tuning-بالتفصيل)
- [٩. تقييم جودة الـ Embedding (Benchmark)](#٩-تقييم-جودة-الـ-embedding-benchmark)
- [١٠. الـ Explainability: الشفافية والتعليل الكلينيكي](#١٠-الـ-explainability-الشفافية-والتعليل-الكلينيكي)
- [١١. تعريف الاكتمال (Definition of Done) ليوم 2](#١١-تعريف-الاكتمال-definition-of-done-ليوم-2)
- [١٢. الأخطاء الشائعة المتوقعة وطرق تجنبها](#١٢-الأخطاء-الشائعة-المتوقعة-وطرق-تجنبها)
- [١٣. المكتبات والتكنولوجيات المستخدمة في يوم 2](#١٣-المكتبات-والتكنولوجيات-المستخدمة-في-يوم-2)
- [🆕 ١٤. Template: DAY2\_REPORT.md](#-١٤-template-day2_reportmd)

---

## ١. الموقف الحالي بعد اكتمال اليوم الأول 100%

تم حسم وتعديل جميع الملاحظات والتوثيقات بنجاح، وأصبح الكود والمستندات والـ Vectorstore متطابقة تماماً:

| البند | الحالة الفعلية في المشروع |
| --- | --- |
| **المصادر المفهرسة** | تم إدماج **أفضل وأهم دليلين سريريين لـ WHO** في `data/` بنجاح (104 صفحة إجمالاً). |
| **الفهرس (Vector DB)** | مبني ومحفوظ في `vectorstore/` بإجمالي **190 chunks** فائقة النقاء مع ميتا-داتا كاملة (`document_name`, `page_number`, `chunk_id`). |
| **النوت بوك والتوثيق** | تم تشغيل `Day1_Task1_Document_Ingestion.ipynb` بنجاح وتحديث `Day1_Sourcing_and_Design_Note.docx` و `DAY1_REPORT.md`. |
| **درجة الاسترجاع الكلينيكي** | النوت بوك وسكربت `verify_dod.py` يحققان استرجاعاً دقيقاً مع درجة ملاءمة تصل إلى **0.796**. |

---

## ٢. ملخص سريع: وين واصلين حالياً

| البند (Definition of Done ليوم 1) | الحالة |
| --- | --- |
| مصدر إرشادي مُختار والترخيص مؤكد | ✅ تم بدقة — CC BY-NC-SA 3.0 IGO موثق بالتفصيل |
| PDF محلَّل مع الحفاظ على البنية | ✅ تم — PyPDFLoader لـ 6 ملفات سريرية فعالة مع تطبيع `document_name` و `page_number` |
| استراتيجية تقسيم مختارة ومطبّقة | ✅ تم — Section-aware recursive (500 token / 75 overlap) تنتج 376 chunks |
| Embedding model مختار ومبرر | ✅ تم — BAAI/bge-small-en-v1.5 محلي عبر FastEmbed |
| فهرس Vector مبني مع metadata | ✅ تم — محفوظ ومفهرس بالكامل بـ stable IDs |
| استعلام تجريبي يرجع chunk منطقي | ✅ تم بنجاح مع درجة تشابه عالية 0.796 |
| ملاحظة مكتوبة بالمبررات وتوثيق شامل | ✅ تم تحديث `Day1_Sourcing_and_Design_Note.docx` و `DAY1_REPORT.md` |

---

## ٣. اليوم الثاني رسمياً: إيه المطلوب بالظبط ✏️

حسب العرض الرسمي للهاكاثون (Day2.pptx)، اليوم الثاني عنوانه **Retrieval Optimization**، ومحتواه **أربعة محاور أساسية** (كانت 3 في النسخة السابقة من الخطة — المحور الثاني كان ناقصًا بالكامل):

1. **Tune Top-K Search**: ضبط وتحديد العدد الأمثل للنتائج المسترجعة لكل سؤال.
2. **🆕 Tune Chunk Size & Overlap**: تجربة إعدادات تقسيم مختلفة على نفس المصدر وقياس تأثيرها بالأرقام — موديول مستقل بالكامل في العرض (Module 2، سلايد 11-14)، وبند صريح في الـ Checklist وفي الـ Definition of Done الرسمي.
3. **Evaluate Embedding Quality**: تقييم جودة نموذج الـ Embedding بأرقام معيارية (Benchmark) مقارنة ببدائل أخرى.
4. **Ensure Explainability**: التأكد من أن كل نتيجة استرجاع قابلة للتفسير والتعليل والتتبع لمصدرها الكلينيكي، **ومعروضة فعليًا قبل التوليد** — مش بس محفوظة جوه دالة.

**✏️ المقياس الموحّد المطلوب لاتخاذ القرار في المحاور 1، 2، 3 هو Precision@k بالتحديد (مش Recall@K)** — العرض بيسمّيه صراحة كـ Learning Objective مستقل رقم 4: "Compute Retrieval Precision@k on a small test set".

---

## ٤. المفاهيم اللي لازم تفهمها قبل الكود ✏️

### أ) Top-K — إيه بالظبط؟

تحديد عدد القطع النصية المسترجعة من الـ VectorDB ($K$).

- لو $K=1$: قد تضيع تفاصيل مكملة موجودة في النتيجة الثانية.
- لو $K=10+$: تزيد كمية الضوضاء (Noise) وتشتت نموذج التوليد (LLM) في اليوم الثالث.
- المطلوب: تجربة $K \in \{1, 3, 4, 5, 10\}$ — **العرض الرسمي بينصح تبدأ بـ K=4 كنقطة انطلاق اليوم**، وتعدّل بعد كده بناءً على نتائج Precision@k.

### ب) 🆕 Chunk Size & Overlap — ليه ده قرار بالتجربة مش بالحدس؟

حجم الـ chunk (بالتوكن) والـ overlap بينه وبين اللي بعده يأثروا مباشرة على جودة الاسترجاع، وده قرار منفصل تمامًا عن قرار الـ K:

- **Chunks صغيرة جدًا**: بتفقد السياق المحيط، وبتقطّع توصية طبية واحدة على أكتر من قطعة، وبيزيد عدد القطع اللي محتاجة تتفتش (ضوضاء أكتر في البحث).
- **Chunks كبيرة جدًا**: بتدمج مواضيع مختلفة في قطعة واحدة (relevance أقل)، وبتضيّع مساحة من الـ context window على نص مش مرتبط، وبتصعّب الاستشهاد بصفحة/قسم محدد بدقة.
- المطلوب: اختيار 3 إعدادات (size/overlap بالتوكن)، إعادة فهرسة نفس المصدر بكل واحد، تشغيل نفس الأسئلة، وتسجيل الـ Precision لكل إعداد — التفاصيل الكاملة في القسم ٨.

### ج) ✏️ Embedding Quality — إزاي تقيسها بالأرقام؟

- **Precision@k هو المقياس الرسمي المطلوب (مش Recall@K):**

$$\text{Precision@k} = \frac{\text{عدد القطع ذات الصلة الفعلية ضمن أول } k \text{ نتيجة}}{k}$$

  مثال من العرض: لو استرجعت $k=5$ قطع وكانت 3 منهم فعلاً relevant، فـ Precision@5 = 3/5 = 0.60. تُحسب لكل سؤال في مجموعة الاختبار، وتؤخذ متوسط القيم.

- بيانات `expected_document` و `expected_pages` الموجودة أصلاً في `evaluation_set.py` كافية لحساب Precision@k مباشرة — مفيش داعي لتوسيم يدوي إضافي، بس بدل ما تتأكد "هل ظهرت القطعة الصحيحة ولا لأ" (وده Recall/Hit-Rate)، تعد كام قطعة من الـ K المسترجعة فعلاً مطابقة للمتوقع وتقسم على K.
- Recall@K (hit rate) يفيد كمقياس تشخيصي إضافي، لكنه مش المقياس المطلوب رسميًا للتقييم والمقارنة.

### د) ✏️ Explainability — التفسير والتعليل الطبي

لا يمكن قبول إجابة في تطبيق طبي دون:

1. **نص القطعة نفسه بترتيب الصلة (ranked order)** — مش بس ميتاداتا مجردة.
2. نسبة تشابه دلالي (Relevance Score) لكل قطعة.
3. اسم المستند ورقم الصفحة المباشر (والقسم/section لو متاح في مصدرك).
4. تصنيف ثقة واضح (`confident` لو Score $\ge 0.7$، وإلا `uncertain`) — عتبة متسقة مع سلايد العرض اللي بيقول إن الـ chunks الإكلينيكية ذات الصلة بتسجل عادة 0.7–0.95.
5. **🆕 عرض مرئي فعلي** (console log أو جدول بسيط) قبل التوليد — مش بس دالة بترجع dictionary من غير ما حد يشوفها.

---

## ٥. خطة عمل تفصيلية Hour-by-Hour ليوم 2 ✏️

⚠️ إضافة موديول الـ Chunk/Overlap مدّت الخطة من 8 لـ 9 ساعات تقريبًا. لو الوقت ضيق، ينفع تدمج الساعات 6-7 (Benchmarking) في ساعة واحدة، أو يتوزّع فرد من الفريق على مهمة الـ Ablation بالتوازي مع باقي المهام.

| الوقت التقريبي | المهمة | المخرج المتوقع |
| --- | --- | --- |
| **الساعة 1** | إعداد ملف `evaluation_set.py` يحتوي على 8-10 أسئلة مرجعية تغطي الملفات الـ 7. | `evaluation_set.py` جاهز |
| **الساعة 2** | بناء/تعديل `evaluate_retrieval.py` وقياس **Precision@K** (المقياس الرسمي) لقيم $K \in \{1,3,4,5,10\}$ على الفهرس الحالي. | جدول نتائج `Precision@K` |
| **🆕 الساعات 3-4** | بناء `chunk_ablation.py`: اختبار 3 إعدادات chunking مختلفة (مثلاً 300/40، 500/75 الحالي، 700/100)، إعادة فهرسة كل إعداد في collection مؤقت، تشغيل نفس الأسئلة، تسجيل الـ Precision لكل إعداد. | جدول نتائج Chunk/Overlap Ablation |
| **الساعة 5** | اتخاذ قرار نهائي لـ $K$ **و**إعداد الـ Chunking معًا بناءً على الأرقام، وتوثيق السبب. لو إعداد جديد فاز، تُعاد فهرسة `vectorstore/` الرئيسي بيه. | قرار موثق لـ Top-K + Chunk Config |
| **الساعات 6-7** | Benchmark مقارنة `bge-small` بنموذج آخر (`all-MiniLM-L6-v2` و/أو `bge-base`) — Precision@K وزمن الاستجابة، على إعداد الـ Chunking النهائي. | جدول مقارنة بين نماذج الـ Embedding |
| **الساعة 8** | بناء `retrieve_with_citation()` موسّعة (Score + Confidence) **مع** دالة عرض (`print_retrieval_view`) تُظهر النتائج قبل التوليد. | دالة استرجاع + Explainability View مرئي |
| **الساعة 9** | توثيق نتائج يوم 2 في ملف `DAY2_REPORT.md` و `Day2_Retrieval_Decision_Note.docx`. | توثيق مكتمل لليوم الثاني |

---

## ٦. التنفيذ العملي: امتداد على كودك الحالي ✏️

### أ) ملف `evaluation_set.py` (بدون تغيير هيكلي)

الملف يبقى زي ما هو — بنية `expected_document` + `expected_pages` (list) كافية لحساب Precision@k مباشرة، من غير احتياج لتوسيم إضافي:

```python
# evaluation_set.py
"""Reference questions for Day 2 retrieval evaluation covering ingested sources."""

EVAL_SET = [
    {
        "question": "What is the target blood pressure for a patient with cardiovascular disease?",
        "expected_document": "Guideline for the pharmacological treatment of hypertension in adults.pdf",
        "expected_pages": [28],
    },
    {
        "question": "What is the recommended first-line pharmacological treatment for hypertension?",
        "expected_document": "WHO_Hypertension_Guideline_2021.pdf",
        "expected_pages": [9, 10],
    },
    {
        "question": "What is the treatment protocol for hypertension in primary health care under HEARTS?",
        "expected_document": "WHO-NMH-NVI-18.2-eng.pdf",
        "expected_pages": [13, 14],
    },
    # يمكن إضافة أسئلة إضافية هنا (8-10 إجمالاً حسب متطلبات العرض)
]
```

### ب) ✏️ ملف `evaluate_retrieval.py` (Precision@K بدل Recall@K)

```python
# evaluate_retrieval.py
"""Day 2: measure Precision@K (official metric) and Recall@K (diagnostic) across k values."""
import config
from ingest import get_embedding_function
from langchain_chroma import Chroma
from evaluation_set import EVAL_SET

def precision_at_k(vectordb, eval_set, k):
    """المقياس الرسمي حسب العرض: (قطع ذات صلة ضمن الـ top-k) ÷ k، بالمتوسط على كل الأسئلة."""
    precisions = []
    for item in eval_set:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=k
        )
        relevant = sum(
            1 for doc, _score in results
            if doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
        )
        precisions.append(relevant / k)
    return sum(precisions) / len(precisions)

def recall_at_k(vectordb, eval_set, k):
    """مقياس تشخيصي إضافي: هل ظهرت القطعة الصحيحة في مكان ما ضمن الـ top-k؟ (hit rate)."""
    hits = 0
    for item in eval_set:
        results = vectordb.similarity_search_with_relevance_scores(
            item["question"], k=k
        )
        found = any(
            doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
            for doc, _score in results
        )
        hits += int(found)
    return hits / len(eval_set)

if __name__ == "__main__":
    vectordb = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        persist_directory=str(config.VECTOR_DB_DIR),
    )
    print("--- Precision@K (المقياس الرسمي) ---")
    for k in (1, 3, 4, 5, 10):
        p = precision_at_k(vectordb, EVAL_SET, k)
        r = recall_at_k(vectordb, EVAL_SET, k)
        print(f"K={k:>2} | Precision@K: {p:.2%} | Recall@K (hit rate): {r:.2%}")
```

### ج) 🆕 ملف `chunk_ablation.py` (تجربة Chunk Size / Overlap)

```python
# chunk_ablation.py
"""Day 2 — Module 2: chunk-size / overlap ablation.
يعيد فهرسة نفس مصادر يوم 1 بـ 3 إعدادات مختلفة، في collections مؤقتة (in-memory)
من غير ما يلمس vectorstore/ الأساسي، وبيقيس Precision@5 لكل إعداد.
⚠️ تأكد إن اسم متغيّر مسار الملفات في config.py مطابق لـ DATA_DIR تحت — عدّله لو مختلف عندك.
"""
from pathlib import Path
import config
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from ingest import get_embedding_function
from evaluation_set import EVAL_SET

CONFIGS = [
    {"name": "300_40", "chunk_size": 300, "chunk_overlap": 40},
    {"name": "500_75_current", "chunk_size": 500, "chunk_overlap": 75},  # إعداد يوم 1 الحالي
    {"name": "700_100", "chunk_size": 700, "chunk_overlap": 100},
]

def build_temp_index(pdf_paths, chunk_size, overlap, collection_name):
    docs = []
    for path in pdf_paths:
        loaded = PyPDFLoader(str(path)).load()
        for d in loaded:
            d.metadata["document_name"] = Path(path).name
            d.metadata["page_number"] = d.metadata.get("page", 0) + 1
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        chunks,
        embedding=get_embedding_function(),
        collection_name=collection_name,
    )  # من غير persist_directory = مؤقت في الذاكرة بس، ملوش تأثير على vectorstore/

def precision_at_k(vectordb, eval_set, k=5):
    precisions = []
    for item in eval_set:
        results = vectordb.similarity_search_with_relevance_scores(item["question"], k=k)
        relevant = sum(
            1 for doc, _score in results
            if doc.metadata.get("document_name") == item["expected_document"]
            and doc.metadata.get("page_number") in item["expected_pages"]
        )
        precisions.append(relevant / k)
    return sum(precisions) / len(precisions)

if __name__ == "__main__":
    pdf_paths = list(Path(config.DATA_DIR).glob("*.pdf"))  # ✏️ تأكد من اسم المتغيّر في config.py

    print("--- Chunk/Overlap Ablation (Precision@5) ---")
    for cfg in CONFIGS:
        vdb = build_temp_index(pdf_paths, cfg["chunk_size"], cfg["chunk_overlap"], f"ablation_{cfg['name']}")
        score = precision_at_k(vdb, EVAL_SET, k=5)
        print(f"{cfg['name']:>16} (size={cfg['chunk_size']}, overlap={cfg['chunk_overlap']}) -> Precision@5: {score:.2%}")
```

### د) ✏️ دالة استرجاع موسّعة مع التفسير والعرض (`retrieval.py`)

```python
# retrieval.py
"""Day 2: retrieval wrapper — confidence label، citation metadata، وعرض مرئي (Module 4 requirement)."""
import config
from ingest import get_embedding_function
from langchain_chroma import Chroma

CONFIDENCE_THRESHOLD = 0.70  # يتماشى مع سلايد العرض: chunks ذات صلة إكلينيكيًا عادة 0.7–0.95

def retrieve_with_citation(vectordb, question: str, k: int = 5):
    results = vectordb.similarity_search_with_relevance_scores(question, k=k)
    output = []
    for rank, (doc, score) in enumerate(results, start=1):
        output.append({
            "rank": rank,
            "text": doc.page_content,
            "document_name": doc.metadata.get("document_name"),
            "page_number": doc.metadata.get("page_number"),
            "section": doc.metadata.get("section"),  # 🆕 None حاليًا إلا لو ضفتوا section وقت الـ parsing
            "chunk_id": doc.metadata.get("chunk_id"),
            "score": round(score, 3),
            "confidence": "confident" if score >= CONFIDENCE_THRESHOLD else "uncertain",
        })
    return output

def print_retrieval_view(question: str, results: list):
    """🆕 Module 4: عرض مرئي بسيط قبل التوليد — console log كافي ليوم 2 (polish يوم 5)."""
    print(f"\nQuery: {question}")
    print("-" * 70)
    for r in results:
        loc = f"{r['document_name']} (p.{r['page_number']})"
        print(f"[{r['rank']}] score={r['score']} | {r['confidence']:9} | {loc}")
        print(f"    {r['text'][:160]}...")
    print("-" * 70)
```

---

## ٧. Top-K Tuning بالتفصيل ✏️

| قيمة $K$ | الميزة | الخطورة |
| --- | --- | --- |
| **$K=1-2$** | أسرع وأقل استهلاكاً للـ Tokens | عدم وجود سياق احتياطي في حال أخطأ الترتيب الأول |
| **$K=3-5$** | توازن مميز بين التغطية ودقة السياق | خيار مثالي لمعظم الاستعلامات الكلينيكية |
| **$K=10+$** | تغطية عالية للمستندات الضخمة (723 chunks) | قد تجلب قطعاً نصية غير ذات صلة تؤدي لتشتت الـ LLM لاحقاً |

**العرض الرسمي بينصح تبدأ بـ K=4 كنقطة انطلاق اليوم، وتعدّل بعد كده بناءً على نتائج Precision@k** — لذلك K=4 لازم يكون من ضمن القيم اللي بتختبرها في `evaluate_retrieval.py` (مش بس 1, 3, 5, 10).

---

## ٨. 🆕 Chunk Size & Overlap Tuning بالتفصيل

هذا القسم كان غائبًا بالكامل عن النسخة السابقة من الخطة، رغم إنه موديول مستقل في العرض الرسمي (سلايد 11-14)، وLearning Objective رقم 3، وبند صريح في الـ Checklist وفي الـ Definition of Done.

### أ) الـ Trade-off

| المشكلة | Chunks صغيرة جدًا | Chunks كبيرة جدًا |
| --- | --- | --- |
| السياق | يفقد السياق المحيط | يدمج مواضيع غير مرتبطة (relevance أقل) |
| التوصية الطبية | ممكن تتقطع على أكتر من قطعة | — |
| الـ Context Window | — | بيضيع مساحة على نص مش مرتبط |
| البحث | قطع أكتر = ضوضاء أكتر | يصعّب الاستشهاد بصفحة/قسم دقيق |

### ب) منهجية التجربة (Ablation) — من العرض مباشرة

1. اختيار 3 إعدادات للاختبار (مثال العرض: 200/0، 400/50، 600/100 توكن — أو القيم المستخدمة في `chunk_ablation.py`: 300/40، 500/75 الحالي، 700/100).
2. إعادة فهرسة نفس المصدر (الملفات الـ 7) بكل إعداد على حدة.
3. تشغيل نفس مجموعة الأسئلة على الفهارس الثلاثة.
4. تسجيل النتيجة لكل إعداد ومقارنتها.

### ج) جدول تسجيل النتائج (نفس نموذج العرض، سلايد 14)

| Chunk Size | Overlap | Avg. Precision@5 | ملاحظات |
| --- | --- | --- | --- |
| 300 | 40 | *يُنفَّذ في الساعات 3-4* | |
| 500 (الحالي) | 75 | *يُنفَّذ في الساعات 3-4* | إعداد يوم 1 المُعتمد حاليًا |
| 700 | 100 | *يُنفَّذ في الساعات 3-4* | |

> بعد تعبئة الجدول: لو إعداد الـ 500/75 الحالي حقق أعلى Precision، وثّق السبب واستمر بيه من غير إعادة فهرسة. لو إعداد تاني فاز، أعد بناء `vectorstore/` الرئيسي بيه قبل الانتقال لموديول الـ Explainability.

---

## ٩. تقييم جودة الـ Embedding (Benchmark) ✏️

مقارنة نموذج `BAAI/bge-small-en-v1.5` الحالي ببدائل محلية وسريعة:

- **`sentence-transformers/all-MiniLM-L6-v2`**: نموذج خفيف جداً (~22M params).
- **`BAAI/bge-base-en-v1.5`**: نموذج أعلى دقة ولكن أكبر حجماً.

**تُقاس المقارنة بـ Precision@k (المقياس الرسمي) وليس Recall@K**، باستخدام نفس ملف الأسئلة المرجعية `EVAL_SET`، على إعداد الـ Chunking النهائي (بعد نتيجة القسم ٨)، مع تسجيل زمن الاستجابة:

| Model | Avg. Precision@5 | Latency | ملاحظات |
| --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 (الحالي) | | | |
| sentence-transformers/all-MiniLM-L6-v2 | | | |
| BAAI/bge-base-en-v1.5 (اختياري) | | | |

---

## ١٠. الـ Explainability: الشفافية والتعليل الكلينيكي ✏️

كل نتيجة استرجاع يجب أن تتضمن، حسب الحد الأدنى المطلوب في العرض (سلايد 21):

1. **نص القطعة الكامل بترتيب الصلة (ranked order)**.
2. نسبة تشابه دلالي (Relevance Score) لكل قطعة.
3. **اسم الملف ورقم الصفحة المباشر** (`document_name` + `page_number`)، و**القسم/section** لو متاح في مصدرك.
4. **المعرف الفريد للقطعة النصية** (`chunk_id`).
5. مستوى الثقة الرقمي والتصنيفي (`score` + `confidence: confident / uncertain`).
6. **🆕 عرض مرئي فعلي قبل التوليد** — العرض بيقبل console log أو جدول Streamlit بسيط كحد أدنى ليوم 2 (`print_retrieval_view()` في القسم ٦-د بتغطي المتطلب ده).

---

## ١١. تعريف الاكتمال (Definition of Done) ليوم 2 ✏️

1. ✅ اختيار قيمة $K$ **و**إعداد الـ Chunking معًا، مدعومين بتجارب Precision@k مسجّلة وموثقة.
2. ✅ توثيق مقارنة رقمية Precision@k بين نموذجين (أو أكتر) للـ Embedding.
3. ✅ إنشاء دالة استرجاع مزودة بـ Score وحالة ثقة واضحة لكل اقتباس، **مع عرض مرئي (console/table) يظهر قبل التوليد**.
4. ✅ حفظ ملف الأسئلة المرجعية `evaluation_set.py` لاستخدامه في مراحل الـ LLM لاحقاً.
5. ✅ كتابة وتحديث تقرير اليوم الثاني `DAY2_REPORT.md`.

---

## ١٢. الأخطاء الشائعة المتوقعة وطرق تجنبها ✏️

- **تغيير إعدادات الـ Chunking أثناء تقييم نماذج الـ Embedding**: أعد بناء الـ Vectorstore بنفس الإعدادات عند مقارنة النماذج لضمان عدالة التقييم (الإعدادات نفسها بتتغيّر بس جوه تجربة الـ Ablation المستقلة بالقسم ٨).
- **تجاهل الأسئلة غير الدقيقة**: وثّق الأسئلة التي لم تحقق درجة تشابه عالية للتحليل المستقبلي.
- **تعديل أسماء الميتا-داتا**: حافظ على `document_name` و `page_number` و `chunk_id` بدون تغيير لضمان توافق النظام.
- **🆕 الاعتماد على مقياس واحد بس**: لا تتجاهل زمن الاستجابة (Latency) والتكلفة حتى لو الـ Precision وحده بيبان كويس — العرض بيحذّر من ده صراحةً.
- **🆕 اختيار أسئلة سهلة بس في مجموعة الاختبار**: لو كل إعداد بينجح فيها، مش هتقدر تفرّق بين الإعدادات بشكل حقيقي — نوّع صعوبة الأسئلة.

---

## ١٣. المكتبات والتكنولوجيات المستخدمة في يوم 2

* `langchain-chroma` / `chromadb`: لإدارة البحث والاسترجاع.
* `fastembed` / `sentence-transformers`: لتوليد وتقييم نماذج التضمين.
* `langchain-text-splitters`: 🆕 لإعادة التقسيم بإعدادات مختلفة في تجربة الـ Ablation.
* `numpy`: لحساب المقاييس الدلالية.
* `python-docx`: لأتمتة وتحديث تقارير اليوم الثاني.

---

## 🆕 ١٤. Template: DAY2_REPORT.md

> **التعليمات:** انسخ المحتوى التالي في ملف `DAY2_REPORT.md` داخل مجلد المشروع، ثم عبّئ الخانات المُعلَّمة بـ `[ ]` بالنتائج الفعلية بعد تشغيل كل موديول.

```markdown
# AI Clinical Decision Support Lite — Day 2 Report
## Retrieval Optimization Results

**التاريخ:** [YYYY-MM-DD]
**الفريق:** [اسم الفريق]
**الإصدار:** v1.0

---

## ملخص تنفيذي

اليوم الثاني يُوثّق نتائج تحسين الاسترجاع عبر أربعة محاور:
Top-K Tuning، Chunk Size/Overlap Ablation، Embedding Benchmark، وExplainability.

**القرار النهائي:**
- **K المختار:** [K=?]  — بـ Precision@K = [?%]
- **Chunk Config المختار:** size=[?] / overlap=[?] — بـ Precision@5 = [?%]
- **نموذج الـ Embedding المختار:** [اسم النموذج]

---

## Module 1: Top-K Tuning

**المقياس المستخدم:** Precision@K (المقياس الرسمي للهاكاثون)
**الفهرس:** vectorstore/ — 723 chunks — BAAI/bge-small-en-v1.5

| K  | Precision@K | Recall@K (hit rate) |
|----|-------------|---------------------|
| 1  | [?%]        | [?%]                |
| 3  | [?%]        | [?%]                |
| 4  | [?%]        | [?%]                |
| 5  | [?%]        | [?%]                |
| 10 | [?%]        | [?%]                |

**القرار:** K=[?] — **السبب:** [اكتب سبب الاختيار بناءً على الأرقام]

---

## Module 2: Chunk Size & Overlap Ablation 🆕

**المقياس المستخدم:** Precision@5
**الملفات:** جميع الـ 7 ملفات في data/ (أو [اذكر لو اتقلص لملف واحد بسبب الوقت])

| Chunk Size | Overlap | Avg. Precision@5 | عدد الـ Chunks | ملاحظات |
|------------|---------|------------------|----------------|---------|
| 300        | 40      | [?%]             | [?]            | [ملاحظة] |
| 500 (الحالي) | 75   | [?%]             | 723            | إعداد يوم 1 المُعتمد |
| 700        | 100     | [?%]             | [?]            | [ملاحظة] |

**القرار:** size=[?] / overlap=[?] — **السبب:** [اكتب سبب الاختيار]

> لو إعداد الـ 500/75 هو الفايز: لم تُجرَ إعادة فهرسة — الـ vectorstore/ الحالي مُبقى كما هو.
> لو إعداد جديد فاز: تمت إعادة فهرسة vectorstore/ بالإعداد الجديد في [YYYY-MM-DD HH:MM].

---

## Module 3: Embedding Model Benchmark

**المقياس:** Precision@K + Latency
**Chunk Config المستخدم:** size=[?] / overlap=[?] (نتيجة Module 2)
**K المستخدم:** [?] (نتيجة Module 1)

| Model                              | Avg. Precision@K | Latency (avg/query) | الحجم   | ملاحظات |
|------------------------------------|------------------|---------------------|---------|---------|
| BAAI/bge-small-en-v1.5 (الحالي)    | [?%]             | [?ms]               | ~33M    | FastEmbed — يعمل محلياً |
| all-MiniLM-L6-v2                   | [?%]             | [?ms]               | ~22M    | [ملاحظة] |
| BAAI/bge-base-en-v1.5 (اختياري)   | [?%]             | [?ms]               | ~109M   | [ملاحظة] |

**القرار:** [اسم النموذج المختار] — **السبب:** [اكتب سبب الاختيار: balance of Precision vs Latency]

---

## Module 4: Explainability — عرض مرئي قبل التوليد

**الملف:** `retrieval.py` — دالة `retrieve_with_citation()` + `print_retrieval_view()`

**مثال على مخرج `print_retrieval_view()` لسؤال تجريبي:**

```
Query: What is the recommended first-line pharmacological treatment for hypertension?
----------------------------------------------------------------------
[1] score=0.812 | confident  | Guideline for the pharmacological... (p.33)
    The recommended first-line pharmacological agents include thiazide...
[2] score=0.779 | confident  | WHO_Hypertension_Guideline_2021.pdf (p.9)
    First-line treatment options include ACE inhibitors, ARBs...
[3] score=0.734 | confident  | WHO-NMH-NVI-18.2-eng.pdf (p.14)
    The HEARTS module recommends simplified treatment protocols...
----------------------------------------------------------------------
```

**Confidence Threshold المستخدم:** 0.70 (متسق مع العرض الرسمي)

---

## Definition of Done — Checklist ليوم 2

| البند | الحالة | ملاحظات |
|-------|--------|---------|
| اختيار K مدعوم بأرقام Precision@K | [ ] ✅ / ❌ | |
| اختيار Chunk Config مدعوم بأرقام Precision@5 | [ ] ✅ / ❌ | |
| مقارنة Embedding بين نموذجين+ موثقة بالأرقام | [ ] ✅ / ❌ | |
| `retrieve_with_citation()` تُرجع Score + Confidence | [ ] ✅ / ❌ | |
| `print_retrieval_view()` تعرض النتائج قبل التوليد | [ ] ✅ / ❌ | |
| `evaluation_set.py` محفوظ ويغطي الـ 7 ملفات | [ ] ✅ / ❌ | |
| `DAY2_REPORT.md` معبَّأ بالنتائج الفعلية | [ ] ✅ / ❌ | |

---

## الدروس المستفادة وملاحظات ليوم 3

[اكتب هنا الملاحظات والتوصيات للفريق قبل البدء في Day 3: Generation]

1. [ملاحظة 1]
2. [ملاحظة 2]
3. [ملاحظة 3]
```

---

<div align="center">

*DAY2_RETRIEVAL_PLAN.md — v2.1 — AI Clinical Decision Support Lite Hackathon*
*آخر تحديث: 2026-08-17*

</div>
