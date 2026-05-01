"""
FakeReview Buster v2 — Proper ML Engine
Models: TF-IDF + Char n-gram + 22 Heuristic Features
        → RandomForest | GradientBoosting | LogisticRegression | SVM
        → VotingClassifier Ensemble (soft voting)
Cross-validation: StratifiedKFold 10-fold
"""

import re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.metrics import (accuracy_score, classification_report,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
import scipy.sparse as sp

warnings.filterwarnings('ignore')

# ── Inline stopwords (no NLTK needed) ────────────────────────────────────────
STOPWORDS = set("""a about above after again against all am an and any are
aren't as at be because been before being below between both but by can't cannot
could couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself no
nor not of off on once only or other ought our ours ourselves out over own same
shan't she she'd she'll she's should shouldn't so some such than that that's the
their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll
we're we've were weren't what what's when when's where where's which while who
who's whom why why's will with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves""".split())


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)


# ── Heuristic Feature Extractor ───────────────────────────────────────────────
class HeuristicFeatureExtractor(BaseEstimator, TransformerMixin):
    SUPERLATIVES = {
        'best','greatest','amazing','incredible','perfect','outstanding',
        'exceptional','wonderful','fantastic','excellent','awesome','superb',
        'extraordinary','phenomenal','fabulous','flawless','spectacular',
        'unbelievable','absolutely','definitely','totally','completely',
        'utterly','entirely','extremely','incredibly','purely','simply'
    }
    GENERIC_PHRASES = [
        'highly recommend','will definitely','fast shipping','great product',
        'great quality','great price','five stars','5 stars','love it',
        'works perfectly','as described','buy this','great seller',
        'must buy','game changer','life changing','changed my life',
        'best purchase','best product','best ever','highly recommended',
        'would recommend','definitely buy again','beyond expectations',
        'exceeded expectations','cannot go wrong','worth every penny',
        'do not hesitate','trust me','do yourself a favor',
        'perfect in every way','exactly what everyone needs'
    ]
    AI_STYLE_PHRASES = [
        'overall','in conclusion','from the moment','exceeded my expectations',
        'seamless experience','top notch','elevates','solid choice',
        'anyone looking for','whether you are','not only','but also',
        'impressed by the quality','perfect addition','stands out',
        'highly versatile','user friendly','makes it easy','look no further',
        'worth considering','ideal for anyone','i was pleasantly surprised',
        'well balanced','well-rounded','designed to','crafted to',
        'offers a great combination','great combination of',
        'combination of quality','blend of quality','reliable performance',
        'exceptional value','enhances the experience','delivers excellent',
        'delivers a satisfying','for anyone seeking','great choice for',
        'perfect choice for','high-quality experience','premium feel',
        'attention to detail','impressive performance','thoughtful design',
        'highly recommend this','excellent choice','great option',
        'fantastic choice','stands out as','makes it a great',
        'caters to','suitable for','meets expectations',
        'easy to use','built to last','worth the investment',
        'value for money','modern design','sleek design',
        'reliable and efficient','quality and performance',
        'coming back again','friendly fast and professional',
        'clean efficient and very user-friendly',
        'small delay','minor improvements','room for improvement',
        'gets the job done','nothing extraordinary',
        'customer support helped','technical glitches',
        'major improvements','multiple issues and delays',
        'did not meet expectations','lack of proper support',
        'improve significantly','wish there were more',
        'more options','more features','not bad',
        'expected a bit more','works fine','could be smoother',
        'could be better','average at best','might try again'
    ]
    FAMILY_REFS = [
        'my husband','my wife','my mother','my father','my friend',
        'my family','my sister','my brother','as a gift','bought as gift',
        'for my mom','for my dad','my kids','my children','my neighbor',
        'my coworker','everyone i know','all my friends','whole family'
    ]
    SPECIFIC_UNITS = re.compile(
        r'\d+\.?\d*\s*(hours?|days?|weeks?|months?|years?|cm|mm|kg|lbs?|'
        r'pounds?|inch(?:es)?|feet|foot|gb|mb|tb|watts?|volts?|amps?|'
        r'°[fc]|degrees?|miles?|km|liters?|gallons?|oz|ounces?|sq\s?ft|'
        r'minutes?|seconds?)', re.I)

    def fit(self, X, y=None): return self

    def transform(self, X):
        return np.array([self._extract(str(t)) for t in X], dtype=np.float32)

    def _extract(self, text):
        tl   = text.lower()
        words = tl.split()
        wc   = max(len(words), 1)
        chars = max(len(text), 1)

        excl_count     = text.count('!')
        excl_ratio     = excl_count / chars
        caps_words     = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        caps_ratio     = caps_words / wc
        sup_hits       = sum(1 for w in words if w in self.SUPERLATIVES)
        sup_ratio      = sup_hits / wc
        generic_hits   = sum(1 for p in self.GENERIC_PHRASES if p in tl)
        family_hits    = sum(1 for f in self.FAMILY_REFS if f in tl)
        specific_hits  = len(self.SPECIFIC_UNITS.findall(text))

        word_freq = {}
        for w in words:
            if w not in STOPWORDS and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1
        max_repeat    = max(word_freq.values(), default=0)
        repeated_words = sum(1 for v in word_freq.values() if v > 1)

        avg_word_len   = np.mean([len(w) for w in words]) if words else 0
        sent_count     = max(len(re.split(r'[.!?]+', text)), 1)
        avg_sent_len   = wc / sent_count

        fp_ratio       = sum(1 for w in words if w in ('i','me','my','myself','mine')) / wc
        multi_excl     = len(re.findall(r'!{2,}', text))
        alpha_chars    = [c for c in text if c.isalpha()]
        char_caps      = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)
        quest_count    = text.count('?')
        ai_style_hits  = sum(1 for p in self.AI_STYLE_PHRASES if p in tl)
        comma_ratio    = text.count(',') / wc
        contrast_hits  = sum(1 for w in words if w in ('but','however','though','although'))
        digit_ratio    = sum(1 for c in text if c.isdigit()) / chars

        return [
            excl_ratio,    caps_ratio,     sup_ratio,      generic_hits,
            family_hits,   specific_hits,  max_repeat,     repeated_words,
            wc,            avg_word_len,   avg_sent_len,   fp_ratio,
            multi_excl,    quest_count,    char_caps,      excl_count,
            sup_hits,      sent_count,
            ai_style_hits, comma_ratio,     contrast_hits,  digit_ratio,
        ]


# ── Combined Feature Builder ──────────────────────────────────────────────────
class CombinedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.word_tfidf = TfidfVectorizer(
            max_features=3000, ngram_range=(1, 3),
            sublinear_tf=True, min_df=1, analyzer='word')
        self.char_tfidf = TfidfVectorizer(
            max_features=2000, ngram_range=(2, 4),
            sublinear_tf=True, min_df=1, analyzer='char_wb')
        self.heuristic  = HeuristicFeatureExtractor()
        self.scaler     = StandardScaler()

    def fit(self, X, y=None):
        clean = [preprocess(t) for t in X]
        self.word_tfidf.fit(clean)
        self.char_tfidf.fit(clean)
        h = self.heuristic.fit_transform(X)
        self.scaler.fit(h)
        return self

    def transform(self, X):
        clean   = [preprocess(t) for t in X]
        w       = self.word_tfidf.transform(clean)
        c       = self.char_tfidf.transform(clean)
        h       = self.heuristic.transform(X)
        h_sc    = sp.csr_matrix(self.scaler.transform(h))
        return sp.hstack([w, c, h_sc])


# ── Dataset Loader ────────────────────────────────────────────────────────────
def load_dataset():
    path = Path(__file__).parent / 'dataset.csv'
    df   = pd.read_csv(path).dropna(subset=['review', 'label'])
    df['review'] = df['review'].astype(str).str.strip()
    df   = df[df['review'].str.len() > 5]
    texts = df['review'].tolist()
    labels = df['label'].astype(int).tolist()
    aug_texts, aug_labels = training_augmentation()
    return texts + aug_texts, labels + aug_labels


def training_augmentation():
    """Small hard-example set to improve behavior outside the tiny CSV."""
    genuine = [
        "The zipper catches near the top if I pull too fast, but the stitching stayed tight after two weeks.",
        "Battery lasted 5 hours and 40 minutes with screen brightness around 60 percent.",
        "The medium fits my shoulders well, though the sleeves are about one inch longer than expected.",
        "Setup took 18 minutes. The QR code failed once, then paired after I restarted the app.",
        "Sound is clear for calls, but bass is weak compared with my older wired headphones.",
        "The pan heats evenly on the center burner. The handle gets warm after about 12 minutes.",
        "Packaging had a dent on the corner, but the product inside was not scratched.",
        "It works with my laptop USB-C port, but the cable is stiff and only 3 feet long.",
        "Color is slightly darker than the photo. Fabric feels durable after the first wash.",
        "Instructions skipped one screw size, so assembly took about 35 minutes instead of 20.",
        "The lid seals well in my bag, although coffee stays hot closer to 3 hours than 6.",
        "Print quality is fine for labels. It jammed twice when I used thinner paper.",
        "The texture is smooth but a little slippery when wet, so I use it only on the counter.",
        "It arrived two days late. The product itself works, but the box had a torn side flap.",
        "The light has three brightness levels. The middle setting is enough for reading at night.",
        "After a week of use, the left button feels slightly loose but still registers every click.",
    ]
    fake = [
        "Overall, this product exceeded my expectations in every possible way and I highly recommend it to anyone looking for quality.",
        "From the moment I opened the box, I knew this was a top notch purchase that would change my daily routine.",
        "Whether you are a beginner or an expert, this is the perfect addition and you should not hesitate to buy it.",
        "This item is not only beautifully designed but also incredibly useful, making it a must have for everyone.",
        "I was pleasantly surprised by the premium quality and seamless experience. Five stars without question.",
        "Look no further, this is a solid choice for anyone looking for value, performance, and reliability.",
        "The product stands out because it is user friendly, highly versatile, and worth every penny.",
        "Amazing quality and fast shipping. It works perfectly and I will definitely buy again.",
        "Best product ever made. My whole family loves it and everyone I know should buy one.",
        "Absolutely flawless experience from start to finish. This purchase exceeded all expectations.",
        "In conclusion, this product delivers outstanding results and is ideal for anyone who wants excellence.",
        "Perfect in every way, great quality, great price, and exactly what everyone needs right now.",
        "Perfect in every way and exactly what everyone needs. The quality and value are simply excellent.",
        "Great quality, great price, and a perfect addition for any home. I recommend it to everyone.",
        "This is exactly what everyone should buy because it offers amazing quality and unbeatable value.",
        "Everything about this product feels perfect, useful, and worth recommending to all my friends.",
    ]
    return genuine + fake, [0] * len(genuine) + [1] * len(fake)


# ── Main Detector Class ───────────────────────────────────────────────────────
class FakeReviewDetector:

    def __init__(self):
        self.features   = CombinedFeatures()
        self.model      = None
        self.trained    = False
        self.cv_results = {}

    # ── build ensemble ──
    def _build_ensemble(self):
        lr  = LogisticRegression(C=2.0, max_iter=1000, solver='lbfgs',
                                  class_weight='balanced', random_state=42)
        rf  = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                      random_state=42, n_jobs=-1)
        gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                          max_depth=4, subsample=0.8, random_state=42)
        svm = CalibratedClassifierCV(
                LinearSVC(C=1.0, max_iter=2000, class_weight='balanced',
                          random_state=42), cv=3)
        return VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft', weights=[2, 2, 2, 1])

    def _build_pipeline(self, clf):
        return Pipeline([
            ('features', CombinedFeatures()),
            ('model', clf),
        ])

    # ── train ──
    def train(self):
        print("📦 Loading dataset...")
        texts, labels = load_dataset()
        print(f"   {len(texts)} samples | {sum(labels)} fake | {len(labels)-sum(labels)} genuine")

        print("🔧 Fitting feature pipeline...")
        y = np.array(labels)
        self.features.fit(texts)
        X = self.features.transform(texts)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        print("\n📊 10-Fold Stratified Cross-Validation:")

        individual = {
            'Logistic Regression': LogisticRegression(
                C=2.0, max_iter=1000, class_weight='balanced', random_state=42),
            'Random Forest':       RandomForestClassifier(
                n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
            'Gradient Boosting':   GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42),
            'Linear SVM':          CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight='balanced', random_state=42), cv=3),
        }

        self.cv_results = {}
        for name, clf in individual.items():
            pipe = self._build_pipeline(clf)
            acc = cross_val_score(pipe, texts, y, cv=cv, scoring='accuracy', n_jobs=-1)
            auc = cross_val_score(pipe, texts, y, cv=cv, scoring='roc_auc',  n_jobs=-1)
            self.cv_results[name] = {
                'accuracy':     round(float(acc.mean()) * 100, 2),
                'accuracy_std': round(float(acc.std())  * 100, 2),
                'auc':          round(float(auc.mean()) * 100, 2),
            }
            print(f"   {name:<25} Acc={acc.mean():.4f} ± {acc.std():.4f}   AUC={auc.mean():.4f}")

        # Ensemble CV
        ens      = self._build_pipeline(self._build_ensemble())
        ens_acc  = cross_val_score(ens, texts, y, cv=cv, scoring='accuracy', n_jobs=-1)
        ens_auc  = cross_val_score(ens, texts, y, cv=cv, scoring='roc_auc',  n_jobs=-1)
        self.cv_results['Ensemble'] = {
            'accuracy':     round(float(ens_acc.mean()) * 100, 2),
            'accuracy_std': round(float(ens_acc.std())  * 100, 2),
            'auc':          round(float(ens_auc.mean()) * 100, 2),
        }
        print(f"   {'Ensemble (Final)':<25} Acc={ens_acc.mean():.4f} ± {ens_acc.std():.4f}   AUC={ens_auc.mean():.4f}")

        # Train final model on full data
        print("\n🤝 Training final ensemble on full dataset...")
        self.model = self._build_ensemble()
        self.model.fit(X, y)

        # Hold-out report
        text_tr, text_te, y_tr, y_te = train_test_split(
            texts, y, test_size=0.2, stratify=y, random_state=42)
        ho = self._build_pipeline(self._build_ensemble())
        ho.fit(text_tr, y_tr)
        y_pred = ho.predict(text_te)
        y_prob = ho.predict_proba(text_te)[:, 1]
        print(f"\n📋 Hold-out Test (20%)  Acc={accuracy_score(y_te,y_pred):.4f}  AUC={roc_auc_score(y_te,y_prob):.4f}")
        report = classification_report(y_te, y_pred, target_names=['Genuine','Fake'])
        print('\n'.join('   ' + l for l in report.splitlines()))

        self.trained = True
        print("✅ Model ready!\n")
        return self.cv_results

    # ── single predict ──
    def predict(self, text: str) -> dict:
        if not self.trained:
            self.train()
        X     = self.features.transform([text])
        proba = self.model.predict_proba(X)[0]
        fp    = float(proba[1])
        h     = HeuristicFeatureExtractor()._extract(text)
        ai_adjustment = self._ai_fake_adjustment(h, text)
        if ai_adjustment >= 7:
            fp = max(fp, 0.82)
        elif ai_adjustment >= 5:
            fp = max(fp, 0.66)
        elif ai_adjustment >= 3:
            fp = max(fp, 0.48)
        human_adjustment = self._human_informal_adjustment(h, text)
        if human_adjustment >= 6 and ai_adjustment < 5:
            fp = min(fp, 0.38)
        elif human_adjustment >= 4 and ai_adjustment < 5:
            fp = min(fp, 0.44)

        if fp >= 0.65:   verdict, risk = "FAKE",       "High Risk"
        elif fp >= 0.45: verdict, risk = "SUSPICIOUS", "Medium Risk"
        else:            verdict, risk = "GENUINE",    "Low Risk"

        return {
            "verdict":           verdict,
            "risk":              risk,
            "fake_score":        round(fp * 100, 1),
            "genuine_score":     round((1 - fp) * 100, 1),
            "signals":           self._signals(h, text),
            "word_count":        int(h[8]),
            "exclamation_count": int(h[15]),
            "specificity_score": int(h[5]),
            "model_accuracy":    self.cv_results.get('Ensemble', {}).get('accuracy', 0),
        }

    def _ai_fake_adjustment(self, h, text):
        tl = text.lower()
        words = [w.strip('.,!?;:()[]{}"\'') for w in tl.split()]
        formal_markers = {
            'overall','moreover','furthermore','additionally','therefore',
            'seamless','versatile','reliable','premium','exceptional',
            'impressive','thoughtful','satisfying','enhances','delivers',
            'excellent','durable','efficient','convenient','comfortable',
            'sleek','modern','stylish','innovative','practical','valuable',
            'performance','quality','experience','design','recommend'
        }
        generic_product_terms = {
            'product','item','purchase','option','choice','solution',
            'quality','performance','design','experience','value',
            'service','team','staff','support','features','options',
            'improvement','improvements','expectations','glitches',
            'issues','delay','delays'
        }
        positive_terms = {
            'good','great','excellent','amazing','fantastic','perfect',
            'impressive','reliable','durable','comfortable','easy',
            'useful','valuable','premium','smooth','best','nice'
        }
        templated_review_terms = {
            'experience','service','quality','support','staff','team',
            'helpful','polite','professional','friendly','smooth',
            'organized','efficient','user-friendly','reliable','average',
            'decent','okay','improvement','improvements','expected',
            'expectations','satisfied','disappointing','recommend',
            'features','options','glitches','issues','delays',
            'smoother','faster','better','later'
        }
        negative_or_specific_terms = {
            'but','however','though','although','scratch','late','broken',
            'loose','small','large','narrow','heavy','warm','hot','slow',
            'noisy','jammed','dent','returned','inch','feet','hours',
            'minutes','battery','screen','cable','box','adapter','zipper'
        }
        marker_hits = sum(1 for w in words if w in formal_markers)
        generic_hits = sum(1 for w in words if w in generic_product_terms)
        positive_hits = sum(1 for w in words if w in positive_terms)
        template_hits = sum(1 for w in words if w in templated_review_terms)
        natural_detail_hits = sum(1 for w in words if w in negative_or_specific_terms)
        pronoun_hits = sum(1 for w in words if w in ('i','me','my','we','our'))
        no_specific_detail = h[5] == 0 and h[21] == 0
        marketing_shape = (
            h[8] >= 18 and h[10] >= 9 and h[14] < 0.08 and h[12] == 0
        )
        polished_positive = (
            h[16] >= 1 and h[5] == 0 and h[21] == 0 and h[0] < 0.01
        )
        generic_positive_review = (
            h[8] >= 14 and generic_hits >= 2 and positive_hits >= 2 and h[0] < 0.01
        )
        impersonal_review = h[8] >= 16 and pronoun_hits <= 1 and h[5] == 0
        polished_with_no_flaws = (
            h[8] >= 18 and positive_hits >= 2 and natural_detail_hits == 0
        )
        templated_no_detail = (
            h[8] >= 7 and no_specific_detail and template_hits >= 2
        )
        short_generic_template = (
            h[8] <= 18 and no_specific_detail and template_hits >= 1 and generic_hits >= 1
        )
        very_short_no_detail = h[8] >= 5 and h[8] <= 12 and no_specific_detail
        score = 0
        score += int(h[18]) * 3
        score += min(int(h[3]), 3) * 2
        score += min(marker_hits, 4)
        score += min(generic_hits, 3)
        score += min(positive_hits, 3)
        score += min(template_hits, 4)
        score += 2 if marketing_shape else 0
        score += 2 if polished_positive else 0
        score += 2 if generic_positive_review else 0
        score += 2 if impersonal_review else 0
        score += 2 if polished_with_no_flaws else 0
        score += 5 if templated_no_detail else 0
        score += 4 if short_generic_template else 0
        score += 4 if very_short_no_detail and (template_hits >= 1 or h[18] >= 1) else 0
        score += 1 if 'recommend' in tl else 0
        score += 1 if 'quality' in tl or 'value' in tl else 0
        return score

    def _human_informal_adjustment(self, h, text):
        tl = text.lower()
        words = [w.strip('.,!?;:()[]{}"\'') for w in tl.split()]
        informal_markers = {
            'bro','ya','yeah','ok','okay','thanks','thank','pls','please',
            'kinda','sorta','much','neat','clean'
        }
        personal_experience_terms = {
            'i','me','my','we','our','got','liked','bought','used',
            'received','delivery','delivered','package','parcel','order'
        }
        informal_hits = sum(1 for w in words if w in informal_markers)
        personal_hits = sum(1 for w in words if w in personal_experience_terms)
        lowercase_i = len(re.findall(r'(?<![A-Za-z])i(?![A-Za-z])', text))
        repeated_plain_words = h[7] >= 1 and h[6] <= 2
        simple_sentence = h[8] >= 8 and h[18] == 0 and h[3] == 0

        score = 0
        score += min(informal_hits, 3) * 2
        score += min(personal_hits, 4)
        score += 2 if lowercase_i >= 1 else 0
        score += 1 if repeated_plain_words else 0
        score += 1 if simple_sentence else 0
        score += 1 if 'thank you' in tl else 0
        return score

    def _signals(self, h, text):
        s = []
        if h[0]  > 0.015: s.append({"flag": "Excessive exclamation marks",                  "weight": "high"})
        if h[1]  > 0.05:  s.append({"flag": "ALL CAPS words detected",                       "weight": "high"})
        if h[2]  > 0.12:  s.append({"flag": "Heavy use of superlatives & extreme words",     "weight": "high"})
        if h[3]  >= 2:    s.append({"flag": "Multiple generic marketing phrases detected",    "weight": "medium"})
        if h[4]  >= 1:    s.append({"flag": "Family/gift references (common in fake reviews)","weight": "medium"})
        if h[5]  == 0:    s.append({"flag": "No specific product details (vague language)",   "weight": "low"})
        if h[6]  >= 3:    s.append({"flag": "Repetitive positive vocabulary",                 "weight": "medium"})
        if h[11] > 0.15:  s.append({"flag": "Overuse of first-person pronouns",              "weight": "low"})
        if h[12] >= 2:    s.append({"flag": "Multiple consecutive exclamation marks",         "weight": "high"})
        if h[14] > 0.15:  s.append({"flag": "High proportion of capital letters",             "weight": "medium"})
        if h[8]  < 12:    s.append({"flag": "Review is unusually short",                      "weight": "medium"})
        if h[10] < 6:     s.append({"flag": "Very short sentences — lack of detail",          "weight": "low"})
        if h[18] >= 2:    s.append({"flag": "AI-like polished/generic phrasing detected",      "weight": "medium"})
        if self._ai_fake_adjustment(h, text) >= 5:
                          s.append({"flag": "AI-generated review pattern detected",            "weight": "high"})
        if h[21] == 0 and h[5] == 0:
                          s.append({"flag": "No measurable details or numbers",               "weight": "low"})
        return s

    # ── batch ──
    def analyze_batch(self, reviews: list) -> dict:
        if not self.trained:
            self.train()
        valid = [(i, r) for i, r in enumerate(reviews) if str(r).strip()]
        if not valid:
            return {"error": "No valid reviews found"}
        results = []
        for i, review in valid:
            r = self.predict(review)
            r["review_text"] = review[:130] + ("…" if len(review) > 130 else "")
            r["index"] = i + 1
            results.append(r)
        scores    = [r["fake_score"] for r in results]
        avg_fake  = float(np.mean(scores))
        return {
            "total":            len(results),
            "fake_count":       sum(1 for r in results if r["verdict"] == "FAKE"),
            "suspicious_count": sum(1 for r in results if r["verdict"] == "SUSPICIOUS"),
            "genuine_count":    sum(1 for r in results if r["verdict"] == "GENUINE"),
            "avg_fake_score":   round(avg_fake, 1),
            "trust_rating":     round(100 - avg_fake, 1),
            "results":          results,
            "cv_results":       self.cv_results,
        }

    def get_model_stats(self):
        if not self.trained:
            self.train()
        return self.cv_results


# ── Singleton ─────────────────────────────────────────────────────────────────
print("🚀 Initializing FakeReview Buster ML Engine v2...")
detector = FakeReviewDetector()
