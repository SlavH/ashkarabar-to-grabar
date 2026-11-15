# Test Error Analysis (E27, current best decode)

- **Model/adapter:** v9 `checkpoints/nllb_lora_grabar/` (unchanged)
- **Decode:** `num_beams=4, length_penalty=1.07, no_repeat_ngram_size=0, repetition_penalty=1.22, min_new_tokens=20, max_new_tokens=192`
- **Frozen test:** `data/test_real.tsv`, n=108 (ashkharabar → grabar)
- **Scores:** **chrF 55.41 / BLEU 25.58** (confirmed, E27)
- **Predictions:** `reports/preds_test_best_e27.jsonl`
- **Per-example auto-tags:** `reports/test_error_tags_e27.tsv` (heuristic; not expert gold labels)

## Heuristic tag distribution (per example, multi-label)

- `ok/close`: 55
- `low_lexical_overlap`: 31
- `end_punct_diff`: 18
- `short/possible_omission`: 14
- `long/possible_addition`: 8
- `repetition`: 6

Notes: `low_lexical_overlap` = <45% of gold word-types present in prediction (crude; penalizes valid paraphrase/morphology/orthography variance). `short/long` based on word-count ratio vs gold (<0.8 / >1.25). Tags are automatic heuristics to guide review, not definitive error labels.

## Observations
- A majority of outputs are in-range (`ok/close` by these heuristics: 55/108), often preserving core meaning and biblical/grabar register.
- **Lexical/register & form differences** remain common (low lexical overlap by surface types in 31 cases): classical vs alternative forms, morphology/case endings, function words/particles, and spelling/orthographic variants (e.g. եւ/և, punctuation, breathing) vs the specific gold reference.
- **Length/completeness drift:** 14 shorter and 8 longer than gold by word count (possible omission/compression or expansion/paraphrase). Some low-overlap cases reflect substantial divergence or alignment to a different phrasing than the reference.
- **Repetition** flagged in 6 cases under current decode (`no_repeat_ngram=0`); generally limited but present.
- **End punctuation** differs in 18 cases (minor, but affects exact-match/metrics slightly).

## Lowest reference-lexical-coverage examples (review candidates)
These are not necessarily “worst” translations, but show largest surface divergence from gold and are useful for manual inspection. Heuristic tags/coverage are approximate.

### #49 (ref_lex_cov=0.154, len_ratio=1.308, tags=long/possible_addition;low_lexical_overlap)
- **SRC:** Մովսէսը չէր կարողանում մտնել վկայութեան խորան, որովհետեւ ամպը ծածկել էր այն, եւ խորանը լցուած էր Տիրոջ փառքով:
- **PRED:** Եւ ոչ կարէր Մովսէս մտանել ի խորանն վկայութեան. քանզի ծածկէր զնա ամպն. եւ լցաւ խորանն փառօք Տեառն:
- **REF:** Յորժամ վերանայր ամպն ի խորանէ անտի, չուէին որդիքն Իսրայէլի հանդերձ ամենայն աղխիւ իւրեանց:
- **Note:** prediction closely follows the source meaning (Moses unable to enter the tabernacle because of the cloud/glory), while gold is different content — suggests a possible **source–gold misalignment** in this test item.

### #107 (ref_lex_cov=0.167, len_ratio=2.75, tags=long/possible_addition;low_lexical_overlap)
- **SRC:** Դրա վրայ ակնաքարերը կը շարես չորս շարք ընդելուզուածքով: Ակնաքարերի շարքերը այսպէս թող լինեն. առաջինը՝ զմրուխտ, սարդիոն եւ տպազիոն՝ մէկ շարք:
- **PRED:** Եւ դիցես ի վերայ նորա ականսն ի չորս շարս ընդելուզեալս. ա՛յսպէս լինիցին շարք ականցն. զառաջինն զմրուխտ, եւ սարդիոն, եւ տպազիոն, շար մի:
- **REF:** եւ Կարգն երկրորդ՝ կարկեհան, եւ շափիղա, եւ յասպիս:
- **Note:** prediction renders the source (first row of stones) reasonably; gold gives the second row — likely **source–gold continuation/misalignment**.

### #98 (ref_lex_cov=0.188, len_ratio=1.062, tags=low_lexical_overlap)
- **SRC:** իսկ միւս զաւակները, որոնց ծնունդ կը տաս սրանցից յետոյ, թող քոնը լինեն: Նրանք իրենց ժառանգութիւնն ստանալու համար թող կոչուեն իրենց եղբայրների անուամբ:
- **PRED:** Եւ զաւակ որ ծնցես զկնի նոցա՝ քո՛յ եղիցի. անուա՛ւն եղբարց իւրեանց կոչեսցի՛ն նոցա, զի ստացուցանիցեն զանուն նոցա։
- **REF:** Եւ ա՛յլ ծնունդք զոր ծնանիցիս յետ այսորիկ՝ քեզ եղիցին. յանուն եղբարց իւրեանց կոչեսցին ի նոցա՛ ժառանգութիւնս։
- **Note:** meaning largely preserved with different lexical/morphological and clause choices vs gold.

### #6 (ref_lex_cov=0.2, len_ratio=0.636, tags=short/possible_omission;low_lexical_overlap)
- **SRC:** Երկու շղթաների երկու ծայրերը կը մտցնես երկու օղակների մէջ եւ վակասի առջեւի կողմից տանելով՝ կ՚անցկացնես վակասի ետեւի կողմը:
- **PRED:** Եւ դիցես յերկոսին օղամանեակսն զերկոսին ծայրս երկոցունց շղթայացն. եւ անցցե՛ս զնոսա յերկրորդ կողմանէ վակասին:
- **REF:** Եւ արասցես երկո՛ւս օղս ոսկիս, եւ տացե՛ս զնոսա ի վերայ երկուց ծայրից տախտակին, առ եզերբն որ իցէ հանդէպ վակասին ի ներքոյ կողմանէ:
- **Note:** partial/shifted rendering; routing/detail relative to breastpiece differs from gold.

### #101 (ref_lex_cov=0.222, len_ratio=0.76, tags=short/possible_omission;end_punct_diff;low_lexical_overlap)
- **SRC:** Տապանն աստիճանաբար նեղացնելով կը շինես, այնպէս, որ նրա ծածկը մի կանգուն լինի: Տապանի դուռը կողքից կ՚անես. ներքնայարկ, միջնայարկ ու վերնայարկ թող ունենայ տապանը:
- **PRED:** Եւ արասցես զտապանն նեղութեամբ. եւ եղիցի ծածկութիւն նորա կանգնոյ. եւ զդուր տապանին արասցես կողմանէ. ներկէ՛ն, եւ միջնայակ, եւ վերնայակ:
- **REF:** Քաղելով գործեսցես զտապանն, եւ ի մի՛ կանգուն կատարեսցես զնա ի վերուստ. եւ զդուրս տապանին ի կողմանէ՛ արասցես։ Ներքնատունս եւ միջնատունս. եւ վերնատունս գործեսցես ի նմա։
- **Note:** key elements present but phrasing/lexicon differ and text is somewhat compressed.

### #14 (ref_lex_cov=0.231, len_ratio=0.5, tags=short/possible_omission;low_lexical_overlap)
- **SRC:** Վեցերորդ օրը թող պատրաստեն նախորդ իւրաքանչիւր օրուայ համար հաւաքածի կրկնապատիկը»:
- **PRED:** Եւ յաւուրն վեցերորդի արասցեն զկեանսն, յերկրորդում աւուրն յերկրորդում:
- **REF:** Եւ եղիցի յաւուրն վեցերորդի, պատրաստիցեն զոր ինչ բերիցենն. եւ եղիցի կրկին քան զայն, զոր օրըստօրէն ժողովիցեն:
- **Note:** meaning degraded/confused (“double portion” not rendered correctly) — a clear translation error.

### #85 (ref_lex_cov=0.25, len_ratio=0.765, tags=short/possible_omission;end_punct_diff;low_lexical_overlap)
- **SRC:** Փշրուեցին զօրութեամբ աղեղները նրանց,ու թուլացան ջլերը նրանց բազուկներիՅակոբի հզօր Աստծու ձեռքով,
- **PRED:** Փշռեցան աղեղք նոցա զօրութեամբ, եւ թուլացա՛ն ջուրք նոցա ի ձեռն Աստուծոյ հզօրոյն Յակոբայ։
- **REF:** եւ փշրեցան զօրութեամբ աղեղունք իւրեանց, եւ լուծան ջիլք բազկաց նոցա ձեռամբ Հզօրին Յակոբայ։ Անտի որ զօրացոյց զԻսրայէլ,
- **Note:** close in structure but contains lexical error (“ջուրք” vs sinews/arms) and omits trailing clause present in gold.

### #47 (ref_lex_cov=0.278, len_ratio=1.19, tags=end_punct_diff;low_lexical_overlap)
- **SRC:** Վերցնելով նրանց կերտած հորթը՝ Մովսէսը կրակը նետեց այն, ապա մանրեց, փոշու վերածեց, փոշին խառնեց ջրի մէջ ու տուեց ժողովրդին, որ խմի:
- **PRED:** Եւ առեալ Մովսիսի զհորթն զոր արարին նոցա՝ ընկէց զնա ի հուր, եւ մանրեաց զնա, եւ փոշեաց զնա. եւ խառնեաց զնա ի ջուր, եւ արբոյց ժողովրդեանն:
- **REF:** Եւ առեա՛լ զորթն զոր արարին՝ այրեա՛ց հրով. եւ խարտեաց զնա մա՛նր, եւ ցանեա՛ց ի վերայ ջրոցն եւ արբո՛յց որդւոցն Իսրայէլի: ՃԼԶ
- **Note:** strong semantic match; differences mainly lexical/verb choice and minor omission/detail vs gold.

## Suggested next focus
- Manual review of low-overlap/length-mismatch items in `reports/test_error_tags_e27.tsv` to distinguish true errors (omission, wrong meaning, wrong name/morphology) from acceptable grabar variants.
- **Possible test misalignments (flagged, not changed):** systematic chargram-overlap scan of all 108 items highlights **#49** and **#107** as outliers (very low pred–ref overlap but much higher src–pred overlap; predictions track source content while golds correspond to different verse/continuation). These appear to originate from `data/parallel_real.tsv` and are present in frozen `data/test_real.tsv`; correcting them would require corpus-level re-alignment/rebuild and explicit confirmation to alter the benchmark.
- Highest-leverage improvement remains **new verified, leak-checked parallel data** and/or modeling changes; decode around the current optimum is saturated (E23–E26).
