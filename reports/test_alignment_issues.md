# Frozen-test alignment issues (flagged for review)

Status: **documentation only**. `data/test_real.tsv` and `data/parallel_real.tsv` are **not modified** (frozen benchmark / corpus integrity). Any correction requires explicit confirmation and a careful rebuild/re-eval because it affects comparability.

Evidence: `reports/preds_test_best_e27.jsonl`, `reports/test_error_analysis_e27.md`, and source scans of `data/parallel_real.tsv` / wikisource texts.

## 1) Test item #49 — likely off-by-one grabar shift (Exod 40 context)

- **Location:** `data/test_real.tsv:49` = `data/parallel_real.tsv:2725`
- **Current pair:**
  - MOD: «Մովսէսը չէր կարողանում մտնել վկայութեան խորան, որովհետեւ ամպը ծածկել էր այն, եւ խորանը լցուած էր Տիրոջ փառքով:»
  - GRB (gold): «Յորժամ վերանայր ամպն ի խորանէ անտի, չուէին որդիքն Իսրայէլի հանդերձ ամենայն աղխիւ իւրեանց:»
- **Observation:** MOD expresses Exod 40:35-like content (Moses could not enter because the cloud/glory filled the tabernacle). Gold GRB expresses Exod 40:36-like content (when the cloud lifted, Israel journeyed) — different verse.
- **Model (E27 best decode):** «Եւ ոչ կարէր Մովսէս մտանել ի խորանն վկայութեան. քանզի ծածկէր զնա ամպն. եւ լցաւ խորանն փառօք Տեառն:» — tracks the MOD/source meaning.
- **Context/evidence (`data/parallel_real.tsv` 2724–2726 and `data/wikisource/exodus.grabar.txt` Exod 40:34–36):** grabar column appears shifted in this block:
  - 2724 MOD «Ամպը ծածկեց վկայութեան խորանը, եւ խորանը լցուեց Տիրոջ փառքով:» (40:34) ↔ current GRB «Եւ ո՛չ կարէր Մովսէս մտանել ի խորանն վկայութեան, զի ծածկեաց զնա ամպն, եւ լի՛ եղեւ խորանն փառօք Տեառն:» (Zohrab 40:35 sense)
  - 2725 MOD «Մովսէսը չէր կարողանում մտնել վկայութեան խորան…» (40:35) ↔ current GRB «Յորժամ վերանայր ամպն ի խորանէ անտի, չուէին որդիքն Իսրայէլի…» (Zohrab 40:36 sense) — **test item**
  - 2726 MOD «Երբ ամպը վերանում էր…» (40:36) ↔ current GRB «Եւ եթէ ո՛չ վերանայր ամպն՝ ո՛չ չուէին…» (Zohrab 40:37 sense)
- **Likely correction for #49 (review only):** pair 2725 MOD with 2724’s current GRB («Եւ ո՛չ կարէր Մովսէս մտանել ի խորանն վկայութեան, զի ծածկեաց զնա ամպն, եւ լի՛ եղեւ խորանն փառօք Տեառն:»), and realign the surrounding block consistently against verse boundaries.
- **Suggested action (if approved):** reconstruct correct MOD↔GRB alignment for this Exod 40 block, update `parallel_real.tsv`, regenerate splits with explicit handling/versioning of the frozen test. Do **not** change silently.

## 2) Test item #107 — likely block misalignment (Exod 28 breastpiece rows)

- **Location:** `data/test_real.tsv:107` = `data/parallel_real.tsv:2298`
- **Current pair:**
  - MOD: «Դրա վրայ ակնաքարերը կը շարես չորս շարք ընդելուզուածքով: Ակնաքարերի շարքերը այսպէս թող լինեն. առաջինը՝ զմրուխտ, սարդիոն եւ տպազիոն՝ մէկ շարք:» (first row)
  - GRB (gold): «եւ Կարգն երկրորդ՝ կարկեհան, եւ շափիղա, եւ յասպիս:» (second row)
- **Observation:** MOD describes the first row of stones; gold is the second row.
- **Model (E27 best decode):** «Եւ դիցես ի վերայ նորա ականսն ի չորս շարս ընդելուզեալս. ա՛յսպէս լինիցին շարք ականցն. զառաջինն զմրուխտ, եւ սարդիոն, եւ տպազիոն, շար մի:» — follows source/first row.
- **Context/evidence (`data/parallel_real.tsv` 2297–2300 and `data/wikisource/exodus.grabar.txt` Exod 28:17–20):** grabar appears shifted relative to modern in this block:
  - 2297 GRB: «Եւ ընդելուզցե՛ս ի նմա ընդելուզածս, ըստ չորեքկարգեա՛ն ականց: Կարգս ականց, Առաջին՝ զըմրուխտ, եւ սարդիոն, եւ տպազիոն. կարգ մի:» (first row, 28:17) — matches 2298 MOD’s content
  - 2298 GRB: «եւ Կարգն երկրորդ՝ կարկեհան, եւ շափիղա, եւ յասպիս:» (28:18) — matches 2299 MOD (“Երկրորդ շարքը…”)
  - 2299 GRB: third row (28:19) — matches 2300 MOD, etc.
- **Likely correction for #107 (review only):** pair 2298 MOD with 2297’s current GRB (first row) and realign the surrounding block consistently against verse boundaries.
- **Suggested action (if approved):** re-align this stone-rows block against Zohrab Exod 28:17–20 (and corresponding modern verse segmentation), update `parallel_real.tsv`, regenerate splits with explicit handling/versioning of the frozen test. Do **not** change silently.

## Notes
- A systematic chargram-overlap scan of all 108 test predictions flagged only #49 and #107 as strong misalignment outliers (very low pred–ref, much higher src–pred).
- Fixing would improve test validity and may change measured chrF/BLEU; therefore it must be treated as a benchmark change, not a silent correction.
- Until approved, evaluation/comparison remains on the existing frozen `data/test_real.tsv` as-is.