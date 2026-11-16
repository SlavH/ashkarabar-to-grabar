# Proposed correction: test items #49 and #107 (review-only, not applied)

**Status:** proposal only. No changes applied to `data/test_real.tsv`, `data/parallel_real.tsv`, or splits. Requires explicit approval and benchmark versioning/re-eval because it affects comparability.

## #49 — Exod 40 block shift (test line 49 = `parallel_real.tsv:2725`)

Current (shifted) grabar in block `parallel_real.tsv:2723-2727`:

| line | MOD (ash) | GRB (current, suspected shifted) |
|---:|---|---|
| 2723 | Բակ կառուցեց... Այսպիսով Մովսէսն աւարտեց իր ամբողջ գործը: | Եւ ծածկեաց ամպն զխորանն վկայութեան. եւ լի՛ եղեւ փառօք Տեառն խորանն: (≈40:34) |
| 2724 | Ամպը ծածկեց վկայութեան խորանը, եւ խորանը լցուեց Տիրոջ փառքով: (40:34) | Եւ ո՛չ կարէր Մովսէս մտանել ի խորանն վկայութեան, զի ծածկեաց զնա ամպն, եւ լի՛ եղեւ խորանն փառօք Տեառն: (≈40:35) |
| **2725 (test #49)** | Մովսէսը չէր կարողանում մտնել վկայութեան խորան... (40:35) | Յորժամ վերանայր ամպն ի խորանէ անտի, չուէին որդիքն Իսրայէլի... (≈40:36) |
| 2726 | Երբ ամպը վերանում էր... (40:36) | Եւ եթէ ո՛չ վերանայր ամպն՝ ո՛չ չուէին... (≈40:37) |
| 2727 | իսկ երբ ամպը չէր վերանում... (40:37) | Զի ամպն Տեառն է՛ր ի վերայ խորանին... (≈40:38) |

**Evidence:** model prediction for #49 tracks MOD 40:35 («Եւ ոչ կարէր Մովսէս մտանել ի խորանն վկայութեան... լի՛ փառօք Տեառն»), matching 2724’s current GRB, while gold is 40:36-like. `data/wikisource/exodus.grabar.txt` Exod 40:34–38 confirms verse contents.

**Minimal proposed fix (block realignment):** shift GRB up by one within 2723–2727 so each MOD aligns to its verse GRB:
- 2723 GRB ← 2724’s current GRB (40:35) **or** verify 2723 should be 40:34/finish — needs verse-level check against wikisource/Zohrab segmentation for 2723 MOD.
- 2724 GRB ← 2723’s current GRB? no — better: align by verse: 2723 MOD (end 40:33) should take appropriate GRB; 2724 MOD (40:34) ← 2723’s current GRB; 2725 MOD (40:35) ← 2724’s current GRB; 2726 MOD (40:36) ← 2725’s current GRB; 2727 MOD (40:37) ← 2726’s current GRB; 2727’s current GRB (≈40:38) becomes orphan/needs matching MOD (2728?) or review.
- **Action required on approval:** perform careful verse-by-verse re-alignment of the Exod 40 tail block (2723 onward, incl. 2728 if affected) against `data/wikisource/exodus.grabar.txt` and modern text, then propagate to `parallel_real.tsv`, `test_real.tsv` (row #49), and rebuild affected splits with explicit versioning.

**Directly evidenced correction for test #49:** set test/parallel 2725 GRB to 2724’s current GRB:  
`Եւ ո՛չ կարէր Մովսէս մտանել ի խորանն վկայութեան, զի ծածկեաց զնա ամպն, եւ լի՛ եղեւ խորանն փառօք Տեառն:`

## #107 — Exod 28 stone-rows block shift (test line 107 = `parallel_real.tsv:2298`)

Current (shifted) grabar in block `parallel_real.tsv:2297-2301`:

| line | MOD (ash) | GRB (current, suspected shifted) |
|---:|---|---|
| 2297 | Քառակուսի կը պատրաստես... (28:17 intro) | Եւ ընդելուզցե՛ս ի նմա... Կարգս ականց, Առաջին՝ ... կարգ մի: (includes 28:17/1st row) |
| **2298 (test #107)** | Դրա վրայ... առաջինը՝ զմրուխտ, սարդիոն եւ տպազիոն՝ մէկ շարք: (1st row) | եւ Կարգն երկրորդ՝ կարկեհան, եւ շափիղա, եւ յասպիս: (2nd row, 28:18) |
| 2299 | Երկրորդ շարքը՝ նռնաքար, շափիւղայ եւ յասպիս: (2nd row) | եւ Կարգն երրորդ՝ գոճազմ, եւ ակատ, եւ սուտակ: (3rd row, 28:19) |
| 2300 | Երրորդ շարքը՝ գոճազմ, ագատ եւ սուտակ: (3rd row) | եւ Կարգն չորրորդ՝ յակինթ, եւ բիւրեղ, եւ եղունգն... (4th row, 28:20) |
| 2301 | Չորրորդ շարքը՝ ... (4th row) | Եւ եղիցին ականքն յանուա՛նց որդւոցն Իսրայէլի... (28:21) |

**Evidence:** model prediction for #107 renders the 1st row (matches 2297’s current GRB tail / MOD 2298 content), while gold is 2nd row. `data/wikisource/exodus.grabar.txt` Exod 28:17–21 confirms.

**Proposed fix (block realignment):** shift GRB up so MOD rows match:
- 2298 GRB ← 2297’s current GRB (or the 1st-row portion aligned to 2298 MOD): `Եւ ընդելուզցե՛ս ի նմա ընդելուզածս, ըստ չորեքկարգեա՛ն ականց: Կարգս ականց, Առաջին՝ զըմրուխտ, եւ սարդիոն, եւ տպազիոն. կարգ մի:` (note: this contains more than 1st row; verse/segment split vs 2297 MOD must be reviewed — 2297 MOD is the “square/1 span” intro, 2297 GRB includes intro+1st row in one cell).
- 2299 GRB ← 2298’s current GRB (2nd row)
- 2300 GRB ← 2299’s current GRB (3rd row)
- 2301 GRB ← 2300’s current GRB (4th row)
- 2301’s current GRB (28:21) moves down to 2302? 2302 MOD is 28:21-like (“twelve stones with names…”), 2302 GRB currently is 28:22-like — suggesting the shift may extend; requires block review through 2302+.

**Directly evidenced correction for test #107:** set test/parallel 2298 GRB to the first-row GRB. The best aligned single-cell candidate from context is 2297’s current GRB, but 2297 MOD/GRB segmentation may need adjustment (2297 MOD lacks the 1st-row list that its GRB contains). **Action required on approval:** re-segment/realign 2297–2302 against Exod 28:17–22 so each MOD/GRB pair is verse-consistent, then propagate and version.

## Impact if approved
- Changes to `data/parallel_real.tsv` (training corpus rows in affected blocks) and `data/test_real.tsv` (rows 49, 107) → **new benchmark version** (e.g. `test_real_v2.tsv` or documented replacement) and rebuilt splits for any retraining; prior chrF numbers not directly comparable.
- Expected: improved test validity; measured chrF/BLEU will change (direction likely up for corrected items, magnitude TBD).
- On approval, implement with a scripted, auditable transform, regenerate splits, re-evaluate with current best adapter/decode on Kaggle, and report before/after deltas.
