# Paper Extraction Prompt (Minimal)

## Usage

Open a new chat in any LLM interface that supports file attachments 
(Claude, ChatGPT, etc.) and attach the following three items:

```
1. This prompt text (or paste the body below into the message)
2. The KMDS schema JSON (latest public version, nullable variant 
   recommended)
   Current version: kmds_v15.2.4_nullable.json
3. The paper PDF to be extracted
```

After sending, manually rename the chat thread for organization. 
Use one thread per paper to avoid context contamination.

---

## Prompt body (send the text below to the model)

Extract the contents of the attached paper PDF into a 
structured JSON conforming to the attached JSON Schema. Record only 
values explicitly stated in the paper; use `null` for missing 
fields.

### Project intent & ground rules

This is part of the **Starrydata-KMDS** project: building a clean, 
structured data record from THIS paper for a materials database. Follow 
these ground rules so the output stays consistent and the run finishes 
promptly:

- **Use only the attached PDF**, specifically its body text, Methods, and 
  figure/table captions. Do not browse the web or consult external sources; 
  do not look up the DOI, the authors, or related work. Supplementary 
  material is out of scope unless it is attached.
- **Record only values explicitly written as numbers** in the text, tables, 
  captions, or axis labels — plus numeric constants in equations and 
  measurement/calculation conditions. **Do not digitize figures or read 
  values off plotted curves.** Leave every `digitization` field as `""`; for 
  `figure`/`graph`/`table` fill only `structure` and `description` briefly 
  (mention series values there in prose, not as data).
- **No inference.** No "typical" or assumed values; if the paper does not 
  state it, use `null`.
- **enum mismatch → `null`.** If a value has no matching enum member, set the 
  field to `null` and put the paper's original wording in the nearest 
  `notes`. Do not map to a "near" enum.
- **Record the open-access license** in `metadata.rights` (these are CC-BY 
  papers; almost always `CC BY`).
- **Traceability.** For each property / process / figure / table, note its 
  source location (Section / Figure / Table number) in the nearest `notes`; 
  include the DOI where helpful.
- **Stay within the schema.** `additionalProperties: false` is strict — never 
  invent keys. Output all `required` fields (use `null` when the paper is 
  silent), but do **not** exhaustively expand every optional field of this 
  large schema to `null`: include an optional field only when the paper 
  provides relevant information for it. Concepts the schema cannot express go 
  to the Schema extension candidates list (or `notes`), never as new keys.
- **Produce the two JSON files directly** — don't spend time exhaustively 
  re-validating the whole schema internally; one structurally-correct pass is 
  enough (long papers should still finish quickly).

### What counts as a material vs a sample

- A **sample** (`metadata.publication.samples[]`) is **one physical bulk 
  specimen that was prepared/synthesized AND measured as an original sample in 
  THIS paper**. One bulk specimen = one sample.
  - Do **not** create samples for specimens cited from other papers — only 
    this paper's original samples count.
  - Do **not** split one bulk into multiple samples for compositionally 
    distinct regions within its microstructure; describe those within the 
    single sample.
  - Do **not** treat purely computational/theoretical models (e.g. DFT 
    supercells) as samples.
- **`materials[]`** holds the distinct material system(s)/composition(s) those 
  original samples embody. Reference or standard pure substances used only for 
  calibration/comparison, literature-cited materials, and theory-only models 
  are **not** material entries — record them in the relevant measurement 
  `notes` or figure `description`.

### Populate structured conditions — do not leave objects empty

When you select a `process` type, a `measurement`, or a `property`, **fill its 
structured sub-fields** with what the paper states. Process and measurement 
*conditions* are usually in Methods as plain text — fully in scope, no 
digitization needed. Selecting the right key but leaving its object `{}` empty 
discards exactly the information the schema exists to capture.

- **Process steps** (`materials[].process[]`): each process type has its own 
  condition sub-fields — e.g. `spark plasma sintering` → `temperature`, 
  `pressure`, `time`, `heating rate`, `voltage`; `encapsulated melting` → 
  `ampoule material`, `ampoule environment`, `thermal cycle`, `cooling 
  method`, `post heat-treatment temperature`/`time`. Open the schema for the 
  chosen process and fill every sub-field the paper reports.
- **Measurement conditions**: fill the linked `measurement` object 
  (`instrument`, `acquisition condition`, …) from Methods.
- **Property values**: fill the structured value fields for numbers the paper 
  states **explicitly** (e.g. a peak `ZT` quoted in the text, or a value in a 
  table). Do **not** digitize plotted curves to populate value arrays — only 
  numbers given as text/tables.
- Leave a sub-field `null` only when the paper truly omits it.

### Schema structure overview

The attached schema is large (tens of thousands of lines). Here is its 
high-level shape so you can navigate it. The root is an object with 
required `metadata` and `materials` (plus optional `system`):

- `metadata` — dataset-level info: `data name`, `data classification[]`, 
  `data generation date`, `data source`, `contributor`, `keywords[]`, 
  `embargo`, `rights`, and `publication`.
  - `metadata.publication` — bibliography, plus:
    - `scope` — the paper's framing: `paradigms[]` (enum), `purposes[]`, 
      `approaches[]`, `conclusions[]` (each `{text, samples}`), 
      `classifications[]`, `comments[]`.
    - `samples[]` — each measured sample: `sample local id`, 
      `components[]`, etc.
- `materials[]` — one entry per material studied: `id`, `name`, 
  `chemical information`, `model`, `structure`, `property` (measured 
  properties), and `process[]` (synthesis/processing steps such as ball 
  milling, arc melting, spark plasma sintering).
- `system` — the device/system class: `name` (enum) with conditional fields.

Conventions used throughout:
- Every `notes` / `comments` field is an **array of strings**.
- Numeric quantities use a `{value, unit, …}` shape with optional uncertainty.
- `figure` / `graph` / `table` each split into `structure` (brief), 
  `description` (objective detail), and `digitization` (leave as empty 
  string `""` — not part of this public extraction).
- Reference a sample by its `sample local id`.

Strict extraction principles: record only what the paper states (never 
infer "typical" values); missing → `null` (keep the field); a value that 
doesn't match an enum → `null` plus a note.

### Output format (two files: English + Japanese)

Produce **two** JSON files, both conforming to the same attached schema:

1. **English** — filename = base name of the input PDF with a `.json` 
   extension (e.g., `smith2024.pdf` → `smith2024.json`).
2. **Japanese** — the same base name with a `_ja.json` suffix 
   (e.g., `smith2024_ja.json`).

The Japanese file is a translation of the English file, **not a separate 
extraction**. The two files MUST be structurally identical:

- **Translate** only natural-language free-text *values*: `purposes`, 
  `approaches`, `conclusions[].text`, every `notes[]` entry, and the 
  `structure` / `description` text of `figure` / `graph` / `table`.
- **Keep byte-for-byte identical** in both files: all JSON keys/field 
  names, every enum value, all numbers and units, sample/material local 
  ids, DOI and other identifiers, chemical formulas and element symbols, 
  instrument names, measurement-method names, journal title, and 
  figure/table numbers and labels. Do not translate, reorder, add, or 
  drop any field. (Translate only general-language prose within the 
  fields listed above.)
- A field that is `null` in English stays `null` in Japanese.

Do not paste the JSON body inline in chat. Provide both as downloadable 
files.

### Schema extension candidates

**Before proposing anything, check it is not already in the schema.** The 
schema already defines an extensive taxonomy — scan 
`$defs.measurement.items.properties` (~75 techniques) and 
`$defs.process.items.properties` (~95 processes) before deciding a technique 
is new. Most named techniques already exist under a standard acronym; map 
synonyms to the existing one and record it under `similar_existing_fields` 
rather than proposing a duplicate. For example: `XPS` / `UPS` → existing 
`PES`; near-edge XAS → existing `XANES`; XRD → `X-ray diffraction`. Only 
propose a candidate when no existing type covers the concept.

If the paper contains concepts genuinely not expressible in the existing 
schema, list them in the chat body as Schema extension candidates 
alongside the JSON file. Include all of the following items for 
each candidate:

- `field_path`: proposed placement in the schema 
  (e.g., `$defs.process.properties.induction_melting`)
- `type`: JSON Schema type (string, array, object, number, etc.)
- `description`: why this field is needed and what it expresses
- `example_value`: concrete instance from the paper
- `category`: top-level classification (metadata / materials / 
  system / process / measurement / analysis)
- `encountered_in`: source location (paper DOI plus section, 
  figure, or table number, e.g., `10.2320/matertrans.MF201613, 
  Section 2`)
- `suggested_subfields`: when applicable, proposed sub-properties 
  (e.g., `temperature`, `holding_time`, `atmosphere`)
- `similar_existing_fields`: any similar fields found by searching 
  the existing schema (state `none` if no similar field exists)

### Expected response structure

```
[Attached English JSON file (e.g., smith2024.json)]
[Attached Japanese JSON file (e.g., smith2024_ja.json)]

## Schema extension candidates

(Bullet list with the eight items above for each candidate; write 
"none" if no candidates.)

## Notes

(Optional: any observations during extraction, ambiguous cases, or 
points worth flagging.)
```
