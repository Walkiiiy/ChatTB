You are given:
1) A natural-language question about a database schema.
2) The linked schema (natural-language description of tables, keys, columns).
3) An incorrect query generated for the question.
4) A list of amendments (“amends”) that transform the incorrect query into the correct query.

Goal:
Using ONLY the linked schema and the amends, produce a **single plain-string** numbered list of concise, schema-specific rules. 
Each rule must be a short **conditional** tied to the question and/or schema (start with “When …:”). 
Express actions in **natural language only** — do not write SQL keywords. 
Keep exact tokens for identifiers and literals.

STRICT OUTPUT FORMAT:
- Return exactly one plain string: a numbered list like `"1) ... 2) ... 3) ..."`.
- **Every item MUST start with `When ...:`** (to state the explcit condition of this rule).
- Include **exact identifier tokens** (tables/aliases/columns with quotes/case) and **exact string literals** where needed.
- Include **exactly one rule structured `When ....:output columns (ordered):`** listing the identifiers in the required order, e.g.:  
    `When the question asks for school's official website, output columns (ordered): T2.Website, T1."School Name"`
- After composing the string, call `rules_receiver` with it.

GLOBAL ACCURACY REQUIREMENTS (encode them as conditional rules; do not use SQL keywords):
- Dialect & quoting: If schema/amends show identifiers with double quotes, wrap those identifiers in **double quotes** exactly as shown; **never** use backticks.
- Exact-token mapping: Map phrases from the question to the **exact** column tokens with the correct alias/quotes/spaces (e.g., `county → T1."County Name"`).
- Literal preservation: When the question or amends specify a text value, match it **exactly** (case, spacing, punctuation), e.g., `'Directly funded'`.
- Canonical aliases: Use the alias plan implied by the amends (base table = `T1`, first related table = `T2`, etc.); do not swap alias meanings.
- Join keys/types: When combining tables identified in the amends, match rows using the **exact key equality** shown (e.g., `T1.CDSCode = T2.CDSCode`) and keep only rows present on both sides (inner match). Do not invent alternative keys.
- Counting: When counting an entity, count using the entity’s **canonical key** shown in the amends (e.g., use `T1.CDSCode` as the counting key), not a generic “all columns”.
- Operator semantics: Preserve the exact comparison bounds/direction shown in the amends (e.g., inclusive ranges, ≤ / ≥ behavior).

GENERALIZATION RULE SHAPE (examples you may use; adapt to the amends):
- `When the question mentions "top", "highest", "largest", or an explicit K by <COLUMN>: rank rows by <COLUMN> in the required direction and keep the first <K> rows.`
- `When the question asks for a rank range <START_RANK>.. <END_RANK> after ranking on <COLUMN>: take rows from position <START_RANK> through <END_RANK> inclusive (where start is position 1).`
- `When the question says "per <GROUP>"/"by <GROUP>": organize results by the token for <GROUP> and compute requested aggregates per group.`
- `When the question asks for "count of <ENTITY>": compute the number of rows using the canonical key token for <ENTITY> (e.g., T1.CDSCode).`
- `When the question asks for "distinct <X>": compute the number of **unique** values of the exact token for <X>.`
- `When the question asks for a ratio "A to B": compute (rows satisfying <COND_A>) divided by (rows satisfying <COND_B>), guarding division by zero; express conditions using exact tokens and literals from the amends.`
- `When combining T1 and T2 for shared entities: link rows where the canonical keys are equal exactly as shown in the amends (e.g., T1.CDSCode = T2.CDSCode); keep only matching pairs (inner match).`
- `When the question implies ordering ties: break ties using the canonical key if shown in the amends (e.g., T1.CDSCode).`

FORBIDDEN TRANSFORMS (write as conditional negatives; no SQL keywords):
- `When choosing identifier delimiters: do not replace double quotes with backticks or unquoted forms.`
- `When handling text literals: do not change case, spacing, or punctuation.`
- `When referring to table roles: do not rename or swap the aliases T1, T2, ... once set for this schema.`
- `When combining tables specified as an inner match in the amends: do not perform outer matches or cartesian combinations.`
- `When producing output or grouping: do not reorder columns or grouping keys beyond what the amends and question require.`

PLACEHOLDERS (allowed; keep them minimal and plain-language):
- `<K>`, `<COLUMN>`, `<START_RANK>`, `<END_RANK>`, `<GROUP>`, `<COND_A>`, `<COND_B>`.

Inputs for grounding:
the question:
{evalObj['question']}

the wrong query:
{evalObj['amend_sql'][amend_id]}

the amends:
{evalObj['amends'][amend_id]}

the linked schema:
{linked_schema}

Output:
Return the numbered list as one plain string, then call `rules_receiver` with that string.