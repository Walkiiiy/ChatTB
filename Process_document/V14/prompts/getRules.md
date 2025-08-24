You are given:
1. A natural language question about a database schema.
2. A list of amendments (amends) that explain how to fix an incorrect SQL query into the correct SQL query.

Your task:
- Extract **general rules (rules)** from the amends.
- Classify each rule into one of the following categories:
  1. Ambiguity in question: Keywords in the question do not clearly map to schema metadata, or the question has multiple possible meanings.
  2. Lack of background knowledge: The question contains unknown domain-specific terms or metrics whose calculation is non-standard.
  3. Schema structural issues: Mistakes caused by schema design, such as uneven table sizes, special foreign keys, or join structures.
  4. Output formatting issues: The logical answer is correct, but the returned result format does not match the expectation.
  5. Reasoning/Planning issues: The overall reasoning or strategy is wrong, and the query must be restructured with a new approach.

Output format:
- Return a list of extracted rules.  
- Each rule should include:
  - A short, generalized statement of the rule.  
  - Its assigned category (from the 5 categories).  

Example:

Input amends:
"Do use a single query with conditional aggregation instead of separate CTEs for counts. Do calculate the ratio directly using SUM(CASE...) instead of separate count columns. Do add the filter StatusType = 'Merged' instead of omitting it. Do remove the CASE statement for division by zero protection since the aggregation handles this implicitly. Do eliminate the CTE structure and column aliases instead of maintaining separate result sets. Do perform the calculation in a single SELECT instead of cross-joining separate count results."

Question:
"What is the ratio of merged Unified School District schools in Orange County to merged Elementary School District schools?"

Expected output rules:
- Rule: Prefer conditional aggregation (SUM with CASE) instead of separate CTE counts. → Category: Reasoning/Planning issues
- Rule: Always include domain-specific filters mentioned in the question (e.g., StatusType = 'Merged'). → Category: Ambiguity in question
- Rule: Avoid unnecessary division-by-zero handling if aggregation already prevents it. → Category: Reasoning/Planning issues
- Rule: Eliminate redundant CTE structures and compute results in a single query. → Category: Schema structural issues
- Rule: Use direct calculation in one SELECT rather than cross-joining separate counts. → Category: Reasoning/Planning issues
