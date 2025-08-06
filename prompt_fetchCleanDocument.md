You are given a dataset consisting of natural language queries paired with databases. Each query includes:

* A `db_id` (database identifier),
* A `question` (natural language query about the database),
* An `evidence` (a clarification or constraint about the semantics of the query, e.g., how specific terms or filters are defined within the data context).

Here is an example query object:

```json
{
  "db_id": "california_schools",
  "question": "Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.",
  "evidence": "Magnet schools or offer a magnet program means that Magnet = 1"
}
```

Your task is as follows:

1. I will provide you with a brief introduction to the database and a list of such query objects.
2. For each query, **search the web for authoritative or relevant online resources (documents, web pages, datasets, or official websites)** that could **support or justify the evidence field**. That is, the materials must allow a human or a system to infer or validate the logic in the `evidence`.
3. For each query, return:

   * A list of relevant links (URLs),
   * Optionally, a short summary or snippet showing how the resource supports the evidence.

**Output format (per query):**

```json
{
  "db_id": "california_schools",
  "question": "...",
  "evidence": "...",
  "sources": [
    {
      "url": "https://example.gov/education/magnet-programs.html",
      "snippet": "In the California Schools dataset, 'Magnet = 1' indicates that a school offers a magnet program..."
    },
    ...
  ]
}
```

Make sure that the retrieved documents are trustworthy and sufficiently clear to support the evidence logically or explicitly.

