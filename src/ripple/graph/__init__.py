"""Graph (M1): build the call/import/inherit graph and traverse it to answer
change-impact questions (transitive callers/dependents = the blast radius).
In-memory (NetworkX) first; Postgres edge tables in M4."""
