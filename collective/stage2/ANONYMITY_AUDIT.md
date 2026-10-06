# Numeric archive scanning

The expanded artifact's initial byte scan found a forbidden four-letter token at byte offset 7,754,745 inside the compressed
`gradient.npy` member of `native_random/profiles/SGCN-wiki_elec-T8-s52002.npz`. The archive has only
float64/float32/int64 numerical arrays; no string arrays or identifying archive names exist. This is
a coincidental compressed-byte sequence, not a reference to another project.

The builder now reads NPZ member names, NPY dtype/header metadata and all string-array values. It rejects
object arrays. Numerical compressed payloads are not interpreted as text; all other text, paths and
checkpoint pickle metadata retain their identifying-string checks. No experimental file is rewritten
for this correction. Source-array hashes and paper outputs are unchanged.
