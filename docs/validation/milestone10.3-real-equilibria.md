# Milestone 10.3 unattended matrix convergence

PLACEHOLDER — regenerated from the final matrix run before the PR is ready.

## Controls and reproduction

```bash
source .venv/bin/activate
for i in 0 1 2 3 4; do
  python examples/converge_cut_equilibria.py \
    --output /tmp/milestone103/shard$i.json --file-index $i --resume &
done
wait
python examples/merge_matrix_shards.py /tmp/milestone103/shard*.json \
  --output docs/validation/milestone10.3-real-equilibria.json
```

## Outcomes

(filled from the final run)

## Failure classes and remediation

(filled from the final run)

## Backend and extractor consistency

(filled from the final run)
