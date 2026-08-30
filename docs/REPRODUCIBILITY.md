# Reproduction levels

1. **Repository-only replay** validates schemas, public hashes, headline summary consistency, citations, and privacy boundaries without data.
2. **Licensed-data recomputation** adds user-obtained UrbanEV and Paris development targets, verifies target hashes, and recomputes headline metrics from target-free release packages.
3. **Model re-execution** uses the GPU lock, executed source snapshots, model/download manifests, and user-obtained data. It is not required for checking the published stored-prediction claims.

For licensed-data recomputation, place the UrbanEV `occupancy.csv` and `inf.csv` files under the registered data root. For Paris development, place the upstream CSV containing `date`, `Station`, `Available`, `Charging`, `Passive`, and `Other` under that root. The public tool restricts Paris reconstruction to timestamps no later than 30 November 2020, rebuilds the non-available-port fraction, aligns public target timestamps and entity IDs, casts to the recorded target dtype, and requires the exact stored target hash before scoring.

Formal and protected Paris roles are outside all public reproduction levels. Their analytical-access count remains zero.
