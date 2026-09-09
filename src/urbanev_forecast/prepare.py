"""Prepare a user-supplied UrbanEV prefix locally; do not redistribute targets."""
import argparse
import hashlib
import io
from pathlib import Path
import numpy as np
import pandas as pd


def prepare_rates(data_root: Path, hours: int, output: Path):
    if not 1 <= hours <= 4344 or output.exists():
        raise ValueError("Require 1..4344 hours and a new output file")
    with (data_root/"occupancy.csv").open("rb") as handle:
        payload=b"".join(handle.readline() for _ in range(hours+1))
    occupancy=pd.read_csv(io.BytesIO(payload),index_col=0,parse_dates=True)
    info=pd.read_csv(data_root/"inf.csv",dtype={"TAZID":str})
    capacity=info.groupby("TAZID",sort=False)["charge_count"].sum()
    occupancy.columns=occupancy.columns.astype(str)
    if occupancy.shape!=(hours,275) or not occupancy.index.equals(pd.date_range("2022-09-01",periods=hours,freq="h")):
        raise ValueError("Unexpected hourly occupancy prefix")
    counts=capacity.reindex(occupancy.columns).to_numpy(np.float32)
    if not np.isfinite(counts).all() or (counts<=0).any():
        raise ValueError("Missing or invalid zone capacities")
    rates=occupancy.to_numpy(np.float32)/counts[None,:]
    if not np.isfinite(rates).all() or ((rates<0)|(rates>1)).any():
        raise ValueError("Invalid rates; do not silently impute or clip source observations")
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rates,index=occupancy.index,columns=occupancy.columns).to_csv(output,index_label="date")
    return {"rows":hours,"channels":275,"prepared_sha256":hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--hours",type=int,default=648,help="648 covers fold-1 train/validation; 4344 covers all six months")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(prepare_rates(args.data_root,args.hours,args.output))


if __name__=="__main__":main()
