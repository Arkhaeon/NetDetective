# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Graph Exp
#     language: python
#     name: graph-exp
# ---

# %%
#### # setup autoreload
# %reload_ext autoreload
# %autoreload 2

# import Python packages
import logging
import math
import numpy as np
import scipy as sp
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import sklearn as skl
from IPython.display import display

# import private/support packages
# import module_support_lib as msl
from graph_functions import *

# suppress warning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# %%
# df = pd.read_csv("data_sample-new.csv", index_col=0)
print("Reading CSVs")
df1 = pl.read_csv("data_day1-new.csv", has_header=True, dtypes={"source.port":pl.Float64(), "destination.port":pl.Float64()})  # , skiprows=0, nrows=10)
df2 = pl.read_csv("data_day2-new.csv", has_header=True, dtypes={"source.port":pl.Float64(), "destination.port":pl.Float64()})  # , skiprows=0, nrows=10)
df1 = df1.with_columns([pl.col("source.port").cast(pl.Int64), pl.col("destination.port").cast(pl.Int64)])
df2 = df2.with_columns([pl.col("source.port").cast(pl.Int64), pl.col("destination.port").cast(pl.Int64)])
# display(df)
# df = pl.concat([df1, df2])
# df.reset_index(drop=True)

print("Filtering")

# Filter out Zeek entries in MALCOLM
# criterion = lambda row: "arkime" in row["event.provider"]
df1 = df1.filter(pl.col("event.provider").str.contains("arkime"))
df2 = df2.filter(pl.col("event.provider").str.contains("arkime"))

# Filter out LLDP
# criterion = lambda row: "lldp" not in row["protocol"]
df1 = df1.filter(pl.col("protocol").str.contains("lldp").not_())
df2 = df2.filter(pl.col("protocol").str.contains("lldp").not_())



# display(df1)
# display(df2)

df = df1.extend(df2)

# %%
label_df = pd.read_csv("label_data.csv", header=0).set_index("ip")

# %%
# ip = "10.170.101.54"
ip = "10.120.0.200"
ip_src_df = df.filter(pl.col("source.ip") == ip)
ip_dst_df = df.filter(pl.col("destination.ip") == ip)

sns.histplot(ip_src_df, x="firstPacket", bins=200)

# %%
sns.histplot(ip_dst_df, x="firstPacket", bins=200)

# %%
sns.histplot(ip_src_df, x="firstPacket", bins=np.arange(1461418200000, 1461454200031, 600000))

# %%
d1_start = np.round(df["firstPacket"].min(), decimals=-2)
d1_end = df.filter(pl.col("lastPacket") <= 1461418000000)["lastPacket"].max()
d2_start = np.round(df.filter(pl.col("firstPacket") >= 1461417000000)["firstPacket"].min(), decimals=-2)
d2_end = df["lastPacket"].max()

ten_min = 600000
one_hour = ten_min * 6

d1_bins = np.arange(d1_start, d1_end + one_hour, one_hour)
d2_bins = np.arange(d2_start, d2_end, one_hour)

d1_names = [f"d1_{i}" for i in range(len(d1_bins))]
d2_names = [f"d2_{i + 1}" for i in range(len(d2_bins))]

time_slices = list(zip(np.append(d1_bins, d2_bins)[:-1], np.append(d1_bins, d2_bins)[1:]))
time_slices.pop(10)

total_time = ((d1_end - d1_start) + (d2_end - d2_start))/(60*60*1000)
print(total_time)

slice_names = (d1_names + d2_names)
slice_names.pop(0)
slice_names.pop(-1)

print(f"Sanity check: {len(time_slices) == len(slice_names)}") 

import pickle

with open("time_slices.pkl", "wb") as f: 
    pickle.dump(time_slices, f)
with open("slice_names.pkl", "wb") as f: 
    pickle.dump(slice_names, f)

# with open("time_slices.pkl", "rb") as f: 
#     time_slices = pickle.load(f)
# with open("slice_names.pkl", "rb") as f: 
#     slice_names = pickle.load(f)

# %%
src_df = (
    df.select(["source.ip", "firstPacket"])
    .group_by(["source.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

dst_df = (
    df.select(["destination.ip", "firstPacket"])
    .group_by(["destination.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

# %%
display(src_df)


# %%
def expand_col(df, col):
        return df.with_columns(
            pl.col(col).list.to_struct(
                fields=lambda idx: col + f"_{idx}",
                n_field_strategy="max_width",
            )
        ).unnest(col)


# %%
hist_src_df = expand_col(expand_col(src_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"])
hist_dst_df = expand_col(expand_col(dst_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"])

hist_dst_df.filter(pl.col("destination.ip").str.starts_with("10.20.20."))

# %%
hist_src_df.to_pandas().set_index("source.ip").to_pickle("hist_src_df.pkl")
hist_dst_df.to_pandas().set_index("destination.ip").to_pickle("hist_dst_df.pkl")

# %%
hist_src_df = pd.read_pickle("hist_src_df.pkl")
hist_dst_df = pd.read_pickle("hist_dst_df.pkl")

# %%
hist_df = hist_src_df.join(hist_dst_df, how="outer", rsuffix="_inc")
hist_df = hist_df.fillna(0)
for column in hist_src_df.columns:
    hist_df[column] = hist_df[column] + hist_df[column + "_inc"]
    hist_df.drop(column + "_inc", axis=1, inplace=True)

# %%
hist_df.loc["10.120.0.200"]

# %%
bool_hist_df = hist_df > 0
for column in hist_df:
    label_df[column] = bool_hist_df[column][label_df.index]

# %%
import swifter

def hist_label(row, col):
    if row[col]:
        return row["red_vs_blue"]
    else:
        return -1

for column in hist_df:
    print(column)
    label_df["rvb_" + column] = label_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: hist_label(row, column), axis=1)

label_df.to_pickle("label_data_time.pkl")

# %%
subnet_df = d1_src_df.filter(pl.col("source.ip").str.starts_with("10.20.20."))


subnet_df_exp = subnet_df.with_columns(
        pl.col("d1_hist").list.to_struct(
            fields=lambda idx: f"d1_hist_{idx}",
            n_field_strategy="max_width",
        )
    ).unnest("d1_hist").drop("d1_hist_10")


# %%
from matplotlib.colors import LogNorm

hist_df_filter = hist_df.filter(pl.col("source.ip"))

sns.heatmap(hist_df_filter.drop("source.ip"), yticklabels=hist_df_filter["source.ip"], norm=LogNorm(clip=True))

# %%
df.columns

# %%
# Label unresp - never sent anything and never replied to anything

ur_src_df = (
    df.select(["source.ip", "firstPacket"])
    .group_by(["source.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

ur_dst_df = (
    df.filter(pl.col("destination.bytes").gt(0)).select(["destination.ip", "firstPacket"])
    .group_by(["destination.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

ur_hist_src_df = expand_col(expand_col(ur_src_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"]).to_pandas().set_index("source.ip")
ur_hist_dst_df = expand_col(expand_col(ur_dst_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"]).to_pandas().set_index("destination.ip")


# %%
ur_hist_df = ur_hist_src_df.join(ur_hist_dst_df, how="outer", rsuffix="_inc")
ur_hist_df = ur_hist_df.fillna(0)
for column in ur_hist_src_df.columns:
    ur_hist_df[column] = ur_hist_df[column] + ur_hist_df[column + "_inc"]
    ur_hist_df.drop(column + "_inc", axis=1, inplace=True)

# %%
ur_hist_df
# 0s are inactive or ignored hosts

# %%
# label ignored - never got any replies to anything and never received anything
ig_src_df = (
    df.filter(pl.col("destination.bytes").gt(0)).select(["source.ip", "firstPacket"])
    .group_by(["source.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

ig_dst_df = (
    df.select(["destination.ip", "firstPacket"])
    .group_by(["destination.ip"]).agg(
        d1_hist = pl.col("firstPacket").hist(bins=list(d1_bins)),
        d2_hist = pl.col("firstPacket").hist(bins=list(d2_bins)),
    )
)

ig_hist_src_df = expand_col(expand_col(ig_src_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"]).to_pandas().set_index("source.ip")
ig_hist_dst_df = expand_col(expand_col(ig_dst_df, "d1_hist"), "d2_hist").drop(["d1_hist_0", "d1_hist_11", "d2_hist_0", "d2_hist_11"]).to_pandas().set_index("destination.ip")

# %%
ig_hist_df = ig_hist_src_df.join(ig_hist_dst_df, how="outer", rsuffix="_inc")
ig_hist_df = ig_hist_df.fillna(0)
for column in ig_hist_src_df.columns:
    ig_hist_df[column] = ig_hist_df[column] + ig_hist_df[column + "_inc"]
    ig_hist_df.drop(column + "_inc", axis=1, inplace=True)

# %%
ig_hist_df
# 0s are inactive or unresp hosts

# %%
type(bool_hist_df.index)

# %%
ig_missing = pd.Index(set(bool_hist_df.index) - set(bool_ig_hist_df.index))
ur_missing = pd.Index(set(bool_hist_df.index) - set(bool_ur_hist_df.index))

# %%
bool_ig_hist_df = ig_hist_df > 0
bool_ig_missing_df = bool_hist_df.loc[ig_missing]
bool_ig_hist_df = pd.concat([bool_ig_hist_df, bool_ig_missing_df])

bool_ur_hist_df = ur_hist_df > 0
bool_ur_missing_df = bool_hist_df.loc[ur_missing]
bool_ur_hist_df = pd.concat([bool_ur_hist_df, bool_ur_missing_df])

# %%
for column in ig_hist_df:
    label_df["ig" + column] = bool_ig_hist_df[column][label_df.index]
for column in ur_hist_df:
    label_df["ur" + column] = bool_ur_hist_df[column][label_df.index]

# %%
import swifter

def hist_label(row, col):
    if not row[col]:
        return -1
    elif not row["ur" + col]:
        return 0
    elif not row["ig" + col]:
        return 1
    else:
        return row["red_vs_blue"]
    
for column in hist_df:
    print(column)
    label_df["rvb_" + column] = label_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: hist_label(row, column), axis=1)

label_df.to_pickle("label_data_time_ext.pkl")


# %%
def ip_is_public(ip_string):
    return ipa.ip_address(ip_string).is_global

public_ips_bool = heuristic_df["source.ip"].map_elements(ip_is_public)

public_ips = pl.DataFrame(heuristic_df.filter(public_ips_bool)["source.ip"])
# public_ips.write_csv("public_ips.csv")

# public_ips = pl.read_csv("public_ips.csv", has_header=True)["source.ip"]

# %%
def ip_is_v4(ip_string):
    return ipa.ip_address(ip_string).version == 4

ip4s_bool = heuristic_df["source.ip"].map_elements(ip_is_v4)

ip4s = pl.DataFrame(heuristic_df.filter(ip4s_bool)["source.ip"])
# ip4s_ips.write_csv("public_ips.csv")

# %%
ips = heuristic_df.filter(public_ips_bool.not_() & ip4s_bool)["source.ip"].unique().to_list()
label_df = label_ips(ips)
label_df["blue_team_bool"] = label_df["blue_team"] != "None"


# %%
import swifter

def red_vs_blue_label(row):
    if row.name not in real_df["source.ip"]:
        if row.name in ghost_df["source.ip"]:
            return 0  # "ghost"
        else:
            return 1
    if row["red_team"]:
        return 2
    if row["blue_team_bool"]:
        return 3
    else:
        return 4


rvb_dict = {
    0: "ghost",
    1: "ddos",
    2: "red",
    3: "blue",
    4: "grey",
}

label_df["red_vs_blue"] = label_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: red_vs_blue_label(row), axis=1)

label_df.to_csv("label_data.csv", mode="w", header=True, index_label="ip")

# label_df = pd.read_csv("label_data.csv", header=0).set_index("ip")
# display(label_df)

# %%
label_df["red_vs_blue"].value_counts()

# %% [markdown]
#
