# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: graph-exp
#     language: python
#     name: python3
# ---

# %%
# # setup autoreload
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

display(df1)
display(df2)

df = df1.extend(df2)

# %%
# Heuristically label ghost IPs
src_df = (
    df.select(["source.ip", "source.bytes", "destination.bytes", "network.bytes"])
    .group_by(["source.ip"])
    .sum()
)

dst_df = (
    df.select(["destination.ip", "source.bytes", "destination.bytes", "network.bytes"])
    .group_by(["destination.ip"])
    .sum()
)

heuristic_df = src_df.join(dst_df, left_on="source.ip", right_on="destination.ip", how="outer", suffix="_dst").fill_null(0)
ghost_df = heuristic_df.filter(((pl.col("source.bytes") == 0) & (pl.col("destination.bytes_dst") == 0)))
ddos_df = heuristic_df.filter((pl.col("destination.bytes") == 0) & (pl.col("source.bytes_dst") == 0))
real_df = heuristic_df.filter((((pl.col("destination.bytes") == 0) & (pl.col("source.bytes_dst") == 0)) | ((pl.col("source.bytes") == 0) & (pl.col("destination.bytes_dst") == 0))).not_())
# "destination.bytes == 0" and "source.bytes_dst" == 0 (Nothing ever replied and nothing ever sent anything to it)

display(ghost_df)
display(ddos_df)
display(real_df)


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
