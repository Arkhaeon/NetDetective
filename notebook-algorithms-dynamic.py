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

# display(df1)
# display(df2)

df = df1.extend(df2)

# %%
public_ips = pl.read_csv("public_ips.csv", has_header=True)["source.ip"]

# %%
import pickle

rvb_dict = {
    0: "unresp",
    1: "ignored",
    2: "red",
    3: "blue",
    4: "gray",
}
rvb_labels = list(rvb_dict.values())

# label_df = pd.read_csv("label_data.csv", header=0).set_index("ip")
label_df = pd.read_pickle("label_data_time_ext.pkl")
# display(label_df)

with open("time_slices.pkl", "rb") as f: 
    time_slices = pickle.load(f)
with open("slice_names.pkl", "rb") as f: 
    slice_names = pickle.load(f)

# %% jupyter={"source_hidden": true}
li_numeric_cols = (
    "source.bytes",
    "destination.bytes",
)

li_port_cols = (
    "source.port",
    "destination.port",
)

li_ip_cols = (
    "byte_1", "byte_2", "byte_3", "byte_4",
)

li_port_agg_tuple = sum(((
    pl.col(port_col).unique().lt(1024).sum().alias(port_col + "_system_unique"),
    pl.col(port_col).filter(pl.col(port_col).lt(1024)).std().alias(port_col + "_system_std"),
    pl.col(port_col).filter(pl.col(port_col).eq(pl.col(port_col).filter(pl.col(port_col).lt(1024)).mode().first())).count().alias(port_col + "_system_most_count"),
    pl.col(port_col).unique().ge(1024).lt(49151).sum().alias(port_col + "_user_unique"),
    pl.col(port_col).filter(pl.col(port_col).ge(1024).lt(49151)).std().alias(port_col + "_user_std"),
    pl.col(port_col).filter(pl.col(port_col).eq(pl.col(port_col).filter(pl.col(port_col).ge(1024).lt(49151)).mode().first())).count().alias(port_col + "_user_most_count"),
    pl.col(port_col).ge(49151).sum().alias(port_col + "_dynamic"),
    ) for port_col in li_port_cols
), ()) + (pl.col("source.port").lt(pl.col("destination.port")).sum().alias("host_lt_other"),)

li_num_agg_tuple = sum(((
    pl.col(numeric_col).mean().alias(numeric_col + "_mean"),
    ) for numeric_col in li_numeric_cols
), ())

li_protocol_agg_tuple = (
    pl.col("ipProtocol").unique().count().alias("ipProtocol_unique"),
    pl.col("ipProtocol").filter(pl.col("ipProtocol").eq(pl.col("ipProtocol").mode().first())).count().alias("ipProtocol_most_count"),
)

li_ip_agg_tuple = sum(((
    pl.col(ip_col).str.to_integer(base=10).mean().alias("ip_octet_" + ip_col) / 255,
) for ip_col in li_ip_cols
), ())

li_agg_tuple = (pl.col("source.bytes").count().alias("sessions"),) + li_port_agg_tuple + li_num_agg_tuple + li_protocol_agg_tuple + li_ip_agg_tuple

# %% jupyter={"source_hidden": true}
stat_node_numeric_cols = (
    "duration",
    "source.bytes",
    "destination.bytes",
    "source.packets",
    "destination.packets",
)

stat_edge_numeric_cols = (
    "duration",
    # "source.bytes",
    "destination.bytes",
    # "source.packets",
    "destination.packets",
)

stat_node_agg_tuple = sum(((
    pl.col(numeric_col).mean().alias(numeric_col + "_mean"),
    pl.col(numeric_col).var().alias(numeric_col + "_var"),
    ) for numeric_col in stat_node_numeric_cols
), ())

stat_edge_agg_tuple = sum(((
    pl.col(numeric_col).mean().alias(numeric_col + "_mean"),
    ) for numeric_col in stat_edge_numeric_cols
), ())


# %% jupyter={"source_hidden": true}
lyu_num_agg_tuple = (
    pl.col("network.bytes").mean().alias("network.bytes_mean"),
    pl.col("network.bytes").var().alias("network.bytes_var"),
)

lyu_protocol_agg_tuple = (
    pl.col("ipProtocol").mode().first().alias("ipProtocol_mode"),
    pl.col("ipProtocol").ne(pl.col("ipProtocol").mode().first()).mean().alias("ipProtocol_minor_fraction"),
)

lyu_port_agg_tuple = (
    (pl.col("source.port").unique().count() / pl.col("destination.port").unique().count()).alias("port_intext_ratio"),
    pl.col("source.port").value_counts(sort=True).alias("source.port_counts"),
    pl.col("destination.port").value_counts(sort=True).alias("destination.port_counts"),
)

lyu_agg_tuple = (pl.col("source.bytes").count().alias("sessions"),) + lyu_num_agg_tuple + lyu_protocol_agg_tuple + lyu_port_agg_tuple

# %% jupyter={"source_hidden": true}
from sklearn.base import BaseEstimator, TransformerMixin

class LiSessionGraphNodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X = pl.from_pandas(X)
        # Remove internet sessions
        global_ips = public_ips

        X = X.filter(
            (pl.col("source.ip").is_in(global_ips).not_())
            & (pl.col("destination.ip").is_in(global_ips).not_())
        )
        # ipv4_bool = X["source.ip"].map_elements(ip_is_v4)
        X = X.filter(pl.col("source.ip").str.contains(":", literal=True).not_())

        # Add session duration field
        X = X.drop([
            "network.bytes",
            "client.bytes",
            "server.bytes",
            "totDataBytes",
            "source.packets",
            "destination.packets",
            "network.packets",
            "firstPacket",
            "lastPacket",
            "event.dataset",
            "event.provider",
            ])

        X = pl.concat([X, X.rename({
            "source.ip": "destination.ip",
            "destination.ip": "source.ip",
            "source.port": "destination.port",
            "destination.port": "source.port",
            "source.bytes": "destination.bytes",
            "destination.bytes": "source.bytes",
            })[X.columns]]).rename({
            "source.ip": "ip", "destination.ip": "other_ip"
        })

        X = X.with_columns(
            [pl.col("other_ip").str.split_exact(".", 3).struct.rename_fields(["byte_1", "byte_2", "byte_3", "byte_4"]).alias("other_IP_bytes")]
        ).unnest("other_IP_bytes")

        X = X.group_by("ip").agg(li_agg_tuple).fill_nan(0).fill_null(0)

        session_scale = ['source.port_system_unique', 'source.port_system_most_count',
                         'source.port_user_unique', 'source.port_user_most_count', 'source.port_dynamic',
                         'destination.port_system_unique', 'destination.port_system_most_count',
                         'destination.port_user_unique', 'destination.port_user_most_count', 'destination.port_dynamic',
                         'ipProtocol_unique', 'ipProtocol_most_count',]
        # minmax_scale = ['source.port_system_std', 'source.port_user_std', 'destination.port_system_std', 'destination.port_user_std',]

        for col in session_scale:
            X.with_columns((pl.col(col) / pl.col("sessions")).alias(col))

        # for col in minmax_scale:
        #     X.with_columns((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()))

        return X.drop(["sessions"],)


# %% jupyter={"source_hidden": true}
from sklearn.base import BaseEstimator, TransformerMixin

class ProSessionGraphNodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X = pl.from_pandas(X)
        # Remove internet sessions
        global_ips = public_ips

        X = X.filter(
            (pl.col("source.ip").is_in(global_ips).not_())
            & (pl.col("destination.ip").is_in(global_ips).not_())
        )
        # ipv4_bool = X["source.ip"].map_elements(ip_is_v4)
        X = X.filter(pl.col("source.ip").str.contains(":", literal=True).not_())

        X = X.with_columns(duration=pl.col("lastPacket") - pl.col("firstPacket"))

        # Add session duration field
        X = X.drop([
            "network.bytes",
            "client.bytes",
            "server.bytes",
            "totDataBytes",
            # "source.packets",
            # "destination.packets",
            "source.port",
            "destination.port",
            "network.packets",
            "firstPacket",
            "lastPacket",
            "event.dataset",
            "event.provider",
            ])

        X = pl.concat([X, X.rename({
            "source.ip": "destination.ip",
            "destination.ip": "source.ip",
            "source.packets": "destination.packets",
            "destination.packets": "source.packets",
            "source.bytes": "destination.bytes",
            "destination.bytes": "source.bytes",
            })[X.columns]]).rename({
            "source.ip": "ip", "destination.ip": "other_ip"
        })

        X = X.group_by("ip").agg(stat_node_agg_tuple).fill_nan(0).fill_null(0)

        return X

class ProSessionGraphEdgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X = pl.from_pandas(X)
        # Remove internet sessions
        global_ips = public_ips

        X = X.filter(
            (pl.col("source.ip").is_in(global_ips).not_())
            & (pl.col("destination.ip").is_in(global_ips).not_())
        )

        X = X.filter(pl.col("source.ip").str.contains(":", literal=True).not_())

        # Add session duration field
        X = X.with_columns(duration=pl.col("lastPacket") - pl.col("firstPacket"))

        # Average over time fields?
        X = X.drop([
                "network.bytes",
                "client.bytes",
                "server.bytes",
                "totDataBytes",
                # "source.packets",
                # "destination.packets",
                "source.port",
                "destination.port",
                "network.packets",
                "firstPacket",
                "lastPacket",
                "event.dataset",
                "event.provider",
            ])

        X = pl.concat([X, X.rename({
            "source.ip": "destination.ip",
            "destination.ip": "source.ip",
            "source.packets": "destination.packets",
            "destination.packets": "source.packets",
            "source.bytes": "destination.bytes",
            "destination.bytes": "source.bytes",
            })[X.columns]]).rename({
            "source.ip": "ip", "destination.ip": "other_ip"
        })

        X = X.group_by(["ip", "other_ip"]).agg(stat_edge_agg_tuple).fill_nan(0).fill_null(0)

        return X


# %% jupyter={"source_hidden": true}
from sklearn.base import BaseEstimator, TransformerMixin

class LyuSessionGraphNodeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X = pl.from_pandas(X)
        # Remove internet sessions
        global_ips = public_ips

        X = X.filter(
            (pl.col("source.ip").is_in(global_ips).not_())
            & (pl.col("destination.ip").is_in(global_ips).not_())
        )
        # ipv4_bool = X["source.ip"].map_elements(ip_is_v4)
        X = X.filter(pl.col("source.ip").str.contains(":", literal=True).not_())

        # Add session duration field
        X = X.drop([
            # "network.bytes",
            "client.bytes",
            "server.bytes",
            "totDataBytes",
            "source.packets",
            "destination.packets",
            "network.packets",
            "firstPacket",
            "lastPacket",
            "event.dataset",
            "event.provider",
            ])

        src_df = X.group_by(["source.ip"]).agg(lyu_agg_tuple).rename({"source.ip": "ip"})

        dst_df = X.group_by(["destination.ip"]).agg(lyu_agg_tuple).rename({"destination.ip": "ip"})

        X = src_df.join(dst_df, on="ip", how="full", suffix="_inc", coalesce=True).fill_nan(0).fill_null(0)

        return X #.drop(["sessions"],)


# %% jupyter={"source_hidden": true}
def gen_features(df):
    li_trans = LiSessionGraphNodeTransformer()
    stat_trans = ProSessionGraphNodeTransformer()
    lyu_trans = LyuSessionGraphNodeTransformer()

    li_df = li_trans.fit_transform(df)
    stat_df = stat_trans.fit_transform(df)
    lyu_df = lyu_trans.fit_transform(df)

    tdf = li_df.join(stat_df, on="ip", how="left", coalesce=True).join(lyu_df, on="ip", how="left", coalesce=True)
    feature_df = tdf.to_pandas().set_index("ip")

    return feature_df


# %% jupyter={"source_hidden": true}
import itertools
import swifter

def compute_features(row, column, direction):
    features = list()
    sessions = row[direction]
    if np.isnan(sessions) or sessions is None or row[column] is None:
        return [1, 1, 1, 1, 0, 0, 0, 0, 0]
    else:
        cum_sum = 0
        q0 = False
        q1 = False
        q2 = False
        q3 = False
        counts = list()
        for item in row[column]:
            cum_sum = cum_sum + item["count"]
            counts.append(item["count"])
            
            if (cum_sum / sessions > 0) and not q0:
                features.append(item["count"] / sessions)
                q0 = True
            if (cum_sum / sessions > 0.25) and not q1:
                features.append(item["count"] / sessions)
                q1 = True
            if (cum_sum / sessions > 0.5) and not q2:
                features.append(item["count"] / sessions)
                q2 = True
            if (cum_sum / sessions > 0.75) and not q3:
                features.append(item["count"] / sessions)
                q3 = True

        features.append(np.var(counts))
        mean = np.mean(counts)
        std = np.std(counts)
        features.append(sum(count > mean for count in counts))
        features.append(sum(count > (mean + std)for count in counts))
        features.append(sum(count > (mean + 2*std)for count in counts))
        features.append(sum(count > (mean + 3*std)for count in counts))
        return features

def compute_top_5(row, column, direction, value):
    if row[column] is not None:
        count = 0
        features = list()
        for item in row[column]:
            if count < 5 and item is not None:
                features.append(int(item.get(value)))
                count = count + 1
            else:
                break
        if len(features) == 5:
            return features
        else:
            padding = 5 - len(features)
            return features + [0] * padding
    else:
        return [0, 0, 0, 0, 0]

def gen_extra_features(feature_df):
    for endpoint, direction in itertools.product(("source", "destination",), ("", "_inc")):
    # endpoint = "source"
    # direction = ""
        fs = endpoint + direction + "_features"
        ts = endpoint + direction + "_top5"
        feature_df[fs] = feature_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: compute_features(row, endpoint + ".port_counts" + direction, "sessions" + direction), axis=1,)
        fdf1 = pd.DataFrame(feature_df[fs].to_list(), columns=[fs + "fracq0", fs + "fracq1", fs + "fracq2", fs + "fracq3", fs + "var", fs + "abvavg", fs + "abvavg1s", fs + "abvavg2s", fs + "abvavg3s"], index=feature_df.index)
        feature_df[ts] = feature_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: compute_top_5(row, endpoint + ".port_counts" + direction, "sessions" + direction, endpoint + ".port"), axis=1,)
        fdf2 = pd.DataFrame(feature_df[ts].to_list(), columns=[ts + "_1", ts + "_2", ts + "_3", ts + "_4", ts + "_5"], index=feature_df.index)
        feature_df = pd.concat([feature_df, fdf1, fdf2], axis=1)
        # feature_df[fs + "fracq0"], feature_df[fs + "fracq1"], feature_df[fs + "fracq2"], feature_df[fs + "fracq3"], feature_df[fs + "var"], feature_df[fs + "abvavg"], feature_df[fs + "abvavg1s"], feature_df[fs + "abvavg2s"], feature_df[fs + "abvavg3s"] = feature_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: compute_features(row, endpoint + ".port_counts" + direction, "sessions" + direction), axis=1, result_type="expand",)
        # feature_df[ts + "_1"], feature_df[ts + "_2"], feature_df[ts + "_3"], feature_df[ts + "_4"], feature_df[ts + "_5"] = feature_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: compute_top_5(row, endpoint + ".port_counts" + direction, "sessions" + direction, endpoint + ".port" + direction), axis=1, result_type="expand",)

    return feature_df


# %% jupyter={"source_hidden": true}
def drop_inter_features(feature_df):

       feature_df = feature_df.drop([
              # "sessions",
              # 'sessions_inc',
              'source.bytes_mean_right', 'destination.bytes_mean_right',
              'source_features', 'source_top5', 'source_inc_features', 'source_inc_top5',
              'destination_features', 'destination_top5', 'destination_inc_features', 'destination_inc_top5',
              'source.port_counts', 'destination.port_counts','source.port_counts_inc', 'destination.port_counts_inc',], axis=1)
       feature_df = feature_df.drop([
              'ipProtocol_most_count',
              'ip_octet_byte_1',
              'ip_octet_byte_2',
              'ip_octet_byte_3',
              'ip_octet_byte_4',
       ], axis=1)

       return feature_df


# %%
# feature_df = drop_inter_features(gen_extra_features(gen_features(df)))
# feature_df["sessions"] = feature_df["sessions"] / total_time
# feature_df["sessions_inc"] = feature_df["sessions_inc"] / total_time
# feature_df.to_pickle("feature_df_full.pkl")

# %%
for slice_name, time_slice in zip(slice_names, time_slices):
    start_time = time_slice[0]
    end_time = time_slice[1]

    slice_df = df.filter(pl.col("firstPacket").gt(start_time) & pl.col("firstPacket").le(end_time))
    slice_feature_df = drop_inter_features(gen_extra_features(gen_features(slice_df)))
    slice_feature_df.to_pickle("slice_feature_df_" + slice_name + ".pkl")


# %%
import pickle

feature_df = pd.read_pickle("feature_df_full.pkl")

slice_dfs = list()
for slice_name in slice_names:
    with open("slice_feature_df_" + slice_name + ".pkl", "rb") as f:
        slice_dfs.append(pickle.load(f))

# %%
label_df.columns

# %%
slice_labels = list()
for slice_name, slice_df in zip(slice_names, slice_dfs):
    # rvb_col = "rvb_" + slice_names[slice_no].split("_")[0] + "_hist_" + slice_names[slice_no].split("_")[-1]
    rvb_col = "rvb_" + slice_name.split("_")[0] + "_hist_" + slice_name.split("_")[-1]
    slice_label = label_df[rvb_col][slice_df.index]
    slice_labels.append(slice_label)
# label_ip_set = set(label_df[rvb_col][label_df[rvb_col] >= 0].index) 
# slice_ip_set = set(slice_dfs[slice_no].index)

# %%
slice_labels[0]

# %%
# read data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight

labels = label_df["red_vs_blue"][feature_df.index]

class_weights = dict(enumerate(class_weight.compute_class_weight(
    "balanced", classes=np.unique(labels), y=labels
)))
print(class_weights)

sample_weights = class_weight.compute_sample_weight(class_weights, y=labels)
print(sample_weights)

X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(feature_df, labels, sample_weights, test_size=0.2, random_state=0, stratify=labels)

# %%
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC

# create model instance
# rf = DecisionTreeClassifier(class_weight=class_weights)
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
# rf.fit(X_train, y_train)

# fit model
full_clf = XGBClassifier()
full_clf.fit(X_train, y_train, sample_weight=weights_train)

# make predictions
train_preds = full_clf.predict(X_train)
preds = full_clf.predict(X_test)

print(classification_report(y_train, train_preds))
print(classification_report(y_test, preds))
score = f1_score(y_test, preds, average="macro")
print(score)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print(classification_report(y_test, preds))
disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds, normalize=None), display_labels=rvb_labels
).plot()

# %%
disp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds, normalize="true"), display_labels=rvb_labels
).plot()


# %%
def build_tt(slice_dfs, slice_labels, ips):
    time_list = list()
    X_list = list()
    y_list = list()
    for slice_name, slice_df, slice_label in zip(slice_names, slice_dfs, slice_labels):
        valid_ips = slice_df.index.isin(ips)
        time_list.append([slice_name for _ in range(len(slice_df[valid_ips]))])
        X_list.append(slice_df[valid_ips])
        y_list.append(slice_label[valid_ips])

    times = sum(time_list, [])
    X_s = pd.concat(X_list)
    y_s = pd.concat(y_list)
    return times, X_s, y_s


# %%
times, X_s, y_s = build_tt(slice_dfs, slice_labels, X_train.index)
times_test, X_s_test, y_s_test = build_tt(slice_dfs, slice_labels, X_test.index)

# %%
weights_train

# %%
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# create model instance
# rf = DecisionTreeClassifier(class_weight=class_weights)
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
# rf.fit(X_train, y_train)

# fit model
time_clf = XGBClassifier()
sample_weights = class_weight.compute_sample_weight(class_weights, y=y_s)
time_clf.fit(X_s, y_s,) #sample_weight=sample_weights)

# make predictions
train_preds = time_clf.predict(X_s)
preds = time_clf.predict(X_s_test)

print(classification_report(y_s, train_preds))
print(classification_report(y_s_test, preds))
score = f1_score(y_s_test, preds, average="macro")
print(score)

disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test, preds, normalize=None), display_labels=rvb_labels
).plot()

# %%
preds_ser = pd.Series(preds, index=y_s_test.index)

# %%
comparison_df = pd.concat([y_s_test, preds_ser], axis=1).rename(columns={0: "true", 1:"pred"})

# %%
comparison_df[comparison_df["true"] != comparison_df["pred"]].index.value_counts()

# %%
label_df["red_vs_blue"].loc["10.131.0.5"]

# %%
with pd.option_context('display.max_rows', 20, 'display.max_columns', None): 
    display(X_s_test.loc["10.131.0.5"])

# %%
with pd.option_context('display.max_rows', 20, 'display.max_columns', None): 
    display(X_s_test.loc["172.16.90.115"])

# %%
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# create model instance
# rf = DecisionTreeClassifier(class_weight=class_weights)
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
# rf.fit(X_train, y_train)

# fit model
time_clf = XGBClassifier()
sample_weights = class_weight.compute_sample_weight(class_weights, y=y_s)
time_clf.fit(X_s, y_s, sample_weight=sample_weights)

# make predictions
train_preds = time_clf.predict(X_s)
preds = time_clf.predict(X_s_test)

print(classification_report(y_s, train_preds))
print(classification_report(y_s_test, preds))
score = f1_score(y_s_test, preds, average="macro")
print(score)

disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test, preds, normalize=None), display_labels=rvb_labels
).plot()

# %%
disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test, preds, normalize="true"), display_labels=rvb_labels
).plot()

# %%
import pickle

with open("window_y_test.pkl", "wb") as f: 
    pickle.dump(y_s_test, f)
with open("window_preds.pkl", "wb") as f: 
    pickle.dump(preds, f)

# with open("time_slices.pkl", "rb") as f: 
#     time_slices = pickle.load(f)
# with open("slice_names.pkl", "rb") as f: 
#     slice_names = pickle.load(f)

# %%
y_s_test != preds

# %%
(y_s_test != preds) & (y_s_test == 4)

# %%
np.logical_and(y_s_test != preds, y_s_test == 4)

# %%
box = (y_s_test == 4) & (preds == 2)
y_s_test.index[box][~y_s_test.index[box].duplicated()]

# %%
box = (y_s_test == 2) & (preds == 4)
y_s_test.index[box][~y_s_test.index[box].duplicated()]

# %%
X_s_test.loc[["10.132.33.3",'192.168.55.99']]

# %%
sns.histplot(

# %%
window_model_window_preds = preds

# %%
pred_df = pd.Series(preds, index=y_s_test.index)

# %%
pred_df[~pred_df.index.duplicated(keep=False)]

# %%
dupe_pred_df = pred_df[pred_df.index.duplicated(keep=False)]

# %%
dupe_pred_df.index.unique()

# %%
dupe_ips = dupe_pred_df.index.unique()
pred_arrays = list()
for ip in dupe_ips:
    pred_arrays.append(pred_df.loc[ip])
    # pred_arrays_df = pd.Series([np.array([pred_df.loc[ip]]) if isinstance(pred_df.loc[ip], int) else pred_df.loc[ip].to_numpy() for ip in ips], index=ips)

# %%
pred_arrays[3].value_counts().index[0]


# %%
def label_value_count(value_count):
    if len(value_count) == 1:
        return value_count.index[0]
    # elif value_count.index[0] == 0 or value_count.index[0] == 1:
    else:
        return value_count.index[1]       
    


# %%
undupe_preds = list()
for pred_array in pred_arrays:
    undupe_preds.append(label_value_count(pred_array.value_counts()))

# %%
undupe_preds_df = pd.Series(undupe_preds, index=dupe_ips)

# %%
merged_preds = pd.concat([pred_df[~pred_df.index.duplicated(keep=False)], undupe_preds_df])

# %%
merged_true = label_df["red_vs_blue"][merged_preds.index]

# %%
disp = ConfusionMatrixDisplay(
        confusion_matrix(merged_true, merged_preds, normalize="true"), display_labels=rvb_labels
).plot()

# %%
pred_df.loc["10.60.60.44"].to_numpy()

# %%
np.unique(pred_df.loc["10.60.60.44"].to_numpy(), return_counts=True)

# %%
np.unique(X_s.index.duplicated(), return_counts=True)

# %%
slice_dfs[0][slice_dfs[0].index.isin(X_train.index)]

# %%
y_train

# %%
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# create model instance
# rf = DecisionTreeClassifier(class_weight=class_weights)
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
# rf.fit(X_train, y_train)

# fit model
# time_clf = XGBClassifier()
# sample_weights = class_weight.compute_sample_weight(class_weights, y=y_s)
# time_clf.fit(X_s, y_s, sample_weight=sample_weights)

# make predictions
train_preds = full_clf.predict(X_s)
preds = full_clf.predict(X_s_test)

print(classification_report(y_s, train_preds))
print(classification_report(y_s_test, preds))
score = f1_score(y_s_test, preds, average="macro")
print(score)

disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test, preds, normalize=None), display_labels=rvb_labels
).plot()

# %%
537 / (1 + 1 + 204 + 537 + 263)

# %%
disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test, preds, normalize="true"), display_labels=rvb_labels
).plot()

# %%
full_model_window_preds = preds

# %%
disp = ConfusionMatrixDisplay(
        confusion_matrix(full_model_window_preds, window_model_window_preds), display_labels=rvb_labels
).plot()

# %%
model_agreement = full_model_window_preds == window_model_window_preds

disp = ConfusionMatrixDisplay(
        confusion_matrix(y_s_test[model_agreement], full_model_window_preds[model_agreement]), display_labels=rvb_labels
).plot()


# %%
len(times_test)

# %%
len(window_model_window_preds)

# %%
len(y_s_test)

# %%
y_s_test > 1

# %%
times_test_s = pd.Series(times_test, index=y_s_test.index)

# %%
pd.DataFrame({"time": times_test_s[y_s_test > 1], "preds": (y_s_test == window_model_window_preds)[y_s_test > 1]}).groupby("time").mean()

# %%
acc_over_time = [
    0.942857,
    0.921569,
    0.937500,
    0.942623,
    0.902439,
    0.952381,
    0.938462,
    0.899225,
    0.923077,
    0.877358,
    0.905405,
    0.898305,
    0.914530,
    0.932203,
    0.926829,
    0.949495,
    0.901786,
    0.914286,
    0.93877,
	0.913978,]

# %%
sns.lineplot(acc_over_time)

# %%
import time
import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import StratifiedKFold

time_accs = list()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):

    times, X_s, y_s = build_tt(slice_dfs, slice_labels, labels[train].index)
    times_test, X_s_test, y_s_test = build_tt(slice_dfs, slice_labels, labels[test].index)

    # fit model
    time_clf = XGBClassifier()
    sample_weights = class_weight.compute_sample_weight(class_weights, y=y_s)
    time_clf.fit(X_s, y_s, sample_weight=sample_weights)
    
    # make predictions
    train_preds = time_clf.predict(X_s)
    preds = time_clf.predict(X_s_test)
    
    # print(classification_report(y_s, train_preds))
    print(classification_report(y_s_test, preds))
    score = f1_score(y_s_test, preds, average="macro")
    print(score)
    
    disp = ConfusionMatrixDisplay(
            confusion_matrix(y_s_test, preds, normalize=None), display_labels=rvb_labels
    ).plot()
    
    # # fit model
    # rf = XGBClassifier()
    # rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # # make predictions
    # train_preds = rf.predict(feature_arr[train])
    # preds = rf.predict(feature_arr[test])

    # # print(classification_report(labels[train], train_preds))
    # # print(classification_report(labels[test], preds))
    # score = f1_score(labels[test], preds, average="macro")
    # print(score)

    save_vars = [y_s, pd.Series(train_preds, index=y_s.index), 
                 y_s_test, pd.Series(preds, index=y_s_test.index), 
                 pd.Series(times, index=y_s.index), pd.Series(times_test, index=y_s_test.index),]
    filenames = ["y_train", "pred_train", "y_test", "pred_test", "times_train", "times_test"]

    # for filename, save_var in zip(filenames, save_vars):
        # save_var.to_csv("./paper-results/" + "window_" + "xgb_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

    times_test_s = pd.Series(times_test, index=y_s_test.index)
    time_accs.append(pd.DataFrame({"time": times_test_s[y_s_test > 1], f"preds{i}": (y_s_test == preds)[y_s_test > 1]}).groupby("time").mean())
    # start_time = time.time()
    # importances = rf.feature_importances_
    # # std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    # elapsed_time = time.time() - start_time

    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    # forest_importances = pd.Series(importances, index=feature_df.columns)

    # imps_inds = forest_importances.argsort()
    # # std_sorted = std[imps_inds]

    # with pd.option_context('display.max_rows', None):
    #     print(forest_importances.sort_values(ascending=False)[:10])

# %%
acc_over_time = pd.concat(time_accs, axis=1)
# acc_over_time['mean'] = acc_over_time.mean(axis=1)
# acc_over_time['std'] = acc_over_time.std(axis=1)
acc_over_time['order'] = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 11, 20, 12, 13, 14, 15, 16, 17, 18, 19]
acc_over_time = acc_over_time.sort_values("order")
# acc_over_time = acc_over_time.drop("order", axis=1)

# %%
sns.lineplot(data=acc_over_time)

# %%
sns.lineplot(data=pd.melt(acc_over_time, id_vars="order"), x='order', y='value')
plt.axvline(10.5, 0, 1, color='red')
# redo for full model on windows

# %%
import time
import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import StratifiedKFold

time_accs_full = list()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):

    times, X_s, y_s = build_tt(slice_dfs, slice_labels, labels[train].index)
    times_test, X_s_test, y_s_test = build_tt(slice_dfs, slice_labels, labels[test].index)

    # fit model
    # time_clf = XGBClassifier()
    # sample_weights = class_weight.compute_sample_weight(class_weights, y=y_s)
    # time_clf.fit(X_s, y_s, sample_weight=sample_weights)
    
    # make predictions
    train_preds = full_clf.predict(X_s)
    preds = full_clf.predict(X_s_test)
    
    # print(classification_report(y_s, train_preds))
    print(classification_report(y_s_test, preds))
    score = f1_score(y_s_test, preds, average="macro")
    print(score)
    
    disp = ConfusionMatrixDisplay(
            confusion_matrix(y_s_test, preds, normalize=None), display_labels=rvb_labels
    ).plot()

    save_vars = [y_s, pd.Series(train_preds, index=y_s.index), 
                 y_s_test, pd.Series(preds, index=y_s_test.index), 
                 pd.Series(times, index=y_s.index), pd.Series(times_test, index=y_s_test.index),]
    filenames = ["y_train", "pred_train", "y_test", "pred_test", "times_train", "times_test"]

    for filename, save_var in zip(filenames, save_vars):
        save_var.to_csv("./paper-results/" + "fullwindow_" + "xgb_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

    times_test_s = pd.Series(times_test, index=y_s_test.index)
    time_accs_full.append(pd.DataFrame({"time": times_test_s[y_s_test > 1], f"preds{i}": (y_s_test == preds)[y_s_test > 1]}).groupby("time").mean())


# %%
acc_over_time_full = pd.concat(time_accs_full, axis=1)

acc_over_time_full['order'] = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9, 11, 20, 12, 13, 14, 15, 16, 17, 18, 19]
acc_over_time_full = acc_over_time_full.sort_values("order")

sns.lineplot(data=pd.melt(acc_over_time_full, id_vars="order"), x='order', y='value')
plt.axvline(10.5, 0, 1, color='red')

# %%
acc_over_time

# %%
acc_over_time.to_csv("./paper-results/" + "comparison_dyn_trained.csv", mode='w', header=True)
acc_over_time_full.to_csv("./paper-results/" + "comparison_stat_trained.csv", mode='w', header=True)

# %%
plt.axhline(1.00, 0, 1, color='blue', label="Static Performance")
sns.lineplot(data=pd.melt(acc_over_time, id_vars="order"), x='order', y='value', label='Dyn. Trained')
sns.lineplot(data=pd.melt(acc_over_time_full, id_vars="order"), x='order', y='value', label='Static Trained')
plt.axvline(10.5, 0, 1, color='red', ls='--', label='Exercise Break')
plt.xlabel("Hour")
plt.ylabel("Accuracy")
plt.title("Comparison of accuracy over time on dynamic dataset")
plt.legend()
plt.savefig("./plots/dynamic_comparison.pdf")

# results reflect some sort of rhythm of the exercise
# -> comment: is this just an exercise artifact
# -> counter: this is refelective of real/dynamic human behavioural patterns
# this needs to be captured and understood by the model, so the model needs to be able
# to adjust to regime/seasonal shifts in behaviours over the course of the day, 
# not captured in the features aggregated across the full dataset 
# Can we demonstrate that the accuracy drops again if we further shrink the timescale
# Provide evidence that there's a characterstic time scale of the behaviour in this data
# Dataset is quite defined, red hosts should really only be red or inactive, less
# noisy compared to labels in the real world

# %%
def compare_features(slice_labels, slice_dfs, class_a, class_b):
    a_ips = [ip_labels[ip_labels == class_a].index for ip_labels in slice_labels]
    b_ips = [ip_labels[ip_labels == class_b].index for ip_labels in slice_labels]

    a_features = [features.loc[ips] for ips, features in zip(a_ips, slice_dfs)]
    b_features = [features.loc[ips] for ips, features in zip(b_ips, slice_dfs)]
    
    a_feature = pd.concat([features for features in a_features])
    a_feature["label"] = class_a
    b_feature = pd.concat([features for features in b_features])
    b_feature["label"] = class_b
    
    feature_comp = pd.concat([a_feature, b_feature])

    feature_comp["label"] = feature_comp["label"].astype("category")
    return feature_comp


# %%
feature_comp = compare_features(slice_labels, slice_dfs, 2, 4)

with sns.color_palette("deep"):
    for col in slice_dfs[0].columns:
        g = sns.histplot(data=feature_comp, x=col, 
                         hue="label", 
                         log_scale=True, 
                         element='step', 
                         # multiple='fill',
                         stat="density", 
                         common_norm=False,
                        )
        g.set_yscale('log')
        plt.show()

# add normalisation - done
# highlight misclassified instances - some work

# %%
