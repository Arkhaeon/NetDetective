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

# display(df1)
# display(df2)

df = df1.extend(df2)

# %%
# Sample:       startTime=1461342600&stopTime=1461342700&
# Full Day1:    startTime=1461333000&stopTime=1461418000&
# Full Day2:    startTime=1461418000&stopTime=1461455000&
# Day2 start    2016/04/23 23:00:00 1461418200
# Day2 end      2016/04/24 09:00:00 1461454200
# Day1 start    2016/04/22 23:30:00 1461333600
# Day1 end     2016/04/23 09:00:00 1461367800

# %%
public_ips = pl.read_csv("public_ips.csv", has_header=True)["source.ip"]

# %%
rvb_dict = {
    0: "ghost",
    1: "ddos",
    2: "red",
    3: "blue",
    4: "grey",
}

label_df = pd.read_csv("label_data.csv", header=0).set_index("ip")
# display(label_df)

# %%
label_df["red_vs_blue"].value_counts()

# %%
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
    pl.col(ip_col).str.parse_int(10).mean().alias("ip_octet_" + ip_col) / 255,
) for ip_col in li_ip_cols
), ())

li_agg_tuple = (pl.col("source.bytes").count().alias("sessions"),) + li_port_agg_tuple + li_num_agg_tuple + li_protocol_agg_tuple + li_ip_agg_tuple

# %%
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


# %%
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

# %%
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
        minmax_scale = ['source.port_system_std', 'source.port_user_std', 'destination.port_system_std', 'destination.port_user_std',]

        for col in session_scale:
            X.with_columns((pl.col(col) / pl.col("sessions")).alias(col))

        for col in minmax_scale:
            X.with_columns((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()))

        return X.drop(["sessions"],)


# %%
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


# %%
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

        X = src_df.join(dst_df, on="ip", how="outer", suffix="_inc").fill_nan(0).fill_null(0)

        return X #.drop(["sessions"],)


# %%
li_trans = LiSessionGraphNodeTransformer()
stat_trans = ProSessionGraphNodeTransformer()
lyu_trans = LyuSessionGraphNodeTransformer()

# %%
li_df = li_trans.fit_transform(df)
stat_df = stat_trans.fit_transform(df)
lyu_df = lyu_trans.fit_transform(df)

# %%
li_df.columns

# %%
print(li_df.shape)
print(stat_df.shape)
print(lyu_df.shape)

# %%
tdf = li_df.join(stat_df, on="ip").join(lyu_df, on="ip")

# %%
feature_df = tdf.to_pandas().set_index("ip")
display(feature_df)

# %%
feature_df.columns

# %%
import itertools
import swifter

def compute_features(row, column, direction):
    features = list()
    sessions = row[direction]
    if sessions <= 1:
        return [1, 1, 1, 1, 0, 0, 0, 0, 0]
    else:
        cum_sum = 0
        q0 = False
        q1 = False
        q2 = False
        q3 = False
        counts = list()
        for item in row[column]:
            cum_sum = cum_sum + item["counts"]
            counts.append(item["counts"])
            if (cum_sum / sessions > 0) and not q0:
                features.append(item["counts"] / sessions)
                q0 = True
            if (cum_sum / sessions > 0.25) and not q1:
                features.append(item["counts"] / sessions)
                q1 = True
            if (cum_sum / sessions > 0.5) and not q2:
                features.append(item["counts"] / sessions)
                q2 = True
            if (cum_sum / sessions > 0.75) and not q3:
                features.append(item["counts"] / sessions)
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


# %%
feature_df.columns

# %%
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

# %%
display(feature_df)


# %%
print(feature_df.columns)

# %%
# feature_df.to_pickle("feature_df.pkl")
feature_df = pd.read_pickle("feature_df.pkl")

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
rf = XGBClassifier()
rf.fit(X_train, y_train, sample_weight=weights_train)

# make predictions
train_preds = rf.predict(X_train)
preds = rf.predict(X_test)

print(classification_report(y_train, train_preds))
print(classification_report(y_test, preds))
score = f1_score(y_test, preds, average="macro")
print(score)



# %%
import time

import numpy as np

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):
    # rf = DecisionTreeClassifier(class_weight=class_weights)
    # rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
    # rf.fit(feature_arr[train], labels[train])

    # fit model
    rf = XGBClassifier()
    rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # make predictions
    train_preds = rf.predict(feature_arr[train])
    preds = rf.predict(feature_arr[test])

    # print(classification_report(labels[train], train_preds))
    # print(classification_report(labels[test], preds))
    score = f1_score(labels[test], preds, average="macro")
    print(score)

    save_vars = [pd.Series(labels[train], index=feature_df.index[train]), pd.Series(train_preds, index=feature_df.index[train]), pd.Series(labels[test], index=feature_df.index[test]), pd.Series(preds, index=feature_df.index[test])]
    filenames = ["y_train", "pred_train", "y_test", "pred_test"]

    # for filename, save_var in zip(filenames, save_vars):
        # save_var.to_csv("./paper-results/" + "full_" + "dt_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

    start_time = time.time()
    importances = rf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=feature_df.columns)

    imps_inds = forest_importances.argsort()
    # std_sorted = std[imps_inds]

    with pd.option_context('display.max_rows', None):
        print(forest_importances.sort_values(ascending=False)[:10])

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
rf = DecisionTreeClassifier(class_weight=class_weights)
param_grid = {
    'max_depth': [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, None]
}
clf = GridSearchCV(rf, param_grid, scoring='f1_macro', n_jobs=-1, return_train_score=True, cv=skf.split(feature_arr, labels))
# clf.fit(feature_arr, labels)
clf.fit(feature_arr, labels,)

dt_eval_df = pd.DataFrame(clf.cv_results_)
dt_eval_df = pd.concat([eval_df, eval_df["params"].apply(pd.Series)], axis=1)
dt_eval_df["test_vs_train"] = eval_df["mean_test_score"] - eval_df["mean_train_score"]
dt_scoring_df = eval_df[["max_depth", "mean_test_score", "mean_train_score", "test_vs_train"]]

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
# rf = DecisionTreeClassifier(class_weight=class_weights)
param_grid = {
    'max_depth': [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, None]
}
clf = GridSearchCV(rf, param_grid, scoring='f1_macro', n_jobs=-1, return_train_score=True, cv=skf.split(feature_arr, labels))
# clf.fit(feature_arr, labels)
clf.fit(feature_arr, labels,)

rf_eval_df = pd.DataFrame(clf.cv_results_)
rf_eval_df = pd.concat([eval_df, eval_df["params"].apply(pd.Series)], axis=1)
rf_eval_df["test_vs_train"] = eval_df["mean_test_score"] - eval_df["mean_train_score"]
rf_scoring_df = eval_df[["max_depth", "mean_test_score", "mean_train_score", "test_vs_train"]]

# %%
from sklearn.model_selection import GridSearchCV, StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
# rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
rf = XGBClassifier()
param_grid = {
    'max_depth': [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 0]
}
clf = GridSearchCV(rf, param_grid, scoring='f1_macro', n_jobs=-1, return_train_score=True, cv=skf.split(feature_arr, labels))
# clf.fit(feature_arr, labels)
clf.fit(feature_arr, labels, sample_weight=sample_weights)

xgb_eval_df = pd.DataFrame(clf.cv_results_)
xgb_eval_df = pd.concat([eval_df, eval_df["params"].apply(pd.Series)], axis=1)
xgb_eval_df["test_vs_train"] = eval_df["mean_test_score"] - eval_df["mean_train_score"]
xgb_scoring_df = eval_df[["max_depth", "mean_test_score", "mean_train_score", "test_vs_train"]]

# %%
dt_scoring_df.to_csv("paper-results/depth_dt.csv", mode='w', header=True)
rf_scoring_df.to_csv("paper-results/depth_rf.csv", mode='w', header=True)
xgb_scoring_df.to_csv("paper-results/depth_xgb.csv", mode='w', header=True)

# %%
stat_feats = [
'source.bytes_mean',
'source.bytes_var',
'destination.bytes_mean',
'destination.bytes_var',
'source.packets_mean',
'source.packets_var',
'destination.packets_mean',
'destination.packets_var',
'network.bytes_mean',
'network.bytes_var',
'network.bytes_mean_inc',
'network.bytes_var_inc',
'duration_mean',
'duration_var',
'sessions',
'sessions_inc',
]

prot_feats = [
'ipProtocol_unique',
'ipProtocol_mode',
'ipProtocol_minor_fraction',
'ipProtocol_mode_inc',
'ipProtocol_minor_fraction_inc',
]

coarse_feats = [
'source.port_system_unique',
'source.port_system_std',
'source.port_system_most_count',
'source.port_user_unique',
'source.port_user_std',
'source.port_user_most_count',
'source.port_dynamic',
'destination.port_system_unique',
'destination.port_system_std',
'destination.port_system_most_count',
'destination.port_user_unique',
'destination.port_user_std',
'destination.port_user_most_count',
'destination.port_dynamic',
'host_lt_other',
]

fine_feats = [
'port_intext_ratio',
'port_intext_ratio_inc',
'source_featuresfracq0',
'source_featuresfracq1',
'source_featuresfracq2',
'source_featuresfracq3',
'source_inc_featuresfracq0',
'source_inc_featuresfracq1',
'source_inc_featuresfracq2',
'source_inc_featuresfracq3',
'destination_featuresfracq0',
'destination_featuresfracq1',
'destination_featuresfracq2',
'destination_featuresfracq3',
'destination_inc_featuresfracq0',
'destination_inc_featuresfracq1',
'destination_inc_featuresfracq2',
'destination_inc_featuresfracq3',
'source_featuresvar',
'source_inc_featuresvar',
'destination_featuresvar',
'destination_inc_featuresvar',
'source_featuresabvavg',
'source_featuresabvavg1s',
'source_featuresabvavg2s',
'source_featuresabvavg3s',
'source_inc_featuresabvavg',
'source_inc_featuresabvavg1s',
'source_inc_featuresabvavg2s',
'source_inc_featuresabvavg3s',
'destination_featuresabvavg',
'destination_featuresabvavg1s',
'destination_featuresabvavg2s',
'destination_featuresabvavg3s',
'destination_inc_featuresabvavg',
'destination_inc_featuresabvavg1s',
'destination_inc_featuresabvavg2s',
'destination_inc_featuresabvavg3s',
'source_top5_1',
'source_top5_2',
'source_top5_3',
'source_top5_4',
'source_top5_5',
'source_inc_top5_1',
'source_inc_top5_2',
'source_inc_top5_3',
'source_inc_top5_4',
'source_inc_top5_5',
'destination_top5_1',
'destination_top5_2',
'destination_top5_3',
'destination_top5_4',
'destination_top5_5',
'destination_inc_top5_1',
'destination_inc_top5_2',
'destination_inc_top5_3',
'destination_inc_top5_4',
'destination_inc_top5_5',
]

# %%
stat_df = feature_df.drop(prot_feats, axis=1).drop(coarse_feats, axis=1).drop(fine_feats, axis=1)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = stat_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):
    # rf = DecisionTreeClassifier(class_weight=class_weights)
    # rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
    # rf.fit(feature_arr[train], labels[train])

    # fit model
    rf = XGBClassifier()
    rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # make predictions
    train_preds = rf.predict(feature_arr[train])
    preds = rf.predict(feature_arr[test])

    print(classification_report(labels[train], train_preds))
    print(classification_report(labels[test], preds))
    score = f1_score(labels[test], preds, average="macro")
    print(score)

    save_vars = [pd.Series(labels[train], index=feature_df.index[train]), pd.Series(train_preds, index=feature_df.index[train]), pd.Series(labels[test], index=feature_df.index[test]), pd.Series(preds, index=feature_df.index[test])]
    filenames = ["y_train", "pred_train", "y_test", "pred_test"]

    for filename, save_var in zip(filenames, save_vars):
        save_var.to_csv("./paper-results/" + "x_" + "xgb_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

# %%
stat_df.columns

# %%
cut_df = feature_df.drop(coarse_feats, axis=1).drop(fine_feats, axis=1)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = cut_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):
    # rf = DecisionTreeClassifier(class_weight=class_weights)
    # rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
    # rf.fit(feature_arr[train], labels[train])

    # fit model
    rf = XGBClassifier()
    rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # make predictions
    train_preds = rf.predict(feature_arr[train])
    preds = rf.predict(feature_arr[test])

    print(classification_report(labels[train], train_preds))
    print(classification_report(labels[test], preds))
    score = f1_score(labels[test], preds, average="macro")
    print(score)

    save_vars = [pd.Series(labels[train], index=feature_df.index[train]), pd.Series(train_preds, index=feature_df.index[train]), pd.Series(labels[test], index=feature_df.index[test]), pd.Series(preds, index=feature_df.index[test])]
    filenames = ["y_train", "pred_train", "y_test", "pred_test"]

    for filename, save_var in zip(filenames, save_vars):
        save_var.to_csv("./paper-results/" + "s_" + "xgb_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

# %%
coarse_df = feature_df.drop(fine_feats, axis=1)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = coarse_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):
    # rf = DecisionTreeClassifier(class_weight=class_weights)
    # rf = RandomForestClassifier(class_weight=class_weights, n_jobs=-1)
    # rf.fit(feature_arr[train], labels[train])

    # fit model
    rf = XGBClassifier()
    rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # make predictions
    train_preds = rf.predict(feature_arr[train])
    preds = rf.predict(feature_arr[test])

    print(classification_report(labels[train], train_preds))
    print(classification_report(labels[test], preds))
    score = f1_score(labels[test], preds, average="macro")
    print(score)

    save_vars = [pd.Series(labels[train], index=feature_df.index[train]), pd.Series(train_preds, index=feature_df.index[train]), pd.Series(labels[test], index=feature_df.index[test]), pd.Series(preds, index=feature_df.index[test])]
    filenames = ["y_train", "pred_train", "y_test", "pred_test"]

    for filename, save_var in zip(filenames, save_vars):
        save_var.to_csv("./paper-results/" + "sc_" + "xgb_" + "fold_" + str(i) + "_" + filename, mode='w', header=False)

# %%
import time

import numpy as np

start_time = time.time()
importances = rf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=feature_df.columns)

imps_inds = forest_importances.argsort()
# std_sorted = std[imps_inds]

with pd.option_context('display.max_rows', None):
    print(forest_importances.sort_values(ascending=False)[:10])

# %%
complete_df = pd.concat([feature_df, labels], axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
import umap
import glasbey

classes = [2, 3, 4]
# stat_df = feature_df.drop(prot_feats, axis=1).drop(coarse_feats, axis=1).drop(fine_feats, axis=1)
reducer = umap.UMAP(metric="cosine")
embedding = reducer.fit_transform(complete_df.query("red_vs_blue in @classes").drop("red_vs_blue", axis=1))
class_labels = complete_df.query("red_vs_blue in @classes")["red_vs_blue"].map({2: "red", 3:"blue", 4: "grey"})

sns.scatterplot(
    x=embedding[:, 0],
    y=embedding[:, 1],
    hue=class_labels,
    alpha=0.2,
    palette=glasbey.create_palette(palette_size=len(np.unique(class_labels)))[::-1],
)
plt.gca().set_aspect("equal", "datalim")
plt.title("UMAP projection of the red and grey classes", fontsize=12)
plt.legend()

# %%
label_df["facility"].unique()

# %%
display(label_df[label_df["facility"] == "Confidential Data Storage"]["hostname"].unique())

# %%
import swifter

def facility_label(row):
    if row["red_vs_blue"] == 0:
        return 0  # "ghost"
    if row["red_vs_blue"] == 1:
        return 1  # "ghost"
    if row["facility"] == "Corporate":
        return 2
    if row["facility"] == "Confidential Data Storage":
        return 3
    if row["facility"] == "Cyprus Creek Campus":
        return 4
    else:
        return 5


rvb_dict = {
    0: "unresp",
    1: "ignored",
    2: "Corporate",
    3: "Confidential Data Storage",
    4: "Cyprus Creek Campus",
    5: "Other",
}

label_df["facility_class"] = label_df.swifter.allow_dask_on_strings(enable=True).apply(lambda row: facility_label(row), axis=1)

# label_df.to_csv("label_data.csv", mode="w", header=True, index_label="ip")

# %%
# read data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight

labels = label_df["facility_class"][feature_df.index]

class_weights = dict(enumerate(class_weight.compute_class_weight(
    "balanced", classes=np.unique(labels), y=labels
)))
print(class_weights)

sample_weights = class_weight.compute_sample_weight(class_weights, y=labels)
print(sample_weights)

# %%
import time
import itertools

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC, LinearSVC
import numpy as np

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
feature_arr = feature_df.to_numpy()
for i, (train, test) in enumerate(skf.split(feature_arr, labels)):
    # fit model
    rf = XGBClassifier()
    rf.fit(feature_arr[train], labels[train], sample_weight=sample_weights[train])

    # make predictions
    train_preds = rf.predict(feature_arr[train])
    preds = rf.predict(feature_arr[test])

    print(classification_report(labels[train], train_preds))
    print(classification_report(labels[test], preds))
    score = f1_score(labels[test], preds, average="macro")
    print(score)
