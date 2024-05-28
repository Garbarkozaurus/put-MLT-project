import pandas as pd
import matplotlib.pyplot as plt

PENGUINS_PATH = "./data/penguins.csv"


def plot_attr_grouped_and_subgrouped(
    df: pd.DataFrame, attr_name: str, grouping_attr: str, subgrouping_attr: str
) -> None:
    group_names = pd.unique(df[grouping_attr])
    subgroup_names = pd.unique(df[subgrouping_attr])
    GROUP_OFFSET = 0.5
    SUBGROUP_OFFSET = 0.15
    label_coords = []
    label_values = []
    for i, group_name in enumerate(group_names):
        group_df = df[df[grouping_attr]==group_name][[attr_name, subgrouping_attr]]
        means = group_df.groupby(subgrouping_attr)[attr_name].mean()
        s_devs = group_df.groupby(subgrouping_attr)[attr_name].std()
        xs = [i*GROUP_OFFSET+j*SUBGROUP_OFFSET for j in range(len(subgroup_names))]
        label_coords += xs
        label_values += list(means.index)
        plt.errorbar(xs,means,s_devs, linestyle="None", marker="o", label=group_name, capsize=2)
    plt.legend()
    plt.xticks(label_coords,label_values)
    plt.grid(axis="y")
    plt.title(attr_name)
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(PENGUINS_PATH)
    # print(len(df))
    df = df.dropna()
    # print(len(df))
    # print(pd.value_counts(df["species"]))
    # plot_attr_grouped_and_subgrouped(df, "body_mass_g", "species", "sex")
    plot_attr_grouped_and_subgrouped(df, "bill_length_mm", "species", "sex")
    # plot_attr_grouped_and_subgrouped(df, "bill_depth_mm", "species", "sex")
    # plot_attr_grouped_and_subgrouped(df, "flipper_length_mm", "species", "sex")
    # plot_attr_grouped_and_subgrouped(df, "flipper_length_mm", "sex", "species")
