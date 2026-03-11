# functions use in the visualize_scores.ipynb notebook

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def merge_list_results(results_list, names_list):
    df_list = []
    for i in range(len(results_list)):
        df = pd.DataFrame(results_list[i])[["id", "RAG_answer", "rating"]]
        df.rename(
            columns={
                "RAG_answer": f"RAG_answer_{names_list[i]}",
                "rating": f"rating_{names_list[i]}",
            },
            inplace=True,
        )
        df.set_index("id", inplace=True)
        df_list.append(df)
    merged = df_list[0]
    for i in range(1, len(df_list)):
        merged = merged.merge(df_list[i], how="inner", on="id")
    # filtering bad results
    for name in names_list:
        merged = merged[merged[f"rating_{name}"] <= 5]
    # print mean ratings
    for name in names_list:
        print(f"Average rating for {name}: {merged[f'rating_{name}'].mean()}")
    return merged


def print_bar_plot_merged_df(merged):
    colums_name_ratings = [col for col in merged.columns if "rating" in col]
    df_melted = merged.melt(
        value_vars=colums_name_ratings, var_name="Model", value_name="Rating"
    )

    # Filter valid ratings (1-5)
    df_melted = df_melted[df_melted["Rating"].between(1, 5)]

    # Count occurrences of each rating for each model
    df_counts = df_melted.groupby(["Rating", "Model"]).size().reset_index(name="Count")

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Rating", y="Count", hue="Model", data=df_counts, palette="Set2")

    # Customization
    plt.xlabel("Rating (1-5)")
    plt.ylabel("Count")
    plt.title("Distribution of Ratings for Each Model")
    plt.legend(title="Model")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()
