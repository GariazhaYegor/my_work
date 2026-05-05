# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("steam.csv")

# %%
# Top 5 genres
genres = df["genres"].value_counts().head(5)
print(genres)

# %%
# Top 5 most rated games
top_games = df.nlargest(5, "positive_ratings")[["name", "positive_ratings"]]
print(top_games)

# %%
# Chart 1 - Top 5 Genres
plt.figure()
plt.bar(genres.index, genres.values, color="steelblue")
plt.title("Top 5 Genres")
plt.ylabel("Number of games")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# %%
# Chart 2 - Most rated games
plt.figure()
plt.barh(top_games["name"], top_games["positive_ratings"], color="green")
plt.title("Top 5 Most Rated Games")
plt.xlabel("Positive ratings")
plt.tight_layout()
plt.show()
# %%
print("Hello")