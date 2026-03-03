from recommender import recommend

results = recommend("Java developer with teamwork skills")

for r in results:
    print(r)