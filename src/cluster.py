from sklearn.cluster import KMeans

def get_clusters(df, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df.win_percentage.values.reshape(-1,1))
    clusters = kmeans.labels_
    df['cluster'] = clusters
    strat_bins = df.groupby('cluster')['win_percentage'].max() + 0.001
    strat_bins = sorted(sorted(strat_bins.values)[:-1] + [0.0,1.0])

    # clusters = [round(x,3) for x in sorted(list(kmeans.cluster_centers_.reshape(1,-1)[0]) + [0.0,1.0])]
    return strat_bins