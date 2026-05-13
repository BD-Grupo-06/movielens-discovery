# Cluster Interpretation Report

## Data Source Note
Rating statistics and one-hot features loaded from Week 5 PCA feature matrix (`week05_pca_feature_matrix.parquet`).

## Cluster Overview
|   cluster |   count |   percentage |
|----------:|--------:|-------------:|
|         0 |   15273 |        24.47 |
|         1 |   32747 |        52.46 |
|         2 |    8837 |        14.16 |
|         3 |    5566 |         8.92 |

## Rating Statistics (denormalized) - Differences from Global Mean
|   cluster |   rating_mean |   rating_count |   rating_std |
|----------:|--------------:|---------------:|-------------:|
|         0 |        -0.045 |        540.327 |        0.057 |
|         1 |        -0.109 |       -235.144 |       -0.061 |
|         2 |         0.154 |        148.05  |        0.141 |
|         3 |         0.521 |       -334.258 |       -0.025 |

## Top 5 Genres per Cluster
### Cluster 0 top genres
- genre_drama: 0.511
- genre_thriller: 0.493
- genre_crime: 0.276
- genre_horror: 0.264
- genre_action: 0.262

### Cluster 1 top genres
- genre_drama: 0.470
- genre_comedy: 0.336
- genre_romance: 0.154
- genre_horror: 0.053
- genre_action: 0.049

### Cluster 2 top genres
- genre_comedy: 0.416
- genre_animation: 0.319
- genre_adventure: 0.304
- genre_children: 0.301
- genre_drama: 0.244

### Cluster 3 top genres
- genre_documentary: 0.996
- genre_drama: 0.046
- genre_comedy: 0.033
- genre_musical: 0.020
- genre_war: 0.020

## Top 5 Tags per Cluster
### Cluster 0 top tags
- tag_000_bd_r: 0.140
- tag_002_murder: 0.135
- tag_005_nudity_topless: 0.073
- tag_011_violence: 0.069
- tag_006_based_on_a_book: 0.056

### Cluster 1 top tags
- tag_001_woman_director: 0.071
- tag_003_independent_film: 0.040
- tag_008_drama: 0.032
- tag_000_bd_r: 0.027
- tag_009_romance: 0.023

### Cluster 2 top tags
- tag_014_musical: 0.108
- tag_000_bd_r: 0.088
- tag_004_comedy: 0.061
- tag_010_funny: 0.047
- tag_006_based_on_a_book: 0.031

### Cluster 3 top tags
- tag_001_woman_director: 0.124
- tag_000_bd_r: 0.029
- tag_003_independent_film: 0.017
- tag_015_criterion: 0.011
- tag_002_murder: 0.007

## Top 5 Distinctive Genres per Cluster
### Cluster 0 distinctive genres
- genre_thriller: 0.354
- genre_crime: 0.191
- genre_horror: 0.168
- genre_action: 0.144
- genre_mystery: 0.133

### Cluster 1 distinctive genres
- genre_comedy: 0.066
- genre_drama: 0.060
- genre_romance: 0.030
- genre_war: 0.012
- genre_western: 0.006

### Cluster 2 distinctive genres
- genre_animation: 0.272
- genre_children: 0.254
- genre_adventure: 0.238
- genre_fantasy: 0.198
- genre_comedy: 0.146

### Cluster 3 distinctive genres
- genre_documentary: 0.906
- genre_imax: 0.004
- genre_musical: 0.003
- genre_film_noir: -0.006
- genre_war: -0.010

## Top 5 Distinctive Tags per Cluster
### Cluster 0 distinctive tags
- tag_002_murder: 0.099
- tag_000_bd_r: 0.077
- tag_011_violence: 0.051
- tag_005_nudity_topless: 0.049
- tag_013_revenge: 0.037

### Cluster 1 distinctive tags
- tag_001_woman_director: 0.015
- tag_003_independent_film: 0.011
- tag_008_drama: 0.011
- tag_017_love: 0.004
- tag_009_romance: 0.002

### Cluster 2 distinctive tags
- tag_014_musical: 0.090
- tag_004_comedy: 0.035
- tag_010_funny: 0.028
- tag_000_bd_r: 0.025
- tag_018_family: 0.014

### Cluster 3 distinctive tags
- tag_001_woman_director: 0.068
- tag_015_criterion: -0.007
- tag_018_family: -0.010
- tag_003_independent_film: -0.012
- tag_014_musical: -0.012

## Representative Movies (random & by rating)
### Cluster 0 sample movies (random)
- Movie 950
- Movie 153032
- Movie 5689
- Movie 133569
- Movie 91243
- Movie 58033
- Movie 202159
- Movie 127246
- Movie 8262
- Movie 184755

### Cluster 0 sample movies (closest to mean rating)
- Movie 3184 (rating: 3.23)
- Movie 153 (rating: 3.23)
- Movie 3697 (rating: 3.23)
- Movie 55492 (rating: 3.23)
- Movie 44189 (rating: 3.23)
- Movie 4707 (rating: 3.23)
- Movie 80891 (rating: 3.23)
- Movie 86572 (rating: 3.23)
- Movie 134517 (rating: 3.23)
- Movie 58975 (rating: 3.23)

### Cluster 1 sample movies (random)
- Movie 144846
- Movie 180697
- Movie 188249
- Movie 198931
- Movie 136568
- Movie 87860
- Movie 156857
- Movie 200792
- Movie 131900
- Movie 165467

### Cluster 1 sample movies (closest to mean rating)
- Movie 5449 (rating: 3.17)
- Movie 6012 (rating: 3.17)
- Movie 3642 (rating: 3.17)
- Movie 6154 (rating: 3.17)
- Movie 57942 (rating: 3.17)
- Movie 117932 (rating: 3.17)
- Movie 141692 (rating: 3.17)
- Movie 144418 (rating: 3.17)
- Movie 169198 (rating: 3.17)
- Movie 174503 (rating: 3.17)
- Movie 187085 (rating: 3.17)

### Cluster 2 sample movies (random)
- Movie 127078
- Movie 132356
- Movie 164765
- Movie 175753
- Movie 134684
- Movie 153570
- Movie 70643
- Movie 86846
- Movie 157995
- Movie 4039

### Cluster 2 sample movies (closest to mean rating)
- Movie 143387 (rating: 3.43)
- Movie 184239 (rating: 3.43)
- Movie 5396 (rating: 3.43)
- Movie 4228 (rating: 3.43)
- Movie 6907 (rating: 3.43)
- Movie 4161 (rating: 3.43)
- Movie 101025 (rating: 3.43)
- Movie 1489 (rating: 3.43)
- Movie 278 (rating: 3.43)
- Movie 172421 (rating: 3.43)

### Cluster 3 sample movies (random)
- Movie 65188
- Movie 163989
- Movie 158663
- Movie 152908
- Movie 180023
- Movie 208297
- Movie 106156
- Movie 165571
- Movie 2494
- Movie 140591

### Cluster 3 sample movies (closest to mean rating)
- Movie 143249 (rating: 3.80)
- Movie 188299 (rating: 3.79)
- Movie 4208 (rating: 3.80)
- Movie 2934 (rating: 3.79)
- Movie 113632 (rating: 3.80)
- Movie 1901 (rating: 3.79)
- Movie 65001 (rating: 3.79)
- Movie 68489 (rating: 3.79)
- Movie 80582 (rating: 3.79)
- Movie 96724 (rating: 3.79)
- Movie 104033 (rating: 3.79)
- Movie 104376 (rating: 3.79)
- Movie 113938 (rating: 3.79)
- Movie 138038 (rating: 3.79)
- Movie 148974 (rating: 3.79)
- Movie 149739 (rating: 3.79)
- Movie 163314 (rating: 3.79)
- Movie 171047 (rating: 3.79)
- Movie 183875 (rating: 3.79)
- Movie 194402 (rating: 3.79)

## Final Cluster Labels (heuristic)
| cluster | label | description |
|---:|---|---|
| 0 | Thriller / Crime / 002 Murder / 000 Bd R | Cluster 0: genres=['genre_thriller', 'genre_crime']; tags=['tag_002_murder', 'tag_000_bd_r']. Mean rating=3.23, count=941. |
| 1 | Comedy / Drama / 001 Woman Director / 003 Independent Film / Lower-Rated | Cluster 1: genres=['genre_comedy', 'genre_drama']; tags=['tag_001_woman_director', 'tag_003_independent_film']. Mean rating=3.17, count=165. |
| 2 | Animation / Children / 014 Musical / 004 Comedy / High-Rated | Cluster 2: genres=['genre_animation', 'genre_children']; tags=['tag_014_musical', 'tag_004_comedy']. Mean rating=3.43, count=549. |
| 3 | Documentary / Imax / 001 Woman Director / 015 Criterion / High-Rated | Cluster 3: genres=['genre_documentary', 'genre_imax']; tags=['tag_001_woman_director', 'tag_015_criterion']. Mean rating=3.80, count=66. |