# Cluster Interpretation Report

## Data Source Note
Rating statistics and one-hot features loaded from Week 5 PCA feature matrix (`week05_pca_feature_matrix.parquet`).

## Cluster Overview
|   cluster |   count |   percentage |
|----------:|--------:|-------------:|
|         0 |    9364 |        15    |
|         1 |    2995 |         4.8  |
|         2 |    5362 |         8.59 |
|         3 |   10458 |        16.75 |
|         4 |   25183 |        40.34 |
|         5 |    5497 |         8.81 |
|         6 |    3564 |         5.71 |

## Rating Statistics (denormalized) - Differences from Global Mean
|   cluster |   rating_mean |   rating_count |   rating_std |
|----------:|--------------:|---------------:|-------------:|
|         0 |         0.521 |       1265.19  |        0.218 |
|         1 |        -0.184 |         46.429 |       -0.066 |
|         2 |        -0.055 |        -94.239 |       -0.073 |
|         3 |        -0.374 |         44.426 |        0.189 |
|         4 |        -0.157 |       -392.152 |       -0.156 |
|         5 |         0.52  |       -346.656 |       -0.029 |
|         6 |         0.271 |        -46.146 |        0.186 |

## Top 5 Genres per Cluster
### Cluster 0 top genres
- genre_drama: 0.613
- genre_comedy: 0.452
- genre_romance: 0.252
- genre_thriller: 0.073
- genre_adventure: 0.063

### Cluster 1 top genres
- genre_war: 0.541
- genre_drama: 0.503
- genre_western: 0.456
- genre_action: 0.236
- genre_romance: 0.135

### Cluster 2 top genres
- genre_crime: 0.874
- genre_drama: 0.592
- genre_thriller: 0.416
- genre_action: 0.253
- genre_mystery: 0.217

### Cluster 3 top genres
- genre_thriller: 0.458
- genre_horror: 0.433
- genre_action: 0.319
- genre_drama: 0.265
- genre_sci_fi: 0.260

### Cluster 4 top genres
- genre_drama: 0.470
- genre_comedy: 0.359
- genre_romance: 0.166
- genre_children: 0.040
- genre_action: 0.033

### Cluster 5 top genres
- genre_documentary: 0.999
- genre_drama: 0.043
- genre_comedy: 0.035
- genre_war: 0.021
- genre_musical: 0.015

### Cluster 6 top genres
- genre_animation: 0.780
- genre_children: 0.413
- genre_comedy: 0.319
- genre_adventure: 0.302
- genre_fantasy: 0.227

## Top 5 Tags per Cluster
### Cluster 0 top tags
- tag_000_bd_r: 0.209
- tag_007_clv: 0.124
- tag_010_funny: 0.100
- tag_003_independent_film: 0.098
- tag_005_nudity_topless: 0.098

### Cluster 1 top tags
- tag_000_bd_r: 0.145
- tag_011_violence: 0.029
- tag_006_based_on_a_book: 0.028
- tag_012_based_on_novel_or_book: 0.028
- tag_015_criterion: 0.027

### Cluster 2 top tags
- tag_002_murder: 0.134
- tag_000_bd_r: 0.089
- tag_013_revenge: 0.067
- tag_011_violence: 0.049
- tag_003_independent_film: 0.031

### Cluster 3 top tags
- tag_002_murder: 0.078
- tag_019_action: 0.050
- tag_011_violence: 0.044
- tag_000_bd_r: 0.038
- tag_006_based_on_a_book: 0.036

### Cluster 4 top tags
- tag_001_woman_director: 0.073
- tag_000_bd_r: 0.018
- tag_003_independent_film: 0.015
- tag_014_musical: 0.015
- tag_004_comedy: 0.014

### Cluster 5 top tags
- tag_001_woman_director: 0.125
- tag_000_bd_r: 0.029
- tag_003_independent_film: 0.011
- tag_018_family: 0.008
- tag_015_criterion: 0.008

### Cluster 6 top tags
- tag_014_musical: 0.046
- tag_010_funny: 0.036
- tag_004_comedy: 0.036
- tag_001_woman_director: 0.026
- tag_018_family: 0.026

## Top 5 Distinctive Genres per Cluster
### Cluster 0 distinctive genres
- genre_drama: 0.203
- genre_comedy: 0.182
- genre_romance: 0.128
- genre_musical: 0.040
- genre_fantasy: 0.011

### Cluster 1 distinctive genres
- genre_war: 0.511
- genre_western: 0.434
- genre_action: 0.118
- genre_drama: 0.093
- genre_adventure: 0.054

### Cluster 2 distinctive genres
- genre_crime: 0.789
- genre_thriller: 0.277
- genre_drama: 0.182
- genre_mystery: 0.170
- genre_action: 0.135

### Cluster 3 distinctive genres
- genre_horror: 0.337
- genre_thriller: 0.319
- genre_sci_fi: 0.202
- genre_action: 0.201
- genre_mystery: 0.068

### Cluster 4 distinctive genres
- genre_comedy: 0.089
- genre_drama: 0.060
- genre_romance: 0.042
- genre_imax: -0.003
- genre_film_noir: -0.006

### Cluster 5 distinctive genres
- genre_documentary: 0.909
- genre_imax: 0.001
- genre_musical: -0.002
- genre_film_noir: -0.006
- genre_war: -0.009

### Cluster 6 distinctive genres
- genre_animation: 0.733
- genre_children: 0.366
- genre_adventure: 0.236
- genre_fantasy: 0.183
- genre_sci_fi: 0.065

## Top 5 Distinctive Tags per Cluster
### Cluster 0 distinctive tags
- tag_000_bd_r: 0.146
- tag_007_clv: 0.102
- tag_010_funny: 0.081
- tag_005_nudity_topless: 0.074
- tag_003_independent_film: 0.069

### Cluster 1 distinctive tags
- tag_000_bd_r: 0.082
- tag_011_violence: 0.011
- tag_012_based_on_novel_or_book: 0.010
- tag_015_criterion: 0.009
- tag_016_betamax: 0.008

### Cluster 2 distinctive tags
- tag_002_murder: 0.098
- tag_013_revenge: 0.049
- tag_011_violence: 0.031
- tag_000_bd_r: 0.026
- tag_019_action: 0.010

### Cluster 3 distinctive tags
- tag_002_murder: 0.042
- tag_019_action: 0.034
- tag_011_violence: 0.026
- tag_006_based_on_a_book: 0.014
- tag_013_revenge: 0.013

### Cluster 4 distinctive tags
- tag_001_woman_director: 0.017
- tag_017_love: -0.003
- tag_014_musical: -0.003
- tag_018_family: -0.004
- tag_012_based_on_novel_or_book: -0.006

### Cluster 5 distinctive tags
- tag_001_woman_director: 0.069
- tag_018_family: -0.008
- tag_015_criterion: -0.010
- tag_014_musical: -0.013
- tag_010_funny: -0.015

### Cluster 6 distinctive tags
- tag_014_musical: 0.028
- tag_010_funny: 0.017
- tag_004_comedy: 0.010
- tag_018_family: 0.010
- tag_019_action: 0.003

## Representative Movies (random & by rating)
### Cluster 0 sample movies (random)
- Movie 2894
- Movie 6545
- Movie 112421
- Movie 158268
- Movie 86345
- Movie 467
- Movie 3432
- Movie 73881
- Movie 111113
- Movie 3035

### Cluster 0 sample movies (closest to mean rating)
- Movie 74275 (rating: 3.80)
- Movie 8899 (rating: 3.80)
- Movie 2128 (rating: 3.80)
- Movie 5243 (rating: 3.80)
- Movie 5540 (rating: 3.80)
- Movie 4601 (rating: 3.80)
- Movie 8272 (rating: 3.80)
- Movie 49098 (rating: 3.80)
- Movie 68271 (rating: 3.80)
- Movie 92152 (rating: 3.80)

### Cluster 1 sample movies (random)
- Movie 107664
- Movie 141172
- Movie 6082
- Movie 140433
- Movie 117867
- Movie 3753
- Movie 25809
- Movie 8683
- Movie 74799
- Movie 5456

### Cluster 1 sample movies (closest to mean rating)
- Movie 5207 (rating: 3.09)
- Movie 3025 (rating: 3.10)
- Movie 107352 (rating: 3.10)
- Movie 2055 (rating: 3.08)
- Movie 59834 (rating: 3.08)
- Movie 43419 (rating: 3.10)
- Movie 2328 (rating: 3.11)
- Movie 93193 (rating: 3.08)
- Movie 99380 (rating: 3.11)
- Movie 4463 (rating: 3.07)

### Cluster 2 sample movies (random)
- Movie 193886
- Movie 147256
- Movie 5827
- Movie 154925
- Movie 128732
- Movie 463
- Movie 170573
- Movie 139711
- Movie 186803
- Movie 159837

### Cluster 2 sample movies (closest to mean rating)
- Movie 3517 (rating: 3.22)
- Movie 89588 (rating: 3.22)
- Movie 199319 (rating: 3.22)
- Movie 147847 (rating: 3.22)
- Movie 8652 (rating: 3.22)
- Movie 72129 (rating: 3.22)
- Movie 105351 (rating: 3.22)
- Movie 74641 (rating: 3.22)
- Movie 97697 (rating: 3.22)
- Movie 132106 (rating: 3.23)

### Cluster 3 sample movies (random)
- Movie 140723
- Movie 198831
- Movie 177877
- Movie 160826
- Movie 173251
- Movie 187085
- Movie 169764
- Movie 66579
- Movie 135585
- Movie 81429

### Cluster 3 sample movies (closest to mean rating)
- Movie 2667 (rating: 2.90)
- Movie 53464 (rating: 2.90)
- Movie 89190 (rating: 2.90)
- Movie 8810 (rating: 2.90)
- Movie 5562 (rating: 2.90)
- Movie 34378 (rating: 2.90)
- Movie 5787 (rating: 2.90)
- Movie 167838 (rating: 2.90)
- Movie 190085 (rating: 2.90)
- Movie 5189 (rating: 2.91)

### Cluster 4 sample movies (random)
- Movie 95563
- Movie 169842
- Movie 152593
- Movie 194268
- Movie 162930
- Movie 139118
- Movie 160384
- Movie 101431
- Movie 180237
- Movie 196959

### Cluster 4 sample movies (closest to mean rating)
- Movie 83842 (rating: 3.12)
- Movie 111734 (rating: 3.12)
- Movie 142602 (rating: 3.12)
- Movie 5535 (rating: 3.12)
- Movie 127110 (rating: 3.12)
- Movie 174045 (rating: 3.12)
- Movie 184189 (rating: 3.12)
- Movie 117312 (rating: 3.12)
- Movie 67501 (rating: 3.12)
- Movie 70517 (rating: 3.12)
- Movie 110512 (rating: 3.12)
- Movie 153838 (rating: 3.12)
- Movie 166906 (rating: 3.12)
- Movie 171275 (rating: 3.12)

### Cluster 5 sample movies (random)
- Movie 120823
- Movie 72304
- Movie 120821
- Movie 94341
- Movie 200340
- Movie 173475
- Movie 45928
- Movie 201807
- Movie 196661
- Movie 97834

### Cluster 5 sample movies (closest to mean rating)
- Movie 143249 (rating: 3.80)
- Movie 188299 (rating: 3.79)
- Movie 4208 (rating: 3.80)
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

### Cluster 6 sample movies (random)
- Movie 182063
- Movie 6371
- Movie 165367
- Movie 141154
- Movie 188867
- Movie 112846
- Movie 7790
- Movie 6169
- Movie 142919
- Movie 118958

### Cluster 6 sample movies (closest to mean rating)
- Movie 5002 (rating: 3.55)
- Movie 8965 (rating: 3.55)
- Movie 141676 (rating: 3.55)
- Movie 170293 (rating: 3.55)
- Movie 175393 (rating: 3.55)
- Movie 204158 (rating: 3.55)
- Movie 8974 (rating: 3.54)
- Movie 170961 (rating: 3.55)
- Movie 95182 (rating: 3.55)
- Movie 86286 (rating: 3.55)

## Final Cluster Labels (heuristic)
| cluster | label | description |
|---:|---|---|
| 0 | Drama / Comedy / 000 Bd R / 007 Clv / High-Rated | Cluster 0: genres=['genre_drama', 'genre_comedy']; tags=['tag_000_bd_r', 'tag_007_clv']. Mean rating=3.80, count=1666. |
| 1 | War / Western / 000 Bd R / 011 Violence / Lower-Rated | Cluster 1: genres=['genre_war', 'genre_western']; tags=['tag_000_bd_r', 'tag_011_violence']. Mean rating=3.09, count=447. |
| 2 | Crime / Thriller / 002 Murder / 013 Revenge | Cluster 2: genres=['genre_crime', 'genre_thriller']; tags=['tag_002_murder', 'tag_013_revenge']. Mean rating=3.22, count=306. |
| 3 | Horror / Thriller / 002 Murder / 019 Action / Lower-Rated | Cluster 3: genres=['genre_horror', 'genre_thriller']; tags=['tag_002_murder', 'tag_019_action']. Mean rating=2.90, count=445. |
| 4 | Comedy / Drama / 001 Woman Director / 017 Love / Lower-Rated | Cluster 4: genres=['genre_comedy', 'genre_drama']; tags=['tag_001_woman_director', 'tag_017_love']. Mean rating=3.12, count=8. |
| 5 | Documentary / Imax / 001 Woman Director / 018 Family / High-Rated | Cluster 5: genres=['genre_documentary', 'genre_imax']; tags=['tag_001_woman_director', 'tag_018_family']. Mean rating=3.80, count=54. |
| 6 | Animation / Children / 014 Musical / 010 Funny / High-Rated | Cluster 6: genres=['genre_animation', 'genre_children']; tags=['tag_014_musical', 'tag_010_funny']. Mean rating=3.55, count=354. |