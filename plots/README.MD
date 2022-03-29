# Sigma and Clip Norm effect on Dataset: Overview

### CIFAR10

| Sigma | Clip Norm | Plateau @ | Epochs intended to run for | Accuracy Train | Accuracy Val | Comments                                |   
|-------|-----------|-----------|----------------------------|----------------|--------------|-----------------------------------------|
| 0     | 0.1       | No        | 100                        | ~70            | ~68          ||
| 0     | 1.0       | 60        | 100                        | ~100           | ~76          ||
| 0     | 3.0       | 35        | 100                        | ~100           | ~70          ||
| 0     | 5.0       | 45        | 100                        | ~100           | ~65          ||
| 0     | 10        | No        | 100                        | ~64            | ~60          | But the learning curve is not smooth    |
| 1.0   | 0.1       | No        | 100                        | ~70            | ~68          ||
| 1.0   | 1.0       | 20        | 100                        | <20            | <20          | Abrupt rise and then fall in accuracy   |
| 1.0   | 3.0       | 20        | 100                        | ~12            | ~12          | Consistent fall in accuracy             |
| 1.0   | 5.0       | 20        | 100                        | ~10            | ~10          | Consistent fall in accuracy             |
| 1.0   | 10.0      | No        | 100                        | ~14            | ~14          | Not a smooth learning curve             |
| 1.5   | 0.1       | No        | 100                        | ~65            | ~63          ||
| 1.5   | 1.0       | 20        | 100                        | ~7             | ~7           | Steep decline in accuracy               |
| 1.5   | 3.0       | 20        | 100                        | ~12            | ~12          | Steep decline in accuracy               |
| 1.5   | 5.0       | 20        | 100                        | ~10            | ~9           ||
| 1.5   | 10.0      | 70        | 100                        | ~12.5          | ~12.5        | Irregular training curve                |
| 2.0   | 0.1       | 70        | 100                        | ~57            | ~57          |                                         |
| 2.0   | 1.0       | 20        | 100                        | ~10            | ~10          | Steep decline in accuracy               |
| 2.0   | 3.0       | 20        | 100                        | ~10            | ~10          | Steep decline and irregular curve       |
| 2.0   | 5.0       | 20        | 100                        | ~10            | ~10          |                                         |
| 2.0   | 10.0      | 70        | 100                        | ~13            | ~13          | Irregular learning curves               |
| 3.0   | 0.1       | 30        | 100                        | ~47            | ~47          |                                         |
| 3.0   | 1.0       | 20        | 100                        | ~10            | ~10          | Steep decline in learning curve         |
| 3.0   | 3.0       | 40        | 100                        | ~10.5          | ~10.5        | Irregular learning curve                |
| 3.0   | 5.0       | 30        | 100                        | ~10.5          | ~10.5        | Irregular learning curve                |
| 3.0   | 10.0      | No        | 100                        | ~13            | ~13          | Accuracy keeps oscillating between 12-14 |

** ~=approx.

### Fmnist
| Sigma | Clip Norm | Plateau @ | Epochs intended to run for | Accuracy Train | Accuracy Val | Comments                                                                                                                                                                             |   
|-------|-----------|-----------|----------------------------|----------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | 0.1       | No        | 100                        | ~90            | ~89          ||
| 0     | 1.0       | 80        | 100                        | ~90            | ~87          ||
| 0     | 3.0       | 20        | 100                        | ~10            | 10           | Val accuracy stays constantly at 10                                                                                                                                                  |
| 0     | 5.0       | 20        | 100                        | ~10            | 10           | Val accuracy stays constantly at 10                                                                                                                                                  |
| 0     | 10        | 20        | 100                        | ~9.7           | ~10          | Val accuracy stays constantly at 10                                                                                                                                                  |
| 1.0   | 0.1       | No        | 100                        | ~90            | ~89          |                                                                                                                                                                                      |                                                                                                                                                        |
| 1.0   | 1.0       | 70        | 100                        | ~85            | ~84          | Slight irregularities in the learning curve                                                                                                                                          |
| 1.0   | 3.0       | 50        | 100                        | ~78            | ~78          | Not a smooth learning curve                                                                                                                                                          |
| 1.0   | 5.0       | 20        | 100                        | ~10            | 10           | Val accuracy constantly stays at 10                                                                                                                                                  |
| 1.0   | 10.0      | 50        | 100                        | ~40            | ~40          | The learning curve doesn't necessarily shows a plateau but val accuracy hits over 50% accuracy once and in the subsequent epochs the accuracy doesn't increase,hence the early stop. |
| 1.5   | 0.1       | No        | 100                        | ~90            | ~88          ||
| 1.5   | 1.0       | 35        | 100                        | ~83            | ~80          |                                                                                                                                                                                      |                                                                                                                                                         |
| 1.5   | 3.0       | 45        | 100                        | ~70            | ~70          | Irregular training curve                                                                                                                                                             |                                                                                                                                                          |
| 1.5   | 5.0       | 90        | 100                        | ~50            | ~50          | Very irregular learning curve                                                                                                                                                        |                                                                                                                                                      |
| 1.5   | 10.0      | 80        | 100                        | ~35            | ~35          | Very irregular learning curve                                                                                                                                                        |
| 2.0   | 0.1       | No        | 100                        | ~89            | ~88          |                                                                                                                                                                                      |
| 2.0   | 1.0       | 50        | 100                        | ~80            | ~76          |                                                                                                                                                                                      |
| 2.0   | 3.0       | 80        | 100                        | ~50            | ~50          | Very irregular learning curves                                                                                                                                                       |
| 2.0   | 5.0       | No        | 100                        | ~50            | ~50          | Very Irregular learning curves                                                                                                                                                       |
| 2.0   | 10.0      | 80        | 100                        | ~40            | ~40          | Very Irregular learning curves                                                                                                                                                       |
| 3.0   | 0.1       | 90        | 100                        | ~85            | ~84          |                                                                                                                                                                                      |
| 3.0   | 1.0       | 35        | 100                        | ~72            | ~72          |                                                                                                                                                                                      |
| 3.0   | 3.0       | 55        | 100                        | ~55            | ~55          | Irregular learning curves                                                                                                                                                            |
| 3.0   | 5.0       | 70        | 100                        | ~40            | ~35          | Very irregular learning curves                                                                                                                                                       |
| 3.0   | 10.0      | 35        | 100                        | ~35            | ~30          | Very Irregular learning curves                                                                                                                                                       |


### MNIST
| Sigma | Clip Norm | Plateau @ | Epochs intended to run for | Accuracy Train | Accuracy Val | Comments                                                                             |   
|-------|-----------|-----------|----------------------------|----------------|--------------|--------------------------------------------------------------------------------------|
| 0     | 0.1       | No        | 100                        | ~97            | ~97          |                                                                                      |
| 0     | 1.0       | 60        | 100                        | ~99            | ~98          |                                                                                      |
| 0     | 3.0       | 50        | 100                        | ~100           | ~98.5        |                                                                                      |
| 0     | 5.0       | 35        | 100                        | ~100           | ~98.5        |                                                                                      | 
| 0     | 10        | 35        | 100                        | ~100           | ~98.5        |                                                                                      |
| 1.0   | 0.1       | No        | 100                        | ~99            | ~99          |                                                                                      |
| 1.0   | 1.0       | 20        | 100                        | ~94            | ~94          |                                                                                      |
| 1.0   | 3.0       | 20        | 100                        | ~87            | ~87          | Very irregular learning curve                                                        |
| 1.0   | 5.0       | 20        | 100                        | ~81            | ~82          |                                                                                      |
| 1.0   | 10.0      | 20        | 100                        | ~60            | ~60          | Irregular learning curve and declines rapidly                                        |
| 1.5   | 0.1       | 80        | 100                        | ~98            | ~98          ||
| 1.5   | 1.0       | 20        | 100                        | ~91            | ~91          |                                                                                      |                                                                                                                                                         |
| 1.5   | 3.0       | 20        | 100                        | ~77            | ~77          | Irregular training curve                                                             |                                                                                                                                                          |
| 1.5   | 5.0       | 20        | 100                        | ~65            | ~68          | Very irregular learning curve                                                        |                                                                                                                                                      |
| 1.5   | 10.0      | 20        | 100                        | ~37.5          | ~37.5        | Very irregular learning curve                                                        |
| 2.0   | 0.1       | 60        | 100                        | ~95            | ~95          |                                                                                      |
| 2.0   | 1.0       | 20        | 100                        | ~85            | ~85          |                                                                                      |
| 2.0   | 3.0       | 20        | 100                        | ~64            | ~65          | Very irregular learning curves                                                       |
| 2.0   | 5.0       | 20        | 100                        | ~46            | ~46          | Irregular learning curves + Decline rapidly                                          |
| 2.0   | 10.0      | 20        | 100                        | ~31            | ~32          | Highly Irregular learning curves                                                     |
| 3.0   | 0.1       | 50        | 100                        | ~95            | ~95          |                                                                                      |
| 3.0   | 1.0       | 20        | 100                        | ~75            | ~75          | Irregular learning curves + Decline rapidly                                          |
| 3.0   | 3.0       | 35        | 100                        | ~41            | ~41          | Irregular learning curves                                                            |
| 3.0   | 5.0       | 25        | 100                        | ~34            | ~34          | Very irregular learning curves                                                       |
| 3.0   | 10.0      | 40        | 100                        | ~24            | ~24          | Hits a peak accuracy of >30% at 20th epoch and then doesn't improve hence early stop |


