
'''
Class Info
Class |  Mask      | Gender   | Age             | Sample cnt
0     |  Wear      | Male     | <30             | 2315
1     |  Wear      | Male     | >=30 and < 60   | 1700
2     |  Wear      | Male     | >= 60           | 275 *
3     |  Wear      | Female   | <30             | 3015
4     |  Wear      | Female   | >=30 and < 60   | 3365
5     |  Wear      | Female   | >= 60           | 390 *
6     |  Incorrect | Male     | <30             | 463
7     |  Incorrect | Male     | >=30 and < 60   | 340
8     |  Incorrect | Male     | >= 60           | 55 *
9     |  Incorrect | Female   | <30             | 603
10    |  Incorrect | Female   | >=30 and < 60   | 673
11    |  Incorrect | Female   | >= 60           | 78 *
12    |  Not wear  | Male     | <30             | 463
13    |  Not wear  | Male     | >= 30 and < 60  | 340
14    |  Not wear  | Male     | >= 60           | 755 *
15    |  Not wear  | Female   | <30             | 603 
16    |  Not wear  | Female   | >=30 and < 60   | 673
17    |  Not wear  | Female   | >= 60           | 18 *

MASK = 0 , INCORRECT = 1, NORMAL = 2
MALE = 0 , FEMALE = 1,
YONG = 0, MIDDLE = 1, OLD =2 
2 5 8 11 14 17

'''

'''
EXP 4 

Undersample :[0,1,3,4]
Oversample : [8,11,17]

Class 0 accuracy 94.4
Class 1 accuracy 81.14
Class 2 accuracy 27.24
Class 3 accuracy 95.64
Class 4 accuracy 92.9
Class 5 accuracy 32.88
Class 6 accuracy 85.21
Class 7 accuracy 72.38
Class 8 accuracy 50.6
Class 9 accuracy 93.14
Class 10 accuracy 91.41
Class 11 accuracy 46.56
Class 12 accuracy 89.11
Class 13 accuracy 79.26
Class 14 accuracy 15.77
Class 15 accuracy 94.75
Class 16 accuracy 92.84
Class 17 accuracy 44.68

Exp 5

Undersample :[0,3,4]
Oversample : [2,5,8,11,14,17]

wandb:  Class 0 accuracy 87.93
wandb:  Class 1 accuracy 87.6
wandb:  Class 2 accuracy 37.51
wandb:  Class 3 accuracy 96.11
wandb:  Class 4 accuracy 89.93
wandb:  Class 5 accuracy 45.34
wandb:  Class 6 accuracy 80.74
wandb:  Class 7 accuracy 76.02
wandb:  Class 8 accuracy 44.7
wandb:  Class 9 accuracy 90.93
wandb: Class 10 accuracy 89.98
wandb: Class 11 accuracy 47.31
wandb: Class 12 accuracy 83.51
wandb: Class 13 accuracy 77.76
wandb: Class 14 accuracy 37.92
wandb: Class 15 accuracy 93.15
wandb: Class 16 accuracy 93.08
wandb: Class 17 accuracy 47.26

Exp 6

Undersample :[0,3,4]
Oversample : [2,5,8,11,14,17]

wandb:  Class 0 accuracy 82.34
wandb:  Class 1 accuracy 86.98
wandb:  Class 2 accuracy 40.87
wandb:  Class 3 accuracy 96.37
wandb:  Class 4 accuracy 89.59
wandb:  Class 5 accuracy 44.02
wandb:  Class 6 accuracy 78.26
wandb:  Class 7 accuracy 75.45
wandb:  Class 8 accuracy 48.04
wandb:  Class 9 accuracy 90.0
wandb: Class 10 accuracy 90.07
wandb: Class 11 accuracy 46.61
wandb: Class 12 accuracy 82.83
wandb: Class 13 accuracy 78.1
wandb: Class 14 accuracy 42.86
wandb: Class 15 accuracy 93.72
wandb: Class 16 accuracy 91.7
wandb: Class 17 accuracy 45.16


Exp 10
wandb:  Class 0 accuracy 95.29
wandb:  Class 1 accuracy 84.5
wandb: Class 10 accuracy 89.86
wandb: Class 11 accuracy 51.34
wandb: Class 12 accuracy 94.26
wandb: Class 13 accuracy 79.43
wandb: Class 14 accuracy 41.61
wandb: Class 15 accuracy 97.66
wandb: Class 16 accuracy 92.97
wandb: Class 17 accuracy 50.0
wandb:  Class 2 accuracy 41.73
wandb:  Class 3 accuracy 96.22
wandb:  Class 4 accuracy 89.17
wandb:  Class 5 accuracy 50.99
wandb:  Class 6 accuracy 90.08
wandb:  Class 7 accuracy 77.52
wandb:  Class 8 accuracy 45.89
wandb:  Class 9 accuracy 96.09
'''