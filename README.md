# gofee_modified
Modifcation to gofee code

I modify gofee code (https://gitlab.au.dk/au480665/gofee/-/blob/master/gofee/gofee.py) to make it a little more efficient. 


How to use this modified code:
```
from gofee.gofee_modified import GOFEE
```

I add two following input variables. 

```
estd_thr: float
        Default: 0
        Threshold for surrogate energy uncertainty. 
        If the the surrogate energy uncertainty (Energy_std) < estd_thr, DFT evaluation and surrogate train will be skipped. 
        The evaluation of the best structure will be done using the surrogate model. "
```

``` 
old_trajectory: string
        Default: None
        "If old_trajectory and restart file are present, the surrogate model from the previous run will be used as the starting model of the current run.
        Please use a different name from trajectory file, e.g. 'old_structures.traj'. "
```
