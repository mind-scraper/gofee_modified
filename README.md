# gofee_modified
Modifcation to gofee code

I modify gofee code (https://gitlab.au.dk/au480665/gofee/-/blob/master/gofee/gofee.py) to make it a little more efficient. 

I add two following input variable. 

estd_tress: float
  Default: 0
  Tresshold for surrogate energy uncertainty. 
  If the the surrogate energy uncertainty (Energy_std) < estd_tress, DFT evaluation and surrogate train will be skipped. 
  
old_trajectory: string
  Default: None
  If old_trajectory and restart file are present, the surrogate model from the previous run will be used as the starting model of the current run. 
