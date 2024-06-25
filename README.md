# gplearn-for-finance  

Inspired by [gplearn](https://github.com/trevorstephens/gplearn), this package makes some tweak to allow fitting of panel financial data, which typically takes shape of (n_samples, n_features, n_days, n_firms).  

This package is still in developing.

## To-do:  

1. Support for IS / OS performance stats tracking;  
2. Add time-series / cross-sectional function;  
3. Implement elitism and other similar improvement;  
4. Support for portfolio combination within a population, given some kind of performance threshold.  
5. Improve repair algorithm of raw portfolio weight;  
6. Maybe add support for classification? multi-label could come in handy.