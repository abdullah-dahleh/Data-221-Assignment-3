import pandas as pd

#Load the crime dataset
crime_data_frame = pd.read_csv("crime1.csv")

#Select the target column
violent_crime_rate = crime_data_frame["ViolentCrimesPerPop"]

#Calculate descriptive statistics
mean_crime_rate = violent_crime_rate.mean()
median_crime_rate = violent_crime_rate.median()
standard_deviation_crime_rate = violent_crime_rate.std()
minimum_crime_rate = violent_crime_rate.min()
maximum_crime_rate = violent_crime_rate.max()

#Print results
print("Mean:", mean_crime_rate)
print("Median:", median_crime_rate)
print("Standard Deviation:", standard_deviation_crime_rate)
print("Minimum:", minimum_crime_rate)
print("Maximum:", maximum_crime_rate)

#If the mean is larger than the median, the data is likely right-skewed.
#The mean is more affected by extreme values than the median.