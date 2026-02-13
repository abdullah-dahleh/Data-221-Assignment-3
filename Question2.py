import pandas as pd
import matplotlib.pyplot as plt

#Load the crime dataset
crime_data_frame = pd.read_csv("crime1.csv")

#Select the target column
violent_crime_rate = crime_data_frame["ViolentCrimesPerPop"]

#Create histogram
plt.figure()
plt.hist(violent_crime_rate, bins=20)
plt.title("Histogram of Violent Crimes Per Population")
plt.xlabel("Violent Crimes Per Population")
plt.ylabel("Frequency")
plt.show()

#Create boxplot
plt.figure()
plt.boxplot(violent_crime_rate)
plt.title("Boxplot of Violent Crimes Per Population")
plt.ylabel("Violent Crimes Per Population")
plt.show()

#The histogram shows how the crime rates are distributed.
#The boxplot shows the median and possible outliers.