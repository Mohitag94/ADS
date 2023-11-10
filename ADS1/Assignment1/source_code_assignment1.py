# https://www.kaggle.com/datasets/reenapinto/real-estate-sales-2001-2020
"""
7PAM2000 Applied Data Science 1
Assignment 1: Visualisation
------------------------------------------------------------------------------
Mohit Agarwal (Student ID-22031257)

The Real Estate Dataset for the assignment was taken from 
    https://www.kaggle.com/datasets/reenapinto/real-estate-sales-2001-2020.

Real Estate Dataset Description :
The dataset contains details of the properties sold for 
a price geater then 2000 USD, occuring in between October 1 and September 30, 
from 2001 to 2020, maintained by the Office of Policy and Management.

For each sale recorded, followings items are maintained in the dataset:
- Town
- Property address
- Date of Sales
- Property Type
    - Residental 
    - Apartment 
    - Commerical 
    - Industrial 
    - Vacant land
- Sales Price
- Property Assessment

Total number of entries in the datasets is (Rows: 997213, Columns: 14)

Note: The description is paraphased.
"""


# importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def eda_df(property_sales_df):
    """
    eda_df fucntion for subseting the data, after looking at the data's stats
    """

    # looking at the columns types and null values in it
    print("[INFO] Seeing the columns types and null value:\n",
          property_sales_df.info())
    # looking at the stats of the dataset
    print("[INFO] Seeing the stats of the datasets:\n",
          property_sales_df.describe())

    """
    Looking at the data, we can subset it to create another dataframe,
    as the ""Serial Number"", ""Address"" columns are useless in terms of 
    ploting for this assignment and the last four columns, namely, 
    (""Non Use Code"", ""Assessor Remarks"", ""OPM remarks"", 
    ""Location"") don't provide any informations, additionally, 
    these columns as the most ""NaN"" values. Finally the recoreded
    Date is not of use for this assignment.
    """

    # creating a subset of dataframe from the read file
    property_sales_subset_df = property_sales_df[["List Year",
                                                  "Town", "Assessed Value",
                                                  "Sale Amount",
                                                  "Sales Ratio",
                                                  "Property Type",
                                                  "Residential Type"]]

    # printing the first few lines of the dataframe created
    print("[INFO] A new subset dataframe created.\n\tThe new dataframe:\n",
          property_sales_subset_df.head())

    return property_sales_subset_df
# end of eda function.


def sales_property_type(property_sales_subset_df):
    """
    sales_property_type function for ploting the time-series for
    various residential type over the years
    """

    # creating a sub plot to capture the avg sales per residential type
    avg_sold_resident_df = property_sales_subset_df.pivot_table(
        values="Sale Amount",
        index="List Year",
        columns="Residential Type",
        fill_value=0)

    # ploting lines graphs from the above pivoted dataframe
    plt.figure(figsize=(12, 8), dpi=100)
    # loop for running over the columns
    for i, _ in enumerate(avg_sold_resident_df.columns):
        # ploting the graph for each type
        plt.plot(avg_sold_resident_df.iloc[:, i]/1000,
                 label=avg_sold_resident_df.columns[i])
    # giving a title to the graph
    plt.title("Average Cost of Residential Property Over the Years")
    # adding x-axis label
    plt.xlabel("List Year")
    # adding y-axis label
    plt.ylabel("Sale Amount in Thousands (USD)")
    # showing the legend
    plt.legend(title="Residential Type")
    # making the layout tight
    plt.tight_layout()
    # saving the plot in the local working directory
    plt.savefig("avg_cost_resident.png")
    # displaying the plot
    plt.show()

# end of sales_property_type function


def total_sales(property_sales_subset_df):
    """
    total_sales function for calculating the total sales per town per 
    residential type and ploting in a stacked plot
    """

    # creating a sub-dataframe, caputring the total sales per town per type
    total_town_residential_sale_df = property_sales_subset_df.groupby(
        ["Town", "Residential Type"]).agg({"Sale Amount": "count"})

    # droping the first row, as the town name is unknown
    total_town_residential_sale_df.drop(
        total_town_residential_sale_df.index[0], inplace=True)

    # sorting the multi-index dataframe and unstacking it
    town_resident_count_df = total_town_residential_sale_df.sort_values(
        "Sale Amount", ascending=False).unstack(fill_value=0)

    # creating a list of all residential type
    residential = ["Condo", "Four Family",
                   "Single Family", "Three Family", "Two Family"]

    # ploting total number residential sold per town in the bar plot

    # for stacking the various residential type
    bottom = np.zeros(40)
    # number of towns being plotted
    town = 40

    # initializing the figure
    plt.figure(figsize=(12, 8), dpi=100)
    # looping over the resifential
    for i in range(len(residential)):
        # ploting graph for each residential type
        plt.bar(town_resident_count_df.index[:town],
                town_resident_count_df.iloc[:, i][:town], bottom=bottom)
        # increasing the bottom, indicating the placemnet for next stack
        bottom += town_resident_count_df.iloc[:, i][:town]

    # adding the title
    plt.title("Total Number of Residential Property Sold Per Town")
    # rotating the towns names vertically
    plt.xticks(rotation=90)
    # adding th x-axis
    plt.xlabel("Towns")
    # adding the y-axis
    plt.ylabel("Number of Residential Property")
    # putting the legend
    plt.legend(residential, title="Residential Type")
    # making the layout tight
    plt.tight_layout()
    # saving the plot in the local working directory
    plt.savefig("total_residential_sold.png")
    # displaying the graph
    plt.show()

# end of total_sales function


def market_cap(property_sales_subset_df):
    """
    market_cap function calculates the overal market capture
    by the various property type and shows it through the pie graph 
    """

    # calculating the amount of property sold from 2001-2006
    first_five_property_count_df = property_sales_subset_df[
        (property_sales_subset_df["List Year"] >= 2001) &
        (property_sales_subset_df["List Year"] <= 2007)].groupby(
        ["Property Type"]).agg({"Sale Amount": "count"})
    # renaming the sale amount col to count
    first_five_property_count_df.rename(
        columns={"Sale Amount": "Count"}, inplace=True)
    # sorting the dataframe
    first_five_property_count_df.sort_values(
        "Count", ascending=False, inplace=True)
    # calculating the percentages
    percetage_first_five = 100 * \
        first_five_property_count_df["Count"]\
        / first_five_property_count_df["Count"].sum()
    labels_first_five = ['{0} - {1:1.2f} %'.format(i, j)
                         for i, j in zip(first_five_property_count_df.index,
                                         percetage_first_five)]

    # calculating the amount of property sold from 2015-2020
    last_five_property_count_df = property_sales_subset_df[
        (property_sales_subset_df["List Year"] >= 2015) &
        (property_sales_subset_df["List Year"] <= 2020)].groupby(
        ["Property Type"]).agg({"Sale Amount": "count"})
    # renaming the sale amount col to count
    last_five_property_count_df.rename(
        columns={"Sale Amount": "Count"}, inplace=True)
    # sorting the dataframe
    last_five_property_count_df.sort_values(
        "Count", ascending=False, inplace=True)
    # calculating the percentages
    percetage_last_five = 100 * \
        last_five_property_count_df["Count"]\
        / last_five_property_count_df["Count"].sum()
    # creating the label array for legend
    labels_last_five = ['{0} - {1:1.2f} %'.format(i, j)
                        for i, j in zip(last_five_property_count_df.index,
                                        percetage_last_five)]

    # choosing the color for the pie chart
    colors = ["yellowgreen", "red", "gold", "lightskyblue",
              "lightcoral", "blue", "pink", "darkgreen",
              "lightgrey", "violet", "magenta"]

    # ploting the market capture with the calculated values

    # intializing the chart
    fig = plt.figure(figsize=(12, 8))

    # ploting for first fives in left
    ax_first = fig.add_subplot(2, 1, 1)
    ax_first.pie(first_five_property_count_df["Count"], startangle=90,
                 colors=colors[:len(first_five_property_count_df)])
    # adding the legend to the chart
    ax_first.legend(labels_first_five, loc="center left",
                    bbox_to_anchor=(-0.65, 0.5),
                    fancybox=True, shadow=True, fontsize=8,
                    title="Property Type")
    # setting the title
    ax_first.set_title("First Five Years[2001-2006]")

    # ploting for last five in right
    ax_last = fig.add_subplot(2, 1, 2)
    ax_last.pie(last_five_property_count_df["Count"], startangle=90)
    # adding the legend to the chart
    ax_last.legend(labels_last_five, loc="center right",
                   bbox_to_anchor=(1.65, 0.5),
                   fancybox=True, shadow=True, fontsize=8,
                   title="Property Type")
    # setting the title
    ax_last.set_title("Last Five Years[2015-2020]")

    # adding the title to the entire figure/chart
    plt.suptitle("Market Captured By Property Type",
                 fontsize="xx-large", fontweight="medium")
    # saving the chart to the drive
    plt.savefig("market_cap.png")
    # displaying the chart
    plt.show()

# end of the market_cap function


def main():
    """ 
    Main Function to read the file from the drive and 
    call other functions for various tasks
    """

    print("[INFO] 7PAM2000 APPLIED DATA SCIENCE ASSIGNMENT-1 \n\t\t \
        BY MOHIT AGARWAL[22031257]\n")
    # reading the dataset from the drive
    print("[INFO] READING DATASET\n\t\
        [NOTE] Huge File, Will take Time to READ...")
    property_sales_df = pd.read_excel("Real_Estate_Sales_2001-2020_GL.xlsx")
    print("\tRead.\n\tThe first few lines of dataset:\n",
          property_sales_df.head())

    # cleaning the dataset
    print("[INFO] Looking at the Data and Cleaning(EDA)...")
    property_sales_subset_df = eda_df(property_sales_df)

    # calling the line plot function for avg sale amount per type over years
    print("[INFO] Creating a line plot...")
    sales_property_type(property_sales_subset_df)
    print("\tPlot Created and Saved in the drive.")

    # calling the bar plot to showing the total number sales per town
    print("[INFO] Creating a bar plot...")
    total_sales(property_sales_subset_df)
    print("\tPlot Created and Saved in the drive")

    # calling the pie plot to showing the market cap
    print("[INFO] Creating a pie plot...")
    market_cap(property_sales_subset_df)
    print("\tPlot Created and Saved in the drive")

    # ending the assignment
    print("[INFO] The Three Plot Has Been Created.\n\t\
        BY MOHIT AGARWAL[22031257]")

# end of the main function


"""
Will only call the main function when executed as script, 
not imported as package.
"""

if __name__ == "__main__":
    main()

# end of if-condition
