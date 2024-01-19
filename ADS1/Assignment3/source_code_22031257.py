"""
7PAM2000 Applied Data Science 1
Assignment 3: Clustering and Fitting
------------------------------------------------------------------------------
Mohit Agarwal (Student ID-22031257)

Exploring the impact of Economical Growth on Climate, using indicators
from World-Bank.

https://data.worldbank.org/indicator

The chosen indicators are:
1. GDP (current US$)
    -> https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
2. Forest area (% of land area) 
    -> https://data.worldbank.org/indicator/AG.LND.FRST.ZS
3. Access to electricity (% of population)
    -> https://data.worldbank.org/indicator/EG.ELC.ACCS.ZS
4. CO2 emissions (metric tons per capita)
    -> https://data.worldbank.org/indicator/EN.ATM.CO2E.PC
5. Manufacturing, value added (% of GDP)
    -> https://data.worldbank.org/indicator/NV.IND.MANF.ZS
6. Total greenhouse gas emissions (kt of CO2 equivalent)
    -> https://data.worldbank.org/indicator/EN.ATM.GHGT.KT.CE

The income-group, regional and country wise cluster and fitting is
constructed using kmeans method for cluster and ploynomial of order 2
is taken as function for fitting and forecasting for next 15 years.

Provided error python script file was used to calculate the confidence range.

The program will create three type of graphs, namely:-
1. The silhouette score graph for clustering.
2. The cluster plots.
3. The fitting plots with forecast for next 15 years.

Note: 
No plots/grahs are shown but all are saved directly in the working directory.
"""

# importing required packages...
import importlib as imlib
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import errors as err


# setting plotting style
plt.style.use("seaborn-v0_8-darkgrid")

# listing all the indicators filenames
indicator_filenames = ["API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv",
                       "API_AG.LND.FRST.ZS_DS2_en_csv_v2_6299844.csv",
                       "API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_6299951.csv",
                       "API_EN.ATM.CO2E.PC_DS2_en_csv_v2_6299932.csv",
                       "API_NV.IND.MANF.ZS_DS2_en_csv_v2_6299867.csv",
                       "API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_6299833.csv"]
# listing all the indicators metadata filenames
indicator_metadata_filenames = \
    ["Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_6298258.csv",
     "Metadata_Country_API_AG.LND.FRST.ZS_DS2_en_csv_v2_6299844.csv",
     "Metadata_Country_API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_6299951.csv",
     "Metadata_Country_API_EN.ATM.CO2E.PC_DS2_en_csv_v2_6299932.csv",
     "Metadata_Country_API_NV.IND.MANF.ZS_DS2_en_csv_v2_6299867.csv",
     "Metadata_Country_API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_6299833.csv"]

# listing the years interval: 1990-2020
years = [*map(str, np.arange(1990, 2021, 1))]

# indicator names list created
indicator_names = []

# uncomment if you want to product fitting for these countries
# # list of countries for analysis
# countries = ["Japan", "Germany", "Brazil", "Saudi Arabia",
#              "United States", "India", "South Africa",
#              "Sudan", "China"]

# list of country with China alone
countries = ["China"]

# setting cluster iteration run
init = 20


def curve_fit_plot(xdata, ydata, xforecast, yfit, sigma,
                   category, indicator, title):
    """
    curve_fit_plot function draws two lines plots, namely,
    - the actual data, 
    - the forecasted data 
    - the confidence ranges
    """

    # initializing the figure
    plt.figure(dpi=300)
    # ploting the actual data
    plt.plot(xdata, ydata, label=title)
    # plotting the forecasted data
    plt.plot(xforecast, yfit, label="Forecast")
    # plotting the confidence ranges
    plt.fill_between(xforecast, yfit-sigma, yfit+sigma,
                     color="yellow", alpha=0.8)
    # giving a title
    plt.title(f"{category}'s {title} Forecast",
              fontsize="large", fontweight="semibold")
    # adding ylabel
    plt.ylabel(indicator, fontsize="medium", fontweight="semibold")
    # adding xlabels
    plt.xlabel("Years", fontsize="medium", fontweight="semibold")
    # adding legend
    plt.legend()
    # saving the plot in the local drive
    plt.savefig(f"{category}_{title}_Forecast.png")
    # closing and clearing the image
    plt.clf()
    plt.close()
# end of curve_fit_plot function


def poly2d(x, a, b, c):
    """
    poly2d function is a polynominal function of order 2, 
    computes the values of y for corresponding x with a,b,c parameters.
    ands returns y
    """
    return a*x**2 + b*x + c
# end of poly2d function


def curve_fitting(xdata, ydata, xforecast, function, p0=None):
    """
    curve_fitting function finds the curve that fits the data, using 
    functions, like polynominal, exponential or logarithm and calulates
    the error deviation of the fit and correlation matrix between parameters,
    returns y values from forecast and error deviation
    """

    # relaoding the error file
    imlib.reload(err)
    # calculating the parameters and covariance matrix
    param, pcovar = curve_fit(function, xdata, ydata, p0)
    print(f"\t[INFO] Parameters are:\n{param}")
    print(f"\t[INFO] Covariance Matrix is:\n{pcovar}")
    # finding the y values for the forecast
    y_curve_fit = function(xforecast, *param)
    # finding the correlation matrix
    corr = err.covar_to_corr(pcovar)
    print(f"\t[INFO] Parameter Correlaton Matrix:\n{corr}")
    # find the error deviation
    sigma = err.error_prop(xforecast, function, param, pcovar)
    # returning the curve and sigma
    return y_curve_fit, sigma
# end of curve_fitting function


def curve_fit_main(df, indicator, title, category_list=countries):
    """curve_fit_main function find a curve to be fit on the data, 
    plots the data over the list of country, through calling the two methods,
    curve_fitting and curve_fit_plot
    """
    print(f"[INFO] {title}...")
    # looping over the list provided
    for lst in category_list:
        print(f"\t[INFO] {lst}...")
        # setting xdata
        xdata = df.loc[lst][df.loc[lst].notna()].index.astype("int")
        # setting ydata
        ydata = df.loc[lst][df.loc[lst].notna()].values
        # setting forecast data
        xforecast = np.arange(min(xdata), max(xdata)+16, 1)
        # calling the curve fitting function
        print("\t[INFO] Finding the curve...")
        y_fit, sigma = curve_fitting(xdata, ydata, xforecast, poly2d)
        # calling the curve_fit_plot functon
        print("\t[INFO] Plotting the curve...")
        curve_fit_plot(xdata, ydata, xforecast, y_fit, sigma,
                       lst, indicator, title)
        print("\tDone.")
    print("\t[INFO] Curved Found, Plotted and Saved.")
    print("\n------------------------------------------------\n")
# end curve_fit_main function


def cluster_plot(df, centers, labels, xlabel, ylabel, title):
    """
    cluster_plot function plots the clusters in a scatter plot on a 
    given axes, it also find the which region, income group or countries is
    present in each clusters. 

    Finally the plot is saved in the local drive
    """

    # adding the labels to dataframe
    df["Labels"] = labels
    # a dict for cluster names
    cluster_name_dict = {}
    # a dict for each cluster make-up
    each_cluster_name_dict = {}
    # cluster text
    cluster_text = ""
    # for loop in range with number of clusters
    for cluster_num in list(set(labels)):
        # naming each cluster
        c_text = f"Cluster {cluster_num}"
        cluster_name_dict[cluster_num] = c_text
        # a list to hold the each cluster make-up values
        c_list = []
        # for loop for each unique pair of index and labels
        for pair in list(set(sorted(zip(df.loc[:, "Labels"], df.index)))):
            # checking if the cluster num is present in the pair
            if pair[0] == cluster_num:
                # sorting cluster make-up value
                c_list.append(pair[1])
        # end of inner loop
        # storing all the values in cluster "i"
        each_cluster_name_dict[cluster_num] = c_list
        # creating a text list for make-up clusters
        cluster_text += f"{c_text} --> {c_list}\n"
    # end of outer loop
    # adding center to the cluster name
    cluster_name_dict[-1] = "Centers"
    print("\t[INFO] Cluster Make-Up...\n", cluster_text)
    # ploting
    # initializing the figure
    plt.figure(dpi=300)
    # plotting clusters
    ax1 = plt.scatter(x=df.iloc[:, 0], y=df.iloc[:, 1],
                      c=df.loc[:, "Labels"], s=15,
                      cmap=mpl.colormaps["Paired"], alpha=0.5,
                      label=np.array(list(cluster_name_dict.items()))[:, 1])
    # ploting the centers
    ax2 = plt.scatter(x=centers[:, 0], y=centers[:, 1], marker="*",
                      c=[-1]*len(centers), cmap="CMRmap", s=60)
    # getting the elements of legend ax1 for ploting cluster names
    handles = ax1.legend_elements()[0]
    # adding the center's elements
    handles.append(ax2.legend_elements()[0][0])
    # ploting the legend with cluster names and centers
    plt.legend(handles=handles, labels=cluster_name_dict.values())
    # adding the xlable
    plt.xlabel(xlabel, fontweight="semibold", fontsize="medium")
    # adding the ylable
    plt.ylabel(ylabel, fontweight="semibold", fontsize="medium")
    # adding the title
    plt.title(f"{title}", loc="center", fontweight="semibold",
              fontsize="large")
    # saving the plot in local drive
    plt.savefig(f"{title}_22031257.png".replace("\n", "_"))
    # closing and clearing the image
    plt.clf()
    plt.close()
# end of cluster_plot function


def cluster(df, ncluster):
    """
    cluster function clacluates the clusters for a given a dataframe, using
    kmeans technique and returns the normalized/scaled dataframe and clusters
    centers and the list of labels.
    """

    # initializing a scaler...
    scaler = RobustScaler()
    # initializing the kmeans method for clustering
    kmeans = KMeans(n_clusters=ncluster, n_init=init,
                    max_iter=300, random_state=32)
    # fitting the dataframe to the scaler
    scaler.fit(df)
    # transforming the dataframe - normalizing
    scaled_features = scaler.transform(df)
    # feeding the scaled data to the kmeans method for clustering
    kmeans.fit(scaled_features)
    # reversing the scaling to the find the actual cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # cluster labels
    labels = kmeans.labels_
    # returning the values
    return scaled_features, centers, labels
# end of cluster function


def silhouette_score_graph(df, title):
    """
    silhouette_score_graph function, calculates the silhouette score,
    for a range of clusters number - 2 to 10, and plot it in a 
    line graph, saved in the local drive.
    """
    print(f"\t[INFO] {title}...")

    # initializing the silhouette score list
    silhouette_coefficients = []
    # for loop
    for i in range(2, 11):
        # calling the cluster method
        scaled_features, _, labels = cluster(df, i)
        # calculating the score and sorting it
        score = silhouette_score(scaled_features, labels)
        silhouette_coefficients.append(score)
        # printing the score
        print(f"Silhouette Score for {i: 3} clusters is {score}")
    # end of for loop

    # initializing a figure for plotting
    plt.figure(dpi=300)
    # plotting the score as a line graph
    plt.plot(range(2, 11), silhouette_coefficients)
    # setting xlabel
    plt.xlabel("No. of Clusters", fontsize="medium", fontweight="semibold")
    # setting ylabel
    plt.ylabel("Silhouette Scores", fontsize="medium", fontweight="semibold")
    # setting title
    plt.title(f"Silhouette Scores\n{title}", fontsize="large",
              fontweight="semibold")
    # saving the image
    plt.savefig(f"Silhouette_score-{title}.png")
    print("\t\t[INFO] Silhuette Score Plotted and saved.")
    # closing and clearing the image
    plt.clf()
    plt.close()
    # returning the index of max score
    return np.argmax(silhouette_coefficients)+2
# end of silhouette_score_gragh function


def concat_indicators(df1, df2, keys):
    """
    concat_indicators function stacks the two dataframe and combines
    them cloumns-wise, fills the nans values with interpolation on both
    direction, returns the combined dataframe
    """

    # combining the dataframes
    df = pd.concat([df1.stack(future_stack=True).droplevel(level=1),
                    df2.stack(future_stack=True).droplevel(level=1)],
                   axis="columns", keys=keys).\
        interpolate(limit_direction="both")
    # returing the combined dataframe
    return df
# end of concat_indicators function


def cluster_main(gdp, forest, accs_elec, co2,
                 manufacturing, greenhouse, category):
    """
    cluster_main function combines the two dataframes, computes silhouette 
    score for clusters to find the best number of cluster and plots clusters
    """

    print(f"\n[INFO] {category} Based Clustering...")

    print("\t[INFO] Creating combine tables for...")
    # calling the concat_indicators funciton
    # electricity access and co2
    print("\t[INFO] Electricity Access v/s CO2 Emission...")
    elec_co2_df = concat_indicators(accs_elec, co2,
                                    ["Electricity Access", "CO2 Emission"])
    # manuafacturing and forest
    print("\t[INFO] Manufacturing v/s Forest Area...")
    manufacturing_forest_df = concat_indicators(manufacturing,
                                                forest,
                                                ["Manufacturing", "Forest"])
    # gdp and greenhouse gad emission
    print("\t[INFO] GDP v/s GreenHouse Gas Emission...")
    gdp_greenhouse_df = concat_indicators(gdp, greenhouse,
                                          ["GDP", "Greenhouse Gas"])
    print("\tDone.")

    # calling the silhouette_score_graph function
    print("\n[INFO] Silhouette Score for...")
    # electricity access and co2
    ncluster1 = silhouette_score_graph(elec_co2_df,
                                       f"{category}-Electricity" +
                                       " Access vs CO2 Emission")
    # manuafacturing and forest
    ncluster2 = silhouette_score_graph(manufacturing_forest_df,
                                       f"{category}-Manufacturing" +
                                       " vs Forest Area")
    # gdp and greenhouse gad emission
    ncluster3 = silhouette_score_graph(gdp_greenhouse_df,
                                       f"{category}-GDP vs" +
                                       " GreenHouse Gas Emission")

    # caliing the cluster function
    print("\n[INFO] Creating Clusters for...")
    # electricity access and co2
    print("\t[INFO] Electricity Access v/s CO2 Emission...")
    _, elec_co2_centers, elec_co2_labels = cluster(elec_co2_df, ncluster1)
    # manuafacturing and forest
    print("\t[INFO] Manufacturing v/s Forest Area...")
    _, manufacturing_forest_centers, manufacturing_forest_labels = \
        cluster(manufacturing_forest_df, ncluster2)
    # gdp and greenhouse gad emission
    print("\t[INFO] GDP v/s GreenHouse Gas Emission...")
    _, gdp_greenhouse_centers, gdp_greenhouse_labels = \
        cluster(gdp_greenhouse_df, ncluster3)
    print("\tDone.")

    # plotting the clusters
    print("\n[INFO] Cluster Plotting...")
    # electricity access and co2
    print("\t[INFO] Electricity Access v/s CO2 Emission...")
    # calling the cluster plot function
    cluster_plot(elec_co2_df, elec_co2_centers, elec_co2_labels,
                 indicator_names[2], indicator_names[3],
                 f"{category} Clusters\nElectricity Access vs CO2 Emission")
    # manuafacturing and forest
    print("\t[INFO] Manufacturing v/s Forest Area...")
    # calling the cluster plot function
    cluster_plot(manufacturing_forest_df, manufacturing_forest_centers,
                 manufacturing_forest_labels,
                 indicator_names[4], indicator_names[1],
                 f"{category} Clusters\nManufacturing vs Forest Area")
    # gdp and greenhouse gad emission
    print("\t[INFO] GDP v/s GreenHouse Gas Emission...")
    # calling the cluster plot function
    cluster_plot(gdp_greenhouse_df, gdp_greenhouse_centers,
                 gdp_greenhouse_labels,
                 indicator_names[0], indicator_names[5],
                 f"{category} Clusters\nGDP vs GreenHouse Gas Emission")
    print("\tDone.")
    print("\n------------------------------------------------\n")
# end of cluster_main function


def regional_income_agg(df):
    """
    regional_income_agg function groups the dataframe over regions and 
    incomegroup using pivot table functionality of pandas, and returns both
    """
    # regioanl grouping
    region = df.pivot_table(values=years, index="Region")
    # income-group grouping
    income = df.pivot_table(values=years, index="IncomeGroup")
    # returning both grouped data
    return region, income
# end of the regional_income_agg function


def eda(filename, file_metadata):
    """
    eda function reads the indicators and metadata files, and merges them,
    and drops unnecessary cloumns and sets country name as index.

    Also stores the indicators names in the list.

    Return the read indicator file and the merge file.
    """

    # reading the indicator
    df = pd.read_csv(filename,
                     skiprows=[0, 1, 2, 3])
    # reading the indicator's metadata
    df_metadata = pd.read_csv(file_metadata)
    # sorting the indicator name in the list
    indicator_names.append(df.iloc[0, 2])
    # dropping the columsn from indicator file
    df.drop(columns=["Indicator Name", "Indicator Code", df.columns[-1]],
            axis="columns", inplace=True)
    # droping columns form the metadata file
    df_metadata.drop(columns=[df_metadata.columns[-1],
                              df_metadata.columns[-2],
                              df_metadata.columns[-3]],
                     axis="columns", inplace=True)
    # merging the indicator and metadata file on country code
    df_merge = df.merge(df_metadata, on="Country Code", how="inner")
    # setting country name as index for merge file
    df_merge.set_index("Country Name", inplace=True, drop=True)
    # dropping country code from indicator file
    df.drop(columns="Country Code", axis="columns", inplace=True)
    # dropping country code from merge file
    df_merge.drop(columns="Country Code", axis="columns", inplace=True)
    # setting country name as index for indicator file
    df.set_index("Country Name", inplace=True)
    # droping all the index-country with no entries
    # df.dropna(axis="index", how="all", inplace=True)
    # taking a transpose of the indicator file
    df_transpose = df.T
    # renaming the axis 
    df_transpose.rename_axis("Years", inplace=True)
    # dropping all the nan's values row-wise
    df_transpose.dropna(axis="index", inplace=True)
    # dropping all the nan's values column-wise
    df_transpose.dropna(axis="columns", inplace=True)

    return df, df_merge, df_transpose
# end of the eda function


def main():
    """
    Main Function to read the file from the drive and 
    call other functions for clustering, fitting and ploting.
    """
    print("[INFO] 7PAM2000 APPLIED DATA SCIENCE ASSIGNMENT-3 \n\t\t \
        BY MOHIT AGARWAL[22031257]\n")
    print("\n================================================\n")

    print("[INFO] Reading all the indicators...")
    # calling the eda funciton for indicators
    # gdp
    gdp_df, gdp_df_merge, gdp_df_transpose = eda(
        indicator_filenames[0], indicator_metadata_filenames[0])
    # forest
    forest_df, forest_df_merge, forest_df_transpose = eda(
        indicator_filenames[1], indicator_metadata_filenames[1])
    # electricity access
    accs_elec_df, accs_elec_df_merge, accs_elec_df_transpose = eda(
        indicator_filenames[2], indicator_metadata_filenames[2])
    # co2 emission
    co2_df, co2_df_merge, co2_df_transpose = eda(
        indicator_filenames[3], indicator_metadata_filenames[3])
    # manufacturing
    manufacturing_df, manufacturing_df_merge, \
        manufacturing_df_transpose = eda(
            indicator_filenames[4], indicator_metadata_filenames[4])
    # greenhouse gas emission
    greenhouse_df, greenhouse_df_merge, greenhouse_df_transpose = eda(
        indicator_filenames[5], indicator_metadata_filenames[5])
    print("\tRead.")
    print("\n================================================\n")

    # scaling gdp by 1e10
    gdp_df = gdp_df/1e10
    gdp_df_merge.iloc[:, :-2] = gdp_df_merge.iloc[:, :-2]/1e10
    gdp_df_transpose = gdp_df_transpose/1e10

    # scaling greenhouse gas emission by 1e5
    greenhouse_df = greenhouse_df/1e5
    greenhouse_df_merge.iloc[:, :-2] = greenhouse_df_merge.iloc[:, :-2]/1e5
    greenhouse_df_transpose = greenhouse_df_transpose/1e5

    # changing names in indicator_names for...
    # gdp
    indicator_names[0] = indicator_names[0][:indicator_names[0].find("$")+1]\
        + ": in 10B " + indicator_names[0][indicator_names[0].find("$")+1:]
    # greenhouse gas emission
    indicator_names[5] = indicator_names[5][:indicator_names[5].find("(")+1]\
        + "100000 " + indicator_names[5][indicator_names[5].find("(")+1:]

    print("\n[INFO] Generating Regional and IncomeGroup Tabels...")
    # calling regional_income_agg funcion
    # gdf
    _, gdp_df_income = regional_income_agg(gdp_df_merge)
    # forest
    _, forest_df_income = regional_income_agg(forest_df_merge)
    # electricity access
    _, accs_elec_df_income = regional_income_agg(
        accs_elec_df_merge)
    # co2 emission
    _, co2_df_income = regional_income_agg(co2_df_merge)
    # manufacturing
    _, manufacturing_df_income = regional_income_agg(
        manufacturing_df_merge)
    # greenhouse gas emission
    _, greenhouse_df_income = regional_income_agg(
        greenhouse_df_merge)
    print("\tDone.")
    print("\n================================================\n")

    # calling the cluster_main function for incomegroup
    cluster_main(gdp_df_income, forest_df_income,
                 accs_elec_df_income, co2_df_income,
                 manufacturing_df_income, greenhouse_df_income,
                 "IncomeGroup")

    print("\t[INFO] Clustering Done.")
    print("\n================================================\n")

    # uncomment if you want to see fitting for World
    # # adding world to the countries list
    # countries.append("World")

    # calling the fitting_main fucntion country
    print("\n[INFO] Curve Fitting for...")
    curve_fit_main(gdp_df, indicator_names[0], "GDP")
    curve_fit_main(forest_df, indicator_names[1], "Forest Area")
    curve_fit_main(accs_elec_df, indicator_names[2], "Electricity Access")
    curve_fit_main(co2_df, indicator_names[3], "CO2 Emission")
    curve_fit_main(manufacturing_df, indicator_names[4], "Manufacturing")
    curve_fit_main(greenhouse_df, indicator_names[5],
                   "Total Greenhouse Gas Emission")

    # uncomment if you want to product fitting based on incomegroup
    # # calling the fitting_main fucntion incomegroup
    # print("\n[INFO] Curve Fitting for...")
    # income_names = gdp_df_income.index.unique()
    # curve_fit_main(gdp_df_income, indicator_names[0],
    #                "GDP", income_names)
    # curve_fit_main(forest_df_income, indicator_names[1],
    #                "Forest Area", income_names)
    # curve_fit_main(accs_elec_df_income, indicator_names[2],
    #                "Electricity Access", income_names)
    # curve_fit_main(co2_df_income, indicator_names[3],
    #                "CO2 Emission", income_names)
    # curve_fit_main(manufacturing_df_income, indicator_names[4],
    #                "Manufacturing", income_names)
    # curve_fit_main(greenhouse_df_income, indicator_names[5],
    #                "Total Greenhouse Gas Emission", income_names)

    print("\t[INFO] Curve-Fitting Done.")
    print("\n================================================\n")

    # ending the assignment
    print("[INFO] Clusters and Curve-Fitting are done.\n\t\
        BY MOHIT AGARWAL[22031257]\n")
# end of main function


# main function executes when not imported as package.
if __name__ == "__main__":
    main()
# end of if condition
