"""
7PAM2000 Applied Data Science 1
Assignment 2: Statistics and trends
------------------------------------------------------------------------------
Mohit Agarwal (Student ID-22031257)

Exploring statistics and trends on country-by-country indicators related to 
Climate Change from World-Bank.

https://data.worldbank.org/topic/climate-change

The chosen indicators are:
1.Urban population growth (annual %)
2.Forest area (% of land area)
3.Renewable energy consumption (% of total final energy consumption)
4.Electric power consumption (kWh per capita)
5.CO2 emissions (kt)
6.Annual freshwater withdrawals, total (billion cubic meters)

The 12 selected conturies are:
1.India
2.Japan
3.United States
4.United Kingdom
5.Korea, Dem. People's Rep.
6.Egypt, Arab Rep.
7.Australia
8.Brazil
9.United Arab Emirates
10.Russian Federation
11.China
12.Germany

The stats and trends were looked from the year 1990 to 2020.

Note: 
No plots/grahs are shown but all are saved directly in the working directory.
"""

# importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# list of file name of all indicators being used
data_filenames = ["API_SP.URB.GROW_DS2_en_csv_v2_5996762.csv",
                  "API_AG.LND.FRST.ZS_DS2_en_csv_v2_5994693.csv",
                  "API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5995541.csv",
                  "API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5995551.csv",
                  "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5994970.csv",
                  "API_ER.H2O.FWTL.K3_DS2_en_csv_v2_5995631.csv"]

# list of countries for analysis
countries = ["India", "Japan", "United States", "United Kingdom",
             "Korea, Dem. People's Rep.", "Egypt, Arab Rep.", "Australia",
             "Brazil", "United Arab Emirates", "Russian Federation",
             "China", "Germany"]

# years in increament of 5, starting for 1990
years = ["1990", "1995", "2000", "2005", "2010", "2015", "2020"]

# indicator names list created
indicator_names = []

# reading the country metadata file.
country_metadata_df = pd.read_csv(
    "Metadata_Country_API_SP.URB.GROW_DS2_en_csv_v2_5996762.csv")
# droping the unwanted cloumns
country_metadata_df.drop(country_metadata_df.columns[3:],
                         axis=1, inplace=True)
# droping empty rows
country_metadata_df.dropna(inplace=True)

# color palettes
palettes = [sns.color_palette("Set2"),
            sns.color_palette("Paired"),
            sns.color_palette("mako", as_cmap=True),
            sns.color_palette("rocket"),
            sns.color_palette("cubehelix", as_cmap=True),
            sns.color_palette("icefire", as_cmap=True),
            sns.color_palette("crest"),
            sns.color_palette("magma"),
            sns.color_palette("flare", as_cmap=True),
            sns.color_palette("viridis"),
            sns.color_palette("Spectral", as_cmap=True),
            sns.color_palette("coolwarm", as_cmap=True)]

# short indicator names
short_indicator_names = ["Urban Population(%)",
                         "Forest Area(%)",
                         "Renewable Energy(%)",
                         "Electric Power",
                         "CO2 Emission(kt)",
                         "Freshwater"]


def indicator_stats(indicator_df, country_index, year_index, indicator):
    """
    Looking at the basic statistics of the indicators and ploting some
    of them, region and income group wise.
    """
    print(f"\n[INFO] {indicator}-->")
    print("\t[INFO] Describe method over years:\n", country_index.describe())
    print("\t[INFO] Describe method over countries:\n", year_index.describe())
    # printing the skwen and kurtosis
    print("\t[INFO] Skew and Kurtosis for each countries:\n",
          pd.concat([country_index.skew(axis=1),
                    country_index.kurtosis(axis=1)],
                    keys=["Skew", "Kurtosis"], axis=1))
    # dropping all the empty columns in indicator dataframe
    indicator_df.dropna(how="all", axis="columns", inplace=True)
    # dropping all the empty rows in the dataframe
    indicator_df.iloc[:, 0:-2].ffill(axis="columns", inplace=True)

    # checking for electricial indicator as no data beyond 2014
    if indicator == indicator_names[3]:
        year = ["1980", "1985", "1990", "1995", "2000", "2005", "2010"]
    else:
        year = years

    # creating a bar plot to visualize the median value over regions
    indicator_df.pivot_table(values=year,
                             index="Region", aggfunc="median").plot(
                                 kind="bar", figsize=(12, 8),
                                 xlabel="Region", ylabel=indicator,
                                 title=f"Regions - {indicator}")
    # making the layout tight
    plt.tight_layout()
    # plt.legend(loc="right", bbox_to_anchor=(1.275, 0.5), draggable=True)
    # saving the graph in the drive
    plt.savefig(f"Region vs {indicator}_bar.png")
    plt.close()

    print("[INFO] Region wise Bar graph visualised,",
          "after looking at the yearly median value.")

    # creating a line plot to visualize the median value over income group
    indicator_df.pivot_table(values=indicator_df.columns[:-2],
                             index="IncomeGroup", aggfunc="median").T.plot(
                                 figsize=(12, 8), xlabel="Years",
                                 ylabel=indicator, kind="line",
                                 title=f"Income Group - {indicator}")
    # making the layout tight
    plt.tight_layout()
    # saving the graph in the drive
    plt.savefig(f"IncomeGroup vs {indicator}_bar.png")
    plt.close()

    print("[INFO] IncomeGroup wise Line graph visualised,",
          "after looking at the yearly median value.")

    # creating a bar plot to visualize the mean value over income group
    indicator_df.pivot_table(values=indicator_df.columns[:-2],
                             index="IncomeGroup", aggfunc="mean").T.plot(
                                 figsize=(12, 8), xlabel="Years",
                                 ylabel=indicator, kind="bar",
                                 title=f"Income Group - {indicator}",
                                 grid=True)
    # making the layout tight
    plt.tight_layout()
    # sliming the x-axis 
    plt.xlim(left=20)
    # saving the graph in the drive
    plt.savefig(f"IncomeGroup vs {indicator}_bar.png")
    plt.close()

    print("[INFO] IncomeGroup wise Bar graph visualised,",
          "after looking at the yearly mean value.")

    # visualizing the country wise df as a whole on a bar plot
    country_index.plot(use_index=True, y=years, figsize=(12, 8),
                       kind="bar", title=f"{indicator}",
                       ylabel=indicator)
    # making the layout tight
    plt.tight_layout()
    # saving the graph in the drive
    plt.savefig(f"{indicator}_bar.png")
    plt.close()
    print("[INFO] Bar graph plotted for selected countries.")

    # visualizing the year wise df as a whole on line plot
    year_index.plot(use_index=True, y=year_index.columns, figsize=(12, 8),
                    kind="line", alpha=0.7, title=f"{indicator}",
                    ylabel=indicator)
    # ploting the legend for freshwater 
    if indicator == indicator_names[-1]:
        plt.legend(year_index.columns, loc="center", bbox_to_anchor=(0.5, 0.4))
    # plt.legend(countries, loc="right", bbox_to_anchor=(1.3, 0.5))
    # making the layout tight
    plt.tight_layout()
    # saving the graph in the drive
    plt.savefig(f"{indicator}_line.png")
    plt.close()

    print("[INFO] Line graph plotted for selected countries.")

    # finding the max years for each countries
    max_year = pd.concat([country_index.loc[:, :].idxmax(axis=1),
                          country_index.loc[:, :].max(axis=1)],
                         axis=1, keys=["Year", indicator])
    # sorting in descending
    max_year.sort_values(indicator, ascending=False, inplace=True)
    # ploting the maximum years
    plt.figure(figsize=(12, 8))
    sns.barplot(data=max_year, x=max_year.index, y=indicator, hue="Year")
    plt.xticks(rotation=90)
    plt.title(f"Maximum {indicator} per Country")
    plt.tight_layout()
    plt.savefig(f"max_bar_{indicator}.png")
    plt.close()

    print("[INFO] Maximum value found per Country and",
          "it's corresponding year,ploted as a Bar graph.")

    print("\n================================================\n")

# end of basic_stats function


def indicator_heatmaps(urban_growth_country_index,
                       forest_area_country_index,
                       renewable_energy_country_index,
                       electric_consumption_country_index,
                       co2_emission_country_index,
                       fresh_h2o_country_index):
    """
    Correlation between indicators for each countries over the
    years from 1990 and 2020 and ploting them as a heatmap
    """

    # looping over each countries
    for i, country in enumerate(countries):
        # concating all the indicators countries wise
        country_heatmap = pd.concat([
            urban_growth_country_index.cumsum(axis=1).loc[country],
            forest_area_country_index.loc[country],
            renewable_energy_country_index.loc[country],
            electric_consumption_country_index.loc[country],
            co2_emission_country_index.loc[country],
            fresh_h2o_country_index.loc[country]],
            axis="columns",
            keys=indicator_names)
        # fordward filling method for NULL values
        country_heatmap.ffill(inplace=True)

        # initializing the figure with size of 12x8
        plt.figure(figsize=(12, 8))
        # ploting the heatmap
        sns.heatmap(data=country_heatmap.corr(), annot=True,
                    linecolor="lightgrey",
                    linewidths=1.5, cmap=palettes[i],
                    xticklabels=short_indicator_names,
                    yticklabels=short_indicator_names)
        # adding a title to the headmap
        plt.title(country)
        # tight layout
        plt.tight_layout()
        # rotating x labels by 45 degree
        # plt.xticks(rotation=45)
        # saving the figure
        plt.savefig(f"{country}_heatmap.png")
        # closing the figure
        plt.close()

        print(f"\t[INFO] Correlation for {country} - plotted as heatmap",
              "and saved into the drive")
# end of the country_heatmap function


def read_indicator(filename):
    """
    Reading the indicators, and returning three dataframes:
    1.A merge dataset with country metadata to look at the 
    regions and income group, with Country as index.
    2. A subset of dataset with selected countries as the 
    index and the years as columns
    3. A transpose of the above dataframe with years as index and 
    the countries as columns. 
    """

    # reading the indicator and skipping first few rows
    df = pd.read_csv(filename, skiprows=[0, 1, 2, 3])
    # extracting the name of the indicator, and putting in a list
    indicator_names.append(df["Indicator Name"].values[0])
    # merging the read dataframe with country metadata
    df_merge = df.merge(country_metadata_df, on="Country Code", how="inner")
    # droping the unwanted columns
    df_merge.drop(columns=["Country Code", "Indicator Name", "Indicator Code",
                           "Unnamed: 67"], axis="columns", inplace=True)
    # setting country name as index for the merged df
    df_merge.set_index("Country Name", inplace=True, drop=True)
    # subseting the df by selecting wanted years and countries
    df_subset = df_merge.loc[df_merge.index.isin(
        countries), "1990":"2020"].copy()
    # tranposing the df
    df_subset_tranpose = df_subset.transpose()
    # renaming the index for tranposed df to years
    df_subset_tranpose.rename_axis("Years", inplace=True)

    # returning the three dfs for analysis
    return df_merge, df_subset, df_subset_tranpose
# end of the read_indicator function


def main():
    """
    Main Function to read the file from the drive and 
    call other functions for various tasks and ploting
    few specific plots
    """

    print("[INFO] 7PAM2000 APPLIED DATA SCIENCE ASSIGNMENT-2 \n\t\t \
        BY MOHIT AGARWAL[22031257]\n")

    print("[INFO] Reading all the indicators...")
    # reading the urban growth indicator
    urban_growth_df, urban_growth_country_index, \
        urban_growth_year_index = read_indicator(data_filenames[0])
    # reading the forest area indicator
    forest_area_df, forest_area_country_index, \
        forest_area_year_index = read_indicator(data_filenames[1])
    # reading the renewable energy indicator
    renewable_energy_df, renewable_energy_country_index, \
        renewable_energy_year_index = read_indicator(data_filenames[2])
    # reading the electric consumption indicator
    electric_consumption_df, electric_consumption_country_index, \
        electric_consumption_year_index = read_indicator(data_filenames[3])
    # reading the co2 emission indicator
    co2_emission_df, co2_emission_country_index, \
        co2_emission_year_index = read_indicator(data_filenames[4])
    # reading the fresh water indicator
    fresh_h2o_df, fresh_h2o_country_index, \
        fresh_h2o_year_index = read_indicator(data_filenames[5])
    print("\tRead.")

    # cleaning the electric consumption dataframe
    # dropping all the columns with no entries
    electric_consumption_df.dropna(how="all", axis="columns", inplace=True)
    # dropping all those columns half of its entries empty
    electric_consumption_df.dropna(thresh=int(
        len(electric_consumption_df)/2), axis="columns", inplace=True)
    # filling the remaining nan with forward values then with backward values.
    electric_consumption_df.iloc[:, 0:-2].ffill(axis=1, inplace=True)
    electric_consumption_df.iloc[:, 0:-2].bfill(axis=1, inplace=True)

    print("\n[INFO] Stats for Indicators:\n")

    # calling the stats function for urban growth indicator
    indicator_stats(urban_growth_df,
                    urban_growth_country_index,
                    urban_growth_year_index,
                    indicator_names[0])
    # calling the stats function for forest area indicator
    indicator_stats(forest_area_df, forest_area_country_index,
                    forest_area_year_index, indicator_names[1])
    # calling the stats function for renewable energy indicator
    indicator_stats(renewable_energy_df, renewable_energy_country_index,
                    renewable_energy_year_index, indicator_names[2])
    # calling the stats function for electric consumption indicator
    indicator_stats(electric_consumption_df,
                    electric_consumption_country_index,
                    electric_consumption_year_index, indicator_names[3])
    # calling the stats function for c02 emission indicator
    indicator_stats(co2_emission_df, co2_emission_country_index,
                    co2_emission_year_index, indicator_names[4])
    # calling the stats function for fresh water indicator
    indicator_stats(fresh_h2o_df, fresh_h2o_country_index,
                    fresh_h2o_year_index, indicator_names[5])

    # looking at the cumulative sum on urban growth rate
    urban_growth_country_index.cumsum(axis=1).plot(
        kind="bar", y=years,
        ylabel=indicator_names[0],
        title=f"Cumilative Sum {indicator_names[0]}",
        figsize=(12, 8), color=palettes[1])
    # making the layout tight
    plt.tight_layout()
    # saving the plot
    plt.savefig(f"Cumsum_{indicator_names[0]}.png")
    plt.close()
    print("\n[INFO] Cumulative Sum of Urban",
          "Growth Rate visulaised as bar plot.")

    print("\n[INFO] Heatmaps for selected countries:\n")
    # call the indicator_heatmaps function
    indicator_heatmaps(urban_growth_country_index,
                       forest_area_country_index,
                       renewable_energy_country_index,
                       electric_consumption_country_index,
                       co2_emission_country_index,
                       fresh_h2o_country_index)

    # ending the assignment
    print("[INFO] The Statistics and trends are looked at.\n\t\
        BY MOHIT AGARWAL[22031257]\n")
# end of main function


# main function executes when not imported as package.
if __name__ == "__main__":
    main()
# end of if condition
