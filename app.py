from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import plotly.express as px
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

def detect_outliers(dataframe, contamination=0.1, random_state=42):
    # Selecting only the numerical feature for outlier detection

    # Replace non-numeric and missing values with "N/A"
    dataframe["Number of people undernourished (million) (3-year average)"] = pd.to_numeric(
        dataframe["Number of people undernourished (million) (3-year average)"], errors='coerce')
    dataframe["Number of people undernourished (million) (3-year average)"] = dataframe[
        "Number of people undernourished (million) (3-year average)"].fillna(-1)

    X = dataframe[["Number of people undernourished (million) (3-year average)"]].values

    # Initialize the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model and predict the outliers
    outliers = model.fit_predict(X)

    # Add an 'Outlier' column to the dataframe to indicate outlier status
    dataframe['Outlier'] = np.where(outliers == -1, True, False)

    return dataframe

# Function to perform K-Means clustering
def perform_kmeans_clustering(df, num_clusters):
    # Replace non-numeric and missing values with "N/A"
    df["Number of people undernourished (million) (3-year average)"] = pd.to_numeric(
        df["Number of people undernourished (million) (3-year average)"], errors='coerce')
    df["Number of people undernourished (million) (3-year average)"] = df[
        "Number of people undernourished (million) (3-year average)"].fillna(-1)

    # Select only the feature you want to cluster on
    X = df[["Number of people undernourished (million) (3-year average)"]]

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    return df

# Function to plot world map
def plot_world_map(dataframe, title, color_column):
    fig = px.choropleth(dataframe,
                        locations="Country",
                        locationmode="country names",
                        color=color_column,
                        hover_name="Country",
                        projection="natural earth",
                        title=title,
                        )
    fig.update_geos(showcoastlines=True, coastlinecolor="RebeccaPurple", showland=True, landcolor="LightGreen",
                    showocean=True, oceancolor="LightBlue", showlakes=True, lakecolor="LightBlue")
    st.plotly_chart(fig)

def tab_one():
  # Load the data
  df = pd.read_csv("Data/country-wise-average.csv")

  # Filter by Country for Stunting Bar Chart (Select up to 10 countries)
  st.title("Stunting in Selected Countries")
  selected_stunting_countries = st.multiselect(
    "Select Country(s) for Stunting Bar Chart (Select up to 10 countries)",
    df["Country"].unique(),
    default=[],
    max_selections=10)
  filtered_stunting_data = df[df["Country"].isin(selected_stunting_countries)]

  # Stunting Bar Chart
  fig1 = px.bar(filtered_stunting_data,
                y="Stunting",
                x="Country",
                color="Country",
                color_discrete_sequence=px.colors.sequential.Blugrn)
  fig1.update_layout(title="Stunting in Selected Countries",
                     xaxis_title="Country name",
                     yaxis_title="Stunting")
  fig1.update_xaxes(tickangle=-45)
  st.plotly_chart(fig1)

  # Stunting Choropleth Map
  st.title("Stunting Percentage in All Countries")
  fig2 = go.Figure(data=[
    go.Choropleth(
      locations=df['Country'],
      locationmode='country names',
      z=df['Stunting'],
      colorscale=px.colors.sequential.Blugrn,
      colorbar_title="Stunting %",
    )
  ])
  fig2.update_layout(
    title="Stunting Percentage in All Countries",
    geo=dict(scope='world'),
  )
  st.plotly_chart(fig2)

  # Top 10 Countries for Average Malnutrition
  top_n = 10
  top_countries = df.nlargest(top_n, 'Stunting')
  st.title("Top 10 Countries for Average Malnutrition")
  fig3 = px.bar(top_countries,
                x='Country',
                y='Stunting',
                color='Country',
                color_discrete_sequence=px.colors.sequential.Tealgrn)
  fig3.update_layout(
    title="Top 10 Countries for Average Malnutrition",
    xaxis_title="Country name",
    yaxis_title="Average Malnutrition",
  )
  fig3.update_xaxes(tickangle=-45)
  st.plotly_chart(fig3)


def tab_two():
    st.title("Agricultural Data Visualization on World Map")
    # Load the data
    df = pd.read_csv('Data/data.csv')

    # Dropdown for country name
    countries = df['Area'].unique()
    selected_country = st.selectbox("Select Country", countries)

    # Dropdown for product name
    items = df['Item'].unique()
    selected_item = st.selectbox("Select Product", items)

    # Dropdown for year
    years = df['Year'].unique()
    selected_year = st.selectbox("Select Year", years)

    # Filter the data based on dropdown selections
    filtered_df = df[(df['Area'] == selected_country) & (df['Item'] == selected_item) & (df['Year'] == selected_year)]
    
    # Check if any data exists for the selected filters
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        fig = px.choropleth(filtered_df, locations="Area", locationmode="country names",
                            color="Value", hover_name="Item", animation_frame="Year",
                            color_continuous_scale="Viridis", projection="natural earth")

        fig.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="LightGray",
                        showocean=True, oceancolor="Azure", showlakes=True, lakecolor="Azure")

        fig.update_layout(title="Agricultural Data on World Map",
                          margin={"r":0,"t":30,"l":0,"b":0},
                          coloraxis_colorbar=dict(title="Value"))

        st.plotly_chart(fig)

def tab_three():
    data = pd.read_csv("Data/undernourished.csv")
    df = pd.DataFrame(data)
    
    st.title("World Map Dashboard - Undernourished People")

    # Display the data table
    st.subheader("Data Table")
    st.dataframe(df)

    # Display the world map plot
    st.subheader("World Map")
    plot_world_map(df, "World Map of Undernourished People",
                   "Number of people undernourished (million) (3-year average)")

    # Perform outlier detection
    df_outliers = detect_outliers(df)

    # Display the outlier detection results
    st.subheader("Outlier Detection - Undernourished People")
    st.subheader("Data Table with Outliers")
    st.dataframe(df_outliers)

    # Display the world map plot with outlier detection results
    st.subheader("World Map with Outliers")
    plot_world_map(df_outliers, "Outlier Detection - Undernourished People", "Outlier")

    # Get number of clusters from the user
    num_clusters = st.slider("Choose the number of clusters:", min_value=2, max_value=len(df), value=3)

    # Perform K-Means clustering
    clustered_df = perform_kmeans_clustering(df, num_clusters)

    # Display the clustered data table
    st.subheader("Clustered Data Table")
    st.dataframe(clustered_df)

    # Display the world map plot with clusters colored differently
    st.subheader("World Map - Clustering Results")
    plot_world_map(clustered_df, "World Map with Clustering Results", "Cluster")

def tab_four():
    st.title("Global Hunger Index (2021)")
    # Create a dropdown to select the visualization
    selected_view = st.selectbox("Select a view:", ["Map", "Table", "Chart", "Prediction"])

    data_url = "Data/global-hunger-index.csv"  # Replace with your data URL or file path
    df = pd.read_csv(data_url)

    # Display the selected view
    if selected_view == "Map":
        st.subheader("Global Hunger Index Map")
        
        # Create a dropdown to select the year
        selected_year = st.selectbox("Select a year:", df["Year"].unique())
        
        # Filter the data based on the selected year
        filtered_df = df[df["Year"] == selected_year]
        
        fig_map = px.choropleth(filtered_df, locations="Code", locationmode="ISO-3", color="Global Hunger Index (2021)",
                                hover_name="Entity", projection="natural earth", color_continuous_scale="viridis")
        st.plotly_chart(fig_map)

    elif selected_view == "Table":
        st.subheader("Data Table")
        st.dataframe(df)

    elif selected_view == "Chart":
        # Create a dropdown to select a country (used in Chart and Prediction views)
        selected_country = st.selectbox("Select a country:", df["Entity"].unique(), key="country")
        st.subheader(f"Global Hunger Index for {selected_country}")
        filtered_df = df[df["Entity"] == selected_country]
        fig_bar = px.bar(filtered_df, x="Year", y="Global Hunger Index (2021)", labels={"Global Hunger Index (2021)": "Hunger Index"})
        st.plotly_chart(fig_bar)

    elif selected_view == "Prediction":
        X = df["Year"].values.reshape(-1, 1)
        y = df["Global Hunger Index (2021)"].values
        model = LinearRegression()
        model.fit(X, y)
        st.subheader("Global Hunger Index Prediction")
        year_to_predict = st.slider("Select Future Year:", min_value=2022, max_value=2030)

        prediction = model.predict([[year_to_predict]])

        st.write("Predicted Global Hunger Index for {}: {:.2f}".format(year_to_predict, prediction[0]))

        X = df["Year"]
        y_actual = df["Global Hunger Index (2021)"]
        y_predicted = model.predict(df["Year"].values.reshape(-1, 1))
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y_actual, label="Actual GHI", color='blue')
        plt.plot(X, y_predicted, label="Regression Line", color='red')
        plt.xlabel("Year")
        plt.ylabel("Global Hunger Index")
        plt.title("Global Hunger Index over the Years")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


def main():
  st.title("Streamlit App with Tabs")

  # Create a select box to choose the active tab
  selected_tab = st.selectbox("Select Tab:", ("Shunting Analysis", "Agriculture Analysis", "Undernpourish people Analysis", "Global hunger Index"))

  # Display the selected tab content
  if selected_tab == "Shunting Analysis":
    tab_one()
  elif selected_tab == "Agriculture Analysis":
    tab_two()
  elif selected_tab == "Undernpourish people Analysis":
    tab_three()
  elif selected_tab == "Global hunger Index":
    tab_four()


if __name__ == "__main__":
  main()
