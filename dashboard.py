import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import geopandas as gpd
import json
import folium
from folium import Choropleth, GeoJson  # Import GeoJson here
from streamlit_folium import st_folium
import requests
import plotly.graph_objects as go


import streamlit as st
import os
import pandas as pd
import google.generativeai as genai
from PIL import Image
import base64

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Lok Sabha Election Analysis",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your other Streamlit code starts here
st.markdown(
    """
    <style>
    .gradient-text {
        font-size: 3em;  /* Adjust this to make the text bigger */
        background: -webkit-linear-gradient(left, #FF9933, #FFFFFF, #138808);
        -webkit-background-clip: text;
        color: transparent;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='gradient-text'>Lok Sabha Election Analysis</h1><br><br>", unsafe_allow_html=True)


# Retrieve API key from Streamlit secrets
gemini_api = st.secrets["gemini_api"]

# Configure the Gemini model with the API key
genai.configure(api_key=gemini_api)


# Initialize the Generative Model
if not gemini_api:
    st.sidebar.error("API key is not set. Please configure the GEMINI_API_KEY environment variable in Streamlit Secrets.")
else:
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
    except Exception as e:
        st.sidebar.error(f"Failed to initialize the model: {e}")

# Use the sidebar for both input and output
with st.sidebar:
    st.title("Ask Anything Related to Indian Election")

    # Input box
    input_query = st.text_input("",placeholder="Write your query here:", key="sidebar_query_input")

    # Display the response in the sidebar
    if st.button("Get General Info", key="sidebar_submit_button"):
        if input_query and gemini_api:
            try:
                # Prepare the full prompt for the model
                full_prompt = (
                    f"Role: Act as a Conversational Assistant\n"
                    f"Task: Answer the question '{input_query}' in a clear and concise manner. The response should be informative and include a relevant general assumption and minimal embellishment. Aim for a conversational tone, avoiding robotic language. Keep the response under 50 words.\n"
                    f"Break down the response into points:\n"
                    f"1. Start with a straightforward and accurate answer to the question.\n"
                    f"2. Add a relevant general assumption or context if needed.\n"
                    f"3. Include a touch of embellishment to make the response engaging, but avoid unnecessary details.\n"
                    f"4. Ensure the response is natural and conversational, avoiding any robotic or overly formal language."
                )
                
                # Generate content using the model
                response = model.generate_content(full_prompt)
                
                # Display the response in the sidebar
                st.sidebar.write(response.text)
            except Exception as e:
                st.sidebar.error(f"An error occurred: {e}. Please try again later or check your API key/quota.")
        elif not gemini_api:
            st.sidebar.error("API key is not set. Please configure the GEMINI_API_KEY environment variable.")
        else:
            st.sidebar.write("Please enter a query.")


# Load dataframes (assuming these CSVs are available)
df1 = pd.read_csv('constituency.csv')
df2 = pd.read_csv('Candidates.csv')
df3 = pd.read_csv('Election.csv')
gender_ratio = pd.read_csv('gender_rat.csv')

# Sidebar filters
st.sidebar.title("Filters")
years = st.sidebar.multiselect("Select Year(s)", df3['election_year'].unique(), default=[2019])
states = st.sidebar.multiselect("Select State(s)", df3['state'].unique(), default=['Gujarat'])

# Filter `pc_names` based on selected `states` and `years`
if states and years:
    filtered_df3_for_pc_names = df3[df3['state'].isin(states) & df3['election_year'].isin(years)]
    pc_names_options = filtered_df3_for_pc_names['pc_name'].unique()
else:
    # If no state or year is selected, show all pc_names for all years
    pc_names_options = df3['pc_name'].unique()

# Default to the last 10 `pc_names` if none are selected, or all if fewer than 10
default_pc_names = list(pc_names_options) if len(pc_names_options) <= 10 else list(pc_names_options)[-10:]


# Sidebar multiselect for `pc_names`
pc_names = st.sidebar.multiselect("Select Constituency(ies)", pc_names_options, default=default_pc_names)


# Apply filters to main dataframes
filtered_df1 = df1[df1['election_year'].isin(years) & df1['pc_name'].isin(pc_names)]
filtered_df2 = df2[df2['election_year'].isin(years) & df2['pc_name'].isin(pc_names)]
filtered_df3 = df3[df3['election_year'].isin(years)] # df3 is used for overall election metrics, so only filter by years


with st.sidebar:
    st.title("AI Data Analyst")
    st.write("Engage in insightful conversations with your data through powerful visualizations, empowering you to uncover valuable insights and make informed decisions effortlessly!")
    st.divider() # Added a divider
    with st.expander("Data Visualization"):
        st.write("Made with Gemini pro ")

    st.write("<div>Developed by - <span style=\"color: cyan; font-size: 24px; font-weight: 600;\">Sonu Sinha</span></div>",unsafe_allow_html=True)
    # st.write("<div>Developed by - <span style=\"color: cyan; font-size: 24px; font-weight: 600;\">Jaishree Yadav</span></div>",unsafe_allow_html=True)

# Define a function to format numbers
def format_number(num):
    """Format numbers into human-readable formats (K, M, B, L)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} B"  # Billions
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"  # Millions
    elif num >= 100_000:
        return f"{num / 100_000:.1f} L"  # Lakhs
    elif num >= 1_000:
        return f"{num / 1_000:.1f} K"  # Thousands
    else:
        return str(num)  # No formatting needed

# Calculate metrics
total_votes = round(filtered_df3['votes'].sum(), 2)
total_turnout = round(filtered_df3['turnout'].mean(), 2)
total_constituencies = filtered_df3['pc_name'].nunique()
total_candidates = filtered_df3['winning_candidate'].nunique()

# Format metrics
formatted_total_votes = format_number(total_votes)
formatted_total_turnout = format_number(total_turnout) + "%"
if 2019 in years or 2024 in years:
    formatted_total_constituencies = "543" # Hardcoded for these years as per requirement
else:
    formatted_total_constituencies = format_number(total_constituencies)

formatted_total_candidates = format_number(total_candidates)

st.subheader("Election Metrics 🗳️")

# Election Metrics in a Single Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Votes", formatted_total_votes)
col2.metric("Total Turnout", formatted_total_turnout)
col3.metric("Total Constituencies", formatted_total_constituencies)
col4.metric("Total Candidates", formatted_total_candidates)


st.markdown("<br><br>", unsafe_allow_html=True)
st.subheader("Election Analysis")

# Create three columns
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Top Parties with Seats Won")

    # Group by party and count occurrences
    party_votes = filtered_df3['party'].value_counts()

    # Sort by votes in descending order
    sorted_party_votes = party_votes.sort_values(ascending=False)

    # Get the top 6 parties
    top_parties = sorted_party_votes.head(6)

    # Convert the index to a categorical type to preserve the order
    top_parties.index = pd.CategoricalIndex(top_parties.index, categories=top_parties.index, ordered=True)

    # Convert to DataFrame for Plotly
    top_parties_df = top_parties.reset_index()
    top_parties_df.columns = ['Party', 'Seats'] # Renamed 'Votes' to 'Seats' for clarity

    # Create the bar chart using Plotly Express
    fig = px.bar(top_parties_df, x='Party', y='Seats', text='Seats', title="Top Parties by Seats Won")

    # Rotate the x-tick labels by -45 degrees
    fig.update_layout(
        xaxis_title='Party',
        yaxis_title='Seats Count',
        xaxis_tickangle=-45,     # Rotate the x-ticks
        height=550,              
        yaxis=dict(range=[0, top_parties.max() * 1.2]),  # Set y-axis range slightly above the max value
        template='plotly_white'
    )

    # Automatically position data labels outside the bars
    fig.update_traces(textposition='outside')

    # Display the plot in Streamlit
    st.plotly_chart(fig)

with col2:
    st.write("Candidates by Category")

    # Get the top 3 type categories
    top_type_category = filtered_df3["type_category"].value_counts().head(3).reset_index()
    top_type_category.columns = ['Type Category', 'Count']

    # Create the bar chart using Plotly Express
    fig_category = px.bar(top_type_category, x='Type Category', y='Count', text='Count', title="Top Candidate Categories")

    # Adjust layout to rotate x-ticks and increase the bar height
    fig_category.update_layout(
        xaxis_title='Type Category',
        yaxis_title='Number of Candidates',
        xaxis_tickangle=-45,          # Rotate the x-ticks by -45 degrees
        yaxis=dict(range=[0, top_type_category['Count'].max() * 1.1]),  # Slightly above max value
        height=480,                   # Set a custom height for the chart
        template='plotly_white'
    )

    # Position data labels outside the bars
    fig_category.update_traces(textposition='outside')

    # Display the plot in Streamlit
    st.plotly_chart(fig_category)

# Plotting the Male vs Female Turnout Ratio Over the Years
with col3:
    st.write("Genderwise Voter Turnout")

    # Sort the DataFrame by Year
    gender_ratio = gender_ratio.sort_values('Year')

    # Create Plotly figure
    fig_turnout = go.Figure()

    # Add Female Turnout line (labels below the line)
    fig_turnout.add_trace(go.Scatter(
        x=gender_ratio['Year'],
        y=gender_ratio['Female_Turnout'],
        mode='lines+markers+text',
        name='Female Turnout',
        line=dict(color='red'),
        marker=dict(symbol='circle'),
        text=gender_ratio['Female_Turnout'].apply(lambda x: f"{x:.2f}%"),  # Format as percentage with 2 decimals
        textposition="bottom center"          # Position labels below the markers
    ))

    # Add Male Turnout line (labels above the line)
    fig_turnout.add_trace(go.Scatter(
        x=gender_ratio['Year'],
        y=gender_ratio['Male_Turnout'],
        mode='lines+markers+text',
        name='Male Turnout',
        line=dict(color='blue'),
        marker=dict(symbol='square'),
        text=gender_ratio['Male_Turnout'].apply(lambda x: f"{x:.2f}%"),    # Format as percentage with 2 decimals
        textposition="top center"             # Position labels above the markers
    ))

    # Calculate min and max values
    min_y = min(gender_ratio[['Female_Turnout', 'Male_Turnout']].min()) - 2
    max_y = max(gender_ratio[['Female_Turnout', 'Male_Turnout']].max()) + 2


    # Update layout
    fig_turnout.update_layout(
        xaxis_title='Year',
        yaxis_title='Turnout (%)',
        yaxis=dict(
            range=[min_y, max_y]  # Set lower limit to min_y and upper limit slightly beyond max_y
        ),
        legend_title='Legend',
        legend=dict(
            orientation="h",  # Horizontal orientation for the legend
            yanchor="top",    # Anchor the legend to the top
            y=-0.2,           # Position the legend below the plot
            xanchor="center", # Center the legend horizontally
            x=0.5             # Center the legend horizontally
        ),
        margin=dict(t=50, b=100, l=50, r=50),  # Adjust margins to provide space for the legend
        template='plotly_white',
        height=500,
        title="Gender-wise Voter Turnout Over Years"
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig_turnout)


st.markdown("<br><br>", unsafe_allow_html=True)


# Function to generate a dataframe-related response using Gemini
def generateDataframeResponse(dataFrame, prompt):
    df_summary = dataFrame.to_string()
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        full_prompt = (
            f"Role: Act as PANDAS AI Model.\n"
            f"Task: Given the question '{prompt}', provide a clear, concise response based on the data below, in 70 words and write 'For further details, click on the Analyze button'.\n"
            f"Dataframe summary: {df_summary}\n"
            f"Avoid starting with 'PANDAS AI model here'."
        )
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I'm sorry, I can't process that request right now. Error: {e}"

# Function to generate a detailed analysis including general assumptions using Gemini
def generateDetailedAnalysis(dataFrame, prompt):
    df_summary = dataFrame.to_string()
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        full_prompt = (
            f"Role: Act as an informed analyst.\n"
            f"Task: Given the question '{prompt}', provide a detailed analysis based on the data below and include general assumptions. Respond in 150 words or less.\n"
            f"Dataframe summary: {df_summary}\n"
            f"Break down the response into:\n"
            f"- State-wise analysis (if applicable)\n"
            f"- Party-wise analysis (if applicable)\n"
            f"- General context and assumptions related to the data\n"
            f"Avoid including irrelevant technical details and focus on what would be meaningful and understandable to a general audience."
        )
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I'm sorry, I can't provide a detailed analysis right now. Error: {e}"

# Function to analyze trends using Gemini
def analyzeTrends(dataFrame, input_query):
    try:
        summary = dataFrame.describe().to_string() # Use describe for numerical overview
        model = genai.GenerativeModel('gemini-1.5-pro')
        full_prompt = (
            f"Role: Act as Analyst and a social commentator.\n"
            f"Task: Analyze the trend in the following data and explain the possible real-world reasons behind these trends based on Indian election history:\n\n"
            f"{summary}\n\n"
            f"Question from user: {input_query}\n\n"
            f"Provide insights that consider typical patterns and historical context. Break down your explanation into:\n"
            f"- Analyze the dataframe and explain the trends\n"
            f"- Provide real-life scenarios related to India\n"
            f"Avoid including irrelevant technical details and focus on practical explanations that are understandable to a general audience."
        )
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"I can't answer that question. Error: {e}"


# Streamlit input boxes and image buttons
st.title("Ask Gyan")

# Create columns for inputs and buttons
col1_ai, col2_ai = st.columns(2) # Renamed columns to avoid conflict if any

# Initialize session state for queries if not already present
if "input1_query_submitted" not in st.session_state:
    st.session_state.input1_query_submitted = ""
if "input2_query_submitted" not in st.session_state:
    st.session_state.input2_query_submitted = ""

with col1_ai:
    input1_text = st.text_input("Ask Queries related to Top Parties with Seats Won ✨", placeholder="Ask me about the top parties", key="input1_text_area")
    
    # Button to submit the query for initial response
    if st.button("Get Response", key="submit_button_col1"):
        if input1_text:
            st.session_state.input1_query_submitted = input1_text
            initial_response = generateDataframeResponse(dataFrame=top_parties_df, prompt=st.session_state.input1_query_submitted)
            st.write(initial_response)
        else:
            st.write("Please enter a query.")
            
    # "Analyze" button for detailed analysis
    if st.button("Analyze Details", key="analyze_button_col1"):
        if st.session_state.input1_query_submitted:
            detailed_response = generateDetailedAnalysis(dataFrame=top_parties_df, prompt=st.session_state.input1_query_submitted)
            st.write(detailed_response)
        else:
            st.write("Please submit a query using 'Get Response' first to analyze.")

with col2_ai:
    input2_text = st.text_input("Ask Queries related to Genderwise Voter Turnout ✨", placeholder="Ask me about voter turnout trends", key="input2_text_area")
    
    # Button to submit the query for initial response
    if st.button("Get Response", key="submit_button_col2"):
        if input2_text:
            st.session_state.input2_query_submitted = input2_text
            answer = generateDataframeResponse(dataFrame=gender_ratio, prompt=st.session_state.input2_query_submitted)
            st.write(answer)
        else:
            st.write("Please enter a query.")

    # "Analyze" button for trend analysis
    if st.button("Analyze Trends", key="analyze_button_col2"):
        if st.session_state.input2_query_submitted:
            trend_analysis = analyzeTrends(dataFrame=gender_ratio, input_query=st.session_state.input2_query_submitted)
            st.write(trend_analysis)
        else:
            st.write("Please submit a query using 'Get Response' first to analyze.")


# Constituency Overview Metrics
st.markdown("<br><br>", unsafe_allow_html=True)
st.subheader("Constituency Metrics")
col1_metrics, col2_metrics, col3_metrics, col4_metrics = st.columns(4) # Renamed columns

# Total votes polled
total_votes_polled = filtered_df1['votes_polled'].sum()
col1_metrics.metric("Total Votes Polled 📦", format_number(total_votes_polled))

# Number of constituencies
total_constituencies_filtered = filtered_df1['pc_name'].nunique()
col2_metrics.metric("Total Constituencies", format_number(total_constituencies_filtered))

# Total male electors
total_male_electors = filtered_df1['male_electors'].sum()
col3_metrics.metric("Total Male Electors ", format_number(total_male_electors))

# Total female electors
total_female_electors = filtered_df1['female_electors'].sum()
col4_metrics.metric("Total Female Electors 🙎‍♀️", format_number(total_female_electors)," ")

# Constituency Overview Graphs and Map
st.markdown("<br><br>", unsafe_allow_html=True)
st.subheader("Constituency Analysis")

# Layout with two columns
col_charts_1, col_charts_2 = st.columns([1, 1]) # Renamed columns

with col_charts_1:
    if not filtered_df1.empty:
        grouped_df1 = filtered_df1.groupby('pc_name')['votes_polled'].sum().reset_index()
        fig1 = px.bar(grouped_df1, x='pc_name', y='votes_polled', title='Votes Polled by Constituency',height=500)
        fig1.update_xaxes(title_text='Constituency')
        st.plotly_chart(fig1)
    else:
        st.info("No data available for selected filters to show 'Votes Polled by Constituency'.")
    
    # Trend in Votes Polled by Constituency
    pc_trend_data = df1[df1['pc_name'].isin(pc_names)]
    if not pc_trend_data.empty:
        selected_pc_names_trend = st.multiselect("Select constituencies for trend analysis", options=pc_trend_data['pc_name'].unique(), default=list(pc_trend_data['pc_name'].unique()[:min(2, len(pc_trend_data['pc_name'].unique()))]))

        if selected_pc_names_trend:
            filtered_df_trend = pc_trend_data[pc_trend_data['pc_name'].isin(selected_pc_names_trend)]

            fig_trend = px.line(filtered_df_trend, 
                                x='election_year', 
                                y='votes_polled_percentage', 
                                color='pc_name', 
                                markers=True, 
                                title='Trend in Votes Polled by Constituency',
                                text='votes_polled_percentage',
                                height=500)

            fig_trend.update_layout(legend_title_text='Constituency',
                                xaxis_title='Election Year',
                                yaxis_title='Votes Polled Percentage (%)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(title_font=dict(size=18)),
                                yaxis=dict(title_font=dict(size=18)))
            fig_trend.update_traces(textposition='top center')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Please select at least one constituency for trend analysis.")
    else:
        st.info("No data available for selected filters to show 'Trend in Votes Polled by Constituency'.")


with col_charts_2:
    state_mapping = {
        "Andaman & Nicobar Islands": "Andaman & Nicobar",
        "Andhra Pradesh ": "Andhra Pradesh",
        "Arunachal Pradesh": "Arunachal Pradesh",
        "Assam": "Assam",
        "Bihar ": "Bihar",
        "Chandigarh": "Chandigarh",
        "Chhattisgarh": "Chhattisgarh",
        "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
        "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
        "Delhi": "Delhi",
        "Goa": "Goa",
        "Gujarat": "Gujarat",
        "Haryana": "Haryana",
        "Himachal Pradesh": "Himachal Pradesh",
        "Jammu & Kashmir": "Jammu & Kashmir",
        "Jharkhand": "Jharkhand",
        "Karnataka": "Karnataka",
        "Kerala": "Kerala",
        "Ladakh": "Ladakh",
        "Lakshadweep": "Lakshadweep",
        "Madhya Pradesh ": "Madhya Pradesh",
        "Maharashtra": "Maharashtra",
        "Manipur": "Manipur",
        "Meghalaya": "Meghalaya",
        "Mizoram": "Mizoram",
        "Nagaland": "Nagaland",
        "Orissa": "Odisha",
        "Pondicherry": "Puducherry",
        "Punjab": "Punjab",
        "Rajasthan": "Rajasthan",
        "Sikkim": "Sikkim",
        "Tamil Nadu": "Tamil Nadu",
        "Telangana": "Telangana",
        "Tripura": "Tripura",
        "Uttar Pradesh ": "Uttar Pradesh",
        "Uttarakhand": "Uttarakhand",
        "West Bengal": "West Bengal"
    }

    # Load GeoJSON data from a URL
    geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
    try:
        response = requests.get(geojson_url)
        geojson_data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load GeoJSON data: {e}. Please check your internet connection or the URL.")
        geojson_data = None # Set to None to prevent further errors

    if geojson_data:
        # Extract the state names from the GeoJSON file
        state_names_geojson = [feature['properties']['ST_NM'] for feature in geojson_data['features']]

        # Create a dropdown for selecting a party
        if not filtered_df3.empty:
            selected_party = st.selectbox('Select a Party:', filtered_df3['party'].unique(), key="party_select_map")

            # Filter the data for the selected party
            filtered_data_map = filtered_df3[filtered_df3['party'] == selected_party]

            # Group the filtered data by state and count the number of seats
            agg_data_map = filtered_data_map.groupby('state')['party'].count().reset_index(name='seats')

            # Apply the mapping to the 'state' column
            agg_data_map['state'] = agg_data_map['state'].map(state_mapping)

            # Handle any states that might not match by dropping null values
            agg_data_map = agg_data_map.dropna(subset=['state'])

            # Ensure all states from GeoJSON are present in the data
            agg_data_map_complete = pd.DataFrame({
                'state': state_names_geojson,
                'seats': [agg_data_map[agg_data_map['state'] == state]['seats'].sum() if state in agg_data_map['state'].values else 0 for state in state_names_geojson]
            })

            fig_map = go.Figure(data=go.Choropleth(
                    geojson=geojson_data,
                    featureidkey='properties.ST_NM',
                    locationmode='geojson-id',
                    locations=agg_data_map_complete['state'],
                    z=agg_data_map_complete['seats'],
                    autocolorscale=False,
                    colorscale='Reds',
                    marker_line_color='peachpuff',
                    colorbar=dict(
                title=dict(text="Seats", font=dict(color='white')),
                thickness=15,
                len=0.3,
                bgcolor='#0E1117',
                tick0=0,
                dtick=5,
                xanchor='right',
                x=1.0,
                yanchor='middle',
                y=0.4,
                tickfont=dict(color='white')
            )
            ))
            
            fig_map.update_geos(
                visible=False,
                projection=dict(type='mercator'),
                lonaxis={'range': [68, 98]},
                lataxis={'range': [6, 38]},
                showland=True,
                landcolor='white',
                showcountries=True,
                countrycolor='black',
                showocean=True,
                oceancolor='lightblue'
            )
            
            fig_map.update_layout(
                title=dict(
                    text=f"{selected_party} Seats In Lok Sabha Election",
                    xanchor='center',
                    x=0.5,
                    yref='paper',
                    yanchor='bottom',
                    y=1,
                    pad={'b': 10}
                ),
                margin={'r': 0, 't': 30, 'l': 0, 'b': 0},
                height=900,
                width=750,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                geo=dict(
                    visible=False,
                    showcoastlines=False,
                    showland=False,
                    showocean=False,
                    showcountries=False,
                    showlakes=False
                ),
                dragmode=False,
                uirevision='constant',
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No election data available for party map for the selected filters.")
    else:
        st.warning("Cannot display map as GeoJSON data could not be loaded.")


# Candidates Dashboard
st.subheader("Party by Constituency")

col1_cand, col2_cand = st.columns(2) # Renamed columns

with col1_cand:
    if not filtered_df2.empty:
        constituency_pie = st.selectbox(
            'Select Constituency for Party Vote Percentage:', 
            filtered_df2['pc_name'].unique(),
            index=0,
            key='selectbox_col1_cand_pie'
        )

        party_wise = filtered_df2[filtered_df2['pc_name'] == constituency_pie]
        party_votes_pct = party_wise.groupby('Party')['Votes_Percentage'].mean().reset_index()
        party_votes_pct = party_votes_pct.sort_values(by='Votes_Percentage', ascending=False)

        top_parties_pie = party_votes_pct.head(3)
        others_pie = party_votes_pct.iloc[3:]

        others_sum_pie = pd.DataFrame({
            'Party': ['Others'],
            'Votes_Percentage': [others_pie['Votes_Percentage'].sum()]
        })

        final_data_pie = pd.concat([top_parties_pie, others_sum_pie])

        custom_colors = ['#de7703', '#017a38', '#1ce2fc', '#c41cfc', '#DA70D6', '#87CEEB', '#32CD32', '#FFD700']
        
        fig_pie = px.pie(
            final_data_pie, 
            names='Party', 
            values='Votes_Percentage', 
            title=f'Votes Percentage by Party for {constituency_pie}',
            labels={'Party': 'Party', 'Votes_Percentage': 'Votes Percentage'},
            color_discrete_sequence=custom_colors
        )
        
        fig_pie.update_traces(
            textinfo='percent',
            hoverinfo='label+percent+value',
            textposition='inside',
            texttemplate='%{value:.2f}%',
            textfont=dict(color='white')
        )
        
        fig_pie.update_layout(
            legend_title='Party',
            margin=dict(t=50, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_pie)
    else:
        st.info("No candidate data available for selected filters to show 'Votes Percentage by Party'.")

with col2_cand:
    if not filtered_df2.empty:
        constituency_bar = st.selectbox(
            'Select Constituency for Candidate Votes:', 
            filtered_df2['pc_name'].unique(),
            index=0,
            key='selectbox_col2_cand_bar'
        )

        df_cand_bar = filtered_df2[filtered_df2['pc_name'] == constituency_bar]
        df_cand_bar = df_cand_bar.sort_values(by='Votes', ascending=False)

        # Display only the top 3 candidates initially
        top_n = 3
        top_df_cand_bar = df_cand_bar.head(top_n).sort_values(by='Votes', ascending=True)

        st.write(f"### Top {top_n} Candidates for {constituency_bar}")
        fig_top_cand = px.bar(
            top_df_cand_bar,
            y='Candidate_Name',
            x='Votes',
            color='Votes',
            color_continuous_scale='inferno',
            title=f'Top {top_n} Candidates for {constituency_bar}',
            labels={'Votes': 'Number of Votes', 'Candidate_Name': 'Candidate'},
            orientation='h',
            text='Votes'
        )
        fig_top_cand.update_traces(texttemplate='%{x:.0f}', textposition='outside')
        fig_top_cand.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=24, color='white', family="Arial"),
            font=dict(color='black', size=18, family="Arial"),
            margin=dict(l=150, r=10, t=70, b=50),
            coloraxis_showscale=False,
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_top_cand)

        # Expandable section for remaining candidates
        with st.expander("Show All Candidates"):
            fig_all_cand = px.bar(
                df_cand_bar,
                y='Candidate_Name',
                x='Votes',
                color='Votes',
                color_continuous_scale='inferno',
                title=f'Votes Distribution by All Candidates for {constituency_bar}',
                labels={'Votes': 'Number of Votes', 'Candidate_Name': 'Candidate'},
                orientation='h',
                text='Votes'
            )
            fig_all_cand.update_traces(texttemplate='%{x:.0f}', textposition='outside')
            fig_all_cand.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=24, color='white', family="Arial"),
                font=dict(color='black', size=18, family="Arial"),
                margin=dict(l=150, r=10, t=70, b=50),
                coloraxis_showscale=False,
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig_all_cand)
    else:
        st.info("No candidate data available for selected filters to show 'Candidates by Votes'.")


# Display the complete dashboard
st.markdown("---")

# Provide additional information and credits
st.sidebar.info("Data sourced from provided datasets.")
