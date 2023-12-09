import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import io
import sys
import plotly.express as px
import seaborn as sns
import numpy as np

st.set_page_config(
    page_title="Dashboard",
    page_icon="✅",
    layout="wide",
)

st.title("User Behavior Pattern Exploration")

df = pd.read_csv(r"C:\Users\simra\Downloads\comments_cleaned.csv")
#link to dataset is "https://www.kaggle.com/datasets/sanjanchaudhari/user-behavior-on-instagram"

st.sidebar.title("Options")
# Sidebar options for data exploration
explore_option = st.sidebar.selectbox("Select an Exploration Option", ["Overview","Dashboard", "Emoji Visualisation and Analysis","User Statistics", "Graphical Analysis"])
if explore_option == "Overview":
    st.subheader("Overview of User Behavior Data")
    st.write("Sample Data")
    st.write(df.head())  # first five rows
    st.write("Summary Statistics")
    st.write(df.describe())
    st.write("Values in descending order")
    st.write(df.sort_values(by='id', ascending=False))

    buffer = io.StringIO()
    sys.stdout = buffer
    # Display DataFrame.info() in the redirected output
    df.info()
    # Reset stdout
    sys.stdout = sys.__stdout__
    # Display the captured info in Streamlit
    st.text("DataFrame Info:")
    st.text(buffer.getvalue())

    df.drop(columns=["Unnamed: 0", "id"], inplace=True)

    df.drop_duplicates(inplace=True)  # Drop duplicates
    df.dropna(inplace=True)  # Handle missing values


elif explore_option == "User Statistics":
    st.subheader("User Statistics")

    # Display user statistics, e.g., counts, averages, etc.
    st.write(df.describe())

    # Count the number of comments per photo
    st.write('Distribution of comments per User')
    comments_per_user = df['User  id'].value_counts()
    st.write(comments_per_user)

    st.write('Distribution of comments per Photo')
    photo_comment_counts = df['Photo id'].value_counts()
    st.write(photo_comment_counts)

    # Identify the 10 photos that received the most comments
    top_photos = photo_comment_counts.nlargest(10)
    st.write("The 10 photos that received the most comments")
    st.write(top_photos)

    # Check the frequency of emoji usage in comments
    st.write("Frequency of emoji usage in comments")
    emoji_usage = df['emoji used'].value_counts()
    st.write(emoji_usage)

    # For each user, calculate the average number of hashtags used in their comments
    average_hashtags_by_user = df.groupby('User  id')['Hashtags used count'].mean()
    st.write("Average number of hashtags used in comments")
    st.write(average_hashtags_by_user)

    # Identify the users who use the most and the least number of hashtags on average
    user_with_most_hashtags = average_hashtags_by_user.idxmax()
    st.write("Users who use the most number of hashtags on average")
    st.write(user_with_most_hashtags)
    user_with_least_hashtags = average_hashtags_by_user.idxmin()
    st.write("Users who use the least number of hashtags on average")
    st.write(user_with_least_hashtags)

    Hashtag_count = df[['User  id', 'Hashtags used count']]
    st.write("Hashtag Count")
    st.write(Hashtag_count)

    df['word_count'] = df['comment'].apply(lambda text: len(text.split()))
    df['char_count'] = df['comment'].apply(len)
    average_lengths_emoji = df.groupby('emoji used')[['word_count', 'char_count']].mean()
    st.write("The average length of comments that use emojis and those that don't")
    st.write(average_lengths_emoji)




elif explore_option=="Graphical Analysis":  # Time Series Analysis
    st.subheader("Graphical Analysis")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Convert date strings to datetime.date objects
    df['Date'] = pd.to_datetime(df['created Timestamp'], format='%d-%m-%Y %H:%M')

    # Create an interactive line chart using Plotly Express
    fig = px.line(df, x='Date', y= 'Hashtags used count', title='Engagement Trends Over Time')
    st.plotly_chart(fig)


    fig = px.histogram(df, x='Hashtags used count', nbins=20, title='Distribution of Hashtags Used Count')
    fig.update_xaxes(title_text='Number of Hashtags Used')
    fig.update_yaxes(title_text='Count')
    st.plotly_chart(fig)

    fig1 = px.scatter(df, x='User  id', y='Hashtags used count', color='emoji used',
                     title='Distribution of Emoji and Hashtags Used',
                     labels={'User id': 'User ID', 'Hashtags used count': 'Hashtags Used Count'})
    st.plotly_chart(fig1)

    st.title('User Id Value Count (Pie Chart)')
    value_counts = df['User  id'].value_counts()
    value_counts_df = pd.DataFrame({'User  id': value_counts.index, 'Count': value_counts.values})
    fig = px.pie(value_counts_df, names='User  id', values='Count', title='User Id Value Count')
    st.plotly_chart(fig)


    st.title("Emoji used as per User Id")
    fig = px.scatter(df, x='User  id', y="emoji used", title='Emoji used as per User Id')
    fig.update_traces(marker=dict(size=10, opacity=0.6), selector=dict(mode='markers+text'))
    fig.update_xaxes(showticklabels=True, tickmode='array', tickvals=df['User  id'].unique(), title='User Id')
    fig.update_yaxes(title='Emoji Used')
    custom_x_values = [x for x in range(0,101,10)]  # Define the specific x-axis values you want to display
    fig.update_xaxes(showticklabels=True, tickvals=custom_x_values, title='User Id')

    selected_user_id = st.selectbox("Select User Id", df['User  id'].unique())
    selected_emoji = df[df['User  id'] == selected_user_id]['emoji used'].values[0]
    st.write(f"Emoji used by User Id {selected_user_id}: {selected_emoji}")

    # Display the interactive Plotly chart
    st.plotly_chart(fig)




    # Visualize the average number of words in comments for each sentiment category
    # Calculate the length of each comment in terms of the number of words and characters
    df['word_count'] = df['comment'].apply(lambda text: len(text.split()))
    df['char_count'] = df['comment'].apply(len)
    average_lengths = df.groupby('emoji used')[['word_count', 'char_count']].mean()
    average_lengths['word_count'].plot(kind='bar', figsize=(8, 6))
    plt.title('Average Number of Words in Comments by Emoji Used')
    plt.ylabel('Average Number of Words')
    plt.xlabel('Emoji Used')
    plt.xticks(rotation=0)
    st.pyplot()

    # Calculate the average number of characters in comments for each number of hashtags used
    average_length_by_hashtag_count = df.groupby('Hashtags used count')['char_count'].mean()

    average_length_by_hashtag_count.plot(kind='bar', figsize=(8, 6), color='skyblue')
    plt.title('Average Number of Characters in Comments for Each Number of Hashtags Used')
    plt.ylabel('Average Number of Characters')
    plt.xlabel('Number of Hashtags Used')
    plt.xticks(rotation=0)
    st.pyplot()

elif explore_option=="Emoji Visualisation and Analysis":
    st.title("Emoji Categorisation and Visualisation")
    add_radio = st.radio(
        "Select",
        ("Categorisation", "Visualisation"))
    if add_radio == "Categorisation":
        st.title("Content Categorization")


        def categorize_content(row):
            if row['emoji used'] == 'yes':
                return 'With Emoji'
            elif row['emoji used'] == 'no':
                return 'Without Emoji'
            else:
                return 'Other'


        # Create a new column 'Content Type' based on categorization function
        df['Content Type'] = df.apply(categorize_content, axis=1)

        # Display the DataFrame with the new 'Content Type' column
        st.write("Categorized Content:")
        st.dataframe(df[['User  id','comment','Content Type']])
    if add_radio == "Visualisation":
        st.title("User Engagement by Content Category")
        # Create a bar chart using Plotly Express
        fig = px.bar(df, x='emoji used', y='User  id', title='User Engagement by Content Category')
        fig.update_xaxes(title='Emoji Used')
        fig.update_yaxes(title='User id')
        col1, col2 = st.columns([3, 1])
        col1.plotly_chart(fig, theme="streamlit", use_container_width=True)
        col2.write(df[['User  id', 'emoji used']])

elif explore_option == "Dashboard":
    st.subheader("User Engagement Segmentation")

    # Define thresholds for engagement categories
    high_engagement_threshold = 5  # Customize this threshold as needed
    moderate_engagement_threshold = 2  # Customize this threshold as needed
    def categorize_engagement(row):
        if row['Hashtags used count'] >= high_engagement_threshold >= high_engagement_threshold:
            return "Highly Engaged"
        elif row['Hashtags used count'] >= moderate_engagement_threshold >= moderate_engagement_threshold:
            return "Moderately Engaged"
        else:
            return "Less Engaged"
    df['Engagement Category'] = df.apply(categorize_engagement, axis=1)
    st.dataframe(df[['User  id', 'emoji used', 'Engagement Category']])

# st.title("Engagement Dashboard")
# job_filter = st.selectbox("Select User Id", pd.unique(df["User  id"]))
# fig_col1, fig_col2 = st.columns(2)


# with fig_col1:
#     st.markdown("Density Heatmap")
#     fig = px.density_heatmap(
#         data_frame=df, y="User  id", x="Engagement Category"
#     )
#     st.write(fig)
#
#
# with fig_col2:
#     st.markdown("Histogram")
#     fig2 = px.histogram(data_frame=df, x="Engagement Category")
#     st.write(fig2)





