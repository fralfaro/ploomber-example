import streamlit as st
import base64
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff


def plot_confusion_matrix(y_true, y_pred, class_names):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create confusion matrix plot using Plotly
    fig = ff.create_annotated_heatmap(
        z=cm, x=class_names, y=class_names, colorscale="Blues", showscale=False
    )

    # Add labels
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted Value"),
        yaxis=dict(title="True Value"),
        width=400,
        height=400,
    )

    return fig


def plot_roc_curve(y_true, y_score):
    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Create a Plotly figure for the ROC curve
    fig = go.Figure()

    # Add the ROC curve with customized color
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.2f})",
            line=dict(color="skyblue"),
        )
    )

    # Add diagonal line for reference
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="lightblue", dash="dash"),
        name="Reference (AUC = 0.5)",
    )

    # Customize the layout of the figure
    fig.update_layout(
        title="ROC Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
        width=400,
        height=400,
    )
    return fig


# Initial page config
st.set_page_config(
    page_title="Titanic - ML from Disaster",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """
    Main function to set up the Streamlit app layout.
    """
    cs_sidebar()
    cs_body()
    return None


# Define img_to_bytes() function
def img_to_bytes(img_url):
    response = requests.get(img_url)
    img_bytes = response.content
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# Define the cs_sidebar() function
def cs_sidebar():
    """
    Populate the sidebar with various content sections related to Python.
    """
    st.sidebar.markdown(
        """[<img src='data:image/png;base64,{}' class='img-fluid' width=200 >](https://streamlit.io/)""".format(
            img_to_bytes(
                "https://b.kisscc0.com/20180813/siw/kisscc0-sinking-of-the-rms-titanic-drawing-ship-cartoon-the-titanic-5b713e8fafe551.8235205415341482397205.png"
            )
        ),
        unsafe_allow_html=True,
    )

    st.sidebar.header("Titanic - ML from Disaster")
    st.sidebar.markdown(
        """
        small>The [RMS Titanic](https://es.wikipedia.org/wiki/RMS_Titanic)
        as a British passenger
        liner that tragically sank in the North Atlantic
        Ocean on April 15, 1912, during its maiden voyage
        from Southampton, England, to New York City. It
         was the largest ship afloat at the time and was co
         nsidered to be unsinkable, but it struck an icebe
         rg and went down, resulting in a significant loss of life.</small>
        """,
        unsafe_allow_html=True,
    )

    # why python ?
    st.sidebar.markdown("__🛳️Goals__")
    st.sidebar.markdown(
        """
    <small>  This project aims to unravel
    the hidden patterns and unveil insights
    surrounding the tragic sinking of the RMS Titanic by harnessing
    the power of machine learning and a user-friendly
    web application framework called Streamlit. </small> """,
        unsafe_allow_html=True,
    )

    return None


# Define the cs_body() function
def cs_body():
    """
    Create content sections for the main body of the
     Streamlit cheat sheet with Python examples.
    """

    @st.cache_data()
    def load_data():
        # Load data from CSV file
        data = pd.read_csv(
            "https://raw.githubusercontent.com/fralfaro/ploomber-example/main/data/train.csv"
        )  # Replace 'titanic.csv' with your own data file
        # Convert certain columns to string type
        columns_to_convert = ["Pclass", "SibSp", "Parch"]
        data[columns_to_convert] = data[columns_to_convert].astype(str)
        # Fill missing values in 'Cabin' column with '-' and extract the first character
        data["Cabin"] = data["Cabin"].fillna("-").apply(lambda x: x[0])
        # Convert certain columns to string type again
        columns_to_convert = ["Pclass", "SibSp", "Parch"]
        data[columns_to_convert] = data[columns_to_convert].astype(str)
        return data

    # Title of the application
    st.title("Titanic EDA with Streamlit and Plotly")

    # Load Data: train.csv
    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    # Create a selectbox widget to choose between 'head' and 'tail'
    option = col1.selectbox("Select 'head' or 'tail'", options=["head", "tail"])

    # Create a number_input widget to select the number of rows
    num_rows = col2.slider(
        "Number of rows to display", min_value=1, max_value=50, value=5
    )

    # Function to display the DataFrame according to the selected option
    def display_dataframe(option, num_rows):
        if option == "head":
            st.write(df.head(num_rows))
        elif option == "tail":
            st.write(df.tail(num_rows))

    # Display the DataFrame based on the selected option
    display_dataframe(option, num_rows)

    st.subheader("Plots")
    st.markdown("""<big>Numeric Variables </big>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    x_column_options = ["Age", "Fare"]  # You can add more columns if you wish
    x_column = col1.selectbox("Select column:", options=x_column_options)
    plot_type_options = [
        "univariate",
        "bivariate",
    ]  # You can add more plot types if you wish
    plot_type = col2.selectbox("Numeric Plot type:", options=plot_type_options)

    if plot_type == "univariate":
        # Create a histogram plot with Plotly
        fig = px.histogram(df, x=x_column, title=f"Histogram of {x_column}")
        # Display the plot in Streamlit
        fig.update_traces(
            marker=dict(line=dict(color="black", width=1)), marker_color="skyblue"
        )
        st.plotly_chart(fig)
    else:
        # Create a grouped histogram plot with Plotly
        fig = px.histogram(
            df,
            x=x_column,
            color="Survived",
            title=f"Histogram of {x_column} by Survived",
        )
        # Display the plot in Streamlit
        fig.update_traces(marker=dict(line=dict(color="black", width=1)))
        st.plotly_chart(fig)

    st.markdown("""<big>Categorical Variables </big>""", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    x_column_options_cat = [
        "Pclass",
        "SibSp",
        "Parch",
        "Sex",
        "Cabin",
        "Embarked",
    ]  # You can add more columns if you wish
    x_column_cat = col1.selectbox("Select column:", options=x_column_options_cat)
    plot_type_options_cat = [
        "univariate",
        "bivariate",
    ]  # You can add more plot types if you wish
    plot_type_cat = col2.selectbox(
        "Categorical Plot type:", options=plot_type_options_cat
    )

    if plot_type_cat == "univariate":
        # Create a histogram plot with Plotly
        fig = px.histogram(df, x=x_column_cat, title=f"Barplot of {x_column_cat}")
        fig.update_traces(
            marker=dict(line=dict(color="black", width=1)), marker_color="skyblue"
        )
        st.plotly_chart(fig)
    else:
        # Create a grouped histogram plot with Plotly
        fig = px.histogram(
            df,
            x=x_column_cat,
            color="Survived",
            title=f"Barplot of {x_column_cat} by Survived",
        )
        # Display the plot in Streamlit
        fig.update_traces(marker=dict(line=dict(color="black", width=1)))
        st.plotly_chart(fig)

    st.subheader("Models and Metrics")

    @st.cache_data()
    def load_metrics_data():
        data = pd.read_csv(
            "data/metrics.csv"
        )  # Replace 'titanic.csv' with your own data file

        return data

    @st.cache_data()
    def load_predictions_data():
        data = pd.read_csv(
            "data/y_predictions.csv"
        )  # Replace 'titanic.csv' with your own data file

        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y_score = data["y_score"]

        return y_true, y_pred, y_score

    metrics = load_metrics_data()

    # Create radio buttons to select the column
    selected_column = st.radio(
        "Select a column to sort by:",
        ("Accuracy", "Precision", "Recall", "F1-Score"),
        horizontal=True,
    )

    # Sort the DataFrame according to the selected column
    sorted_df = metrics.sort_values(by=selected_column, ascending=False).reset_index(
        drop=True
    )
    st.write(sorted_df)

    # Feature Importances
    y_true, y_pred, y_score = load_predictions_data()

    col1, col2, col3, col4, col5 = st.columns(5)

    fig1 = plot_confusion_matrix(y_true, y_pred, ["Class 0", "Class 1"])
    fig2 = plot_roc_curve(y_true, y_score)
    col1.plotly_chart(fig1)
    col3.plotly_chart(fig2)

    @st.cache_data()
    def load_importances_data():
        data = pd.read_csv(
            "data/feature_importances.csv"
        )  # Replace 'titanic.csv' with your own data file

        return data

    df_importances = load_importances_data()

    fig3 = px.bar(
        df_importances.loc[lambda x: x["importances"] > 0].sort_values(
            by="importances"
        ),
        x="importances",
        y="feature_names",
        orientation="h",
    )
    fig3.update_layout(
        title="Feature Importances", xaxis_title="Feature", yaxis_title="Importance"
    )
    fig3.update_traces(marker_color="skyblue")
    st.plotly_chart(fig3)

    st.subheader("Predictions")

    @st.cache_data()
    def load_test_data():
        cols = [
            "PassengerId",
            "Survived",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ]
        data = pd.read_csv(
            "data/predictions.csv"
        )  # Replace 'titanic.csv' with your own data file

        return data[cols]

    test = load_test_data()
    st.write(test.head())


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
