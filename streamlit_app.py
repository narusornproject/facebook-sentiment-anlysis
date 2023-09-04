import streamlit as st
import pandas as pd
import numpy as np
import helper_functions as hf

page = np.array(['ALL', 'Life_Motorcycle', 'Business_Banking', 'Thailand_Environment',
       'AMPT', 'ลิขสิทธิ์บอลโลก',
       'Thailand_Law_ต่างชาติถือครองที่ดิน', 'Move1',
       'Thailand_constitution', 'UTNP',
       'Culture_TV and Entertainment (Series)', 'Culture_Art and Design',
       'Life_Travel (Thai)', 'What if_EV Car', 'Life_Car',
       'สถานีชาร์จรถยนต์ไฟฟ้า', 'ประกันรถยนต์ไฟฟ้า',
       'Thailand_Corruption', 'Culture_Book', 'COKE vs PEPSI ', 'Move2',
       'Life_Travel (Foreign)', 'Thailand_Envi_สาธารณสุข',
       'Business_Insurance', 'Thailand_Infra_คมนาคม', 'CP กับ Worldcup',
       'Business_Economy (อุตสาหกรรม)', 'Business_Crypto', 'Life_EV Car',
       'Business_WorkLife', 'Christmas Gift_2022', 'AMPP',
       'ฺีBusiness_Gold Investment'])

st.set_page_config(
    page_title="Facebook Sentiment Analyzer", page_icon="📊", layout="wide"
)


adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

def search_callback():
    # Initialize dataset
    if "original_df" not in st.session_state:
        st.session_state.original_df = pd.read_excel('cleaned_sample_v3.xlsx', index_col=0)

    # Filter by 'type' column based on user input
    if "search_type" in st.session_state:
        if st.session_state.search_type == "post&comment":
            # If user input is "post&comment", don't filter on 'type'
            st.session_state.df = st.session_state.original_df.copy()
        else:
            st.session_state.df = st.session_state.original_df[st.session_state.original_df.type == st.session_state.search_type]
    
    # Further filter by search_term on 'page' column
    if st.session_state.search_term != 'ALL':
        st.session_state.df = st.session_state.df[st.session_state.df.page == st.session_state.search_term]


with st.sidebar:
    st.title("Facebook Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the latest Facebook posts or comments. based on 
            the entered search term. Since the app can only predict positive or 
            negative sentiment, it is more suitable towards analyzing the 
            sentiment of brand, product, service, company, or person. 
            Only Thai posts or comments are supported.
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.subheader("Search Parameters")
    st.selectbox("Search term", page, key="search_term")
    st.radio("Search type:", ["post&comment", "post", "comment"], key='search_type')
    search_callback()
    
    st.markdown(
        "Note: it may take a while to load the results, especially with large number of Facebook posts or comments."
    )

    st.markdown("[Github link](https://github.com/narusornproject/FacebookSentimentAnalyzer)")
    st.markdown("Created by Narusorn Rojrattanatrai")


if "df" in st.session_state:

    def make_dashboard(facebook_df, bar_color, wc_color):
    
        
        # Create a function to format the boxes
        def create_stat_box(stat_name, value):
            return f"""
            <div style="border: 1px solid #ddd; padding: 10px; width: 70%; margin: 0 auto;">
                <p style="margin-bottom: 5px;"><strong>{stat_name}</strong></p>
                <p style="margin: 0;">{value}</p>
            </div>
            """
        
        # Create the boxes for each statistic
        total = create_stat_box("Total Number of Facebook:", len(facebook_df))
        avg_interacts = create_stat_box("Average Number of Interacts:", np.round(facebook_df.interactions.mean(), 3))
        avg_comments = create_stat_box("Average Number of Comments:", np.round(facebook_df.comments.mean(), 3))
        avg_shares = create_stat_box("Average Number of Shares:", np.round(facebook_df.comments.mean(), 3))
        
        # Create two side-by-side columns in Streamlit
        col1, col2, col3, col4 = st.columns(4)

        # Display the statistics in the columns
        col1.markdown(total, unsafe_allow_html=True)
        col2.markdown(avg_interacts, unsafe_allow_html=True)
        col3.markdown(avg_comments, unsafe_allow_html=True)
        col4.markdown(avg_shares, unsafe_allow_html=True)
            
        line_plot = hf.plot_line(facebook_df)
        st.plotly_chart(line_plot, theme=None, use_container_width=True)

        # first row
        col1, col2, col3 = st.columns([28, 34, 38])
        with col1:
            sentiment_plot = hf.plot_sentiment(facebook_df)
            sentiment_plot.update_layout(height=350, title_x=0.5)
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        with col2:
            top_unigram = hf.get_top_n_gram(facebook_df, ngram_range=1, n=10)
            unigram_plot = hf.plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
            unigram_plot.update_layout(height=350)
            st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
        with col3:
            top_bigram = hf.get_top_n_gram(facebook_df, ngram_range=2, n=10)
            bigram_plot = hf.plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
            bigram_plot.update_layout(height=350)
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

        # second row
        col1, col2 = st.columns([60, 40])
        with col1:

            def sentiment_color(sentiment):
                if sentiment == "Positive":
                    return "background-color: #1F77B4; color: white"
                else:
                    return "background-color: #FF7F0E"
            st.dataframe(
                facebook_df[["sentiment", "text"]].style.applymap(
                    sentiment_color, subset=["sentiment"]
                ),
                height=575,
                use_container_width=True
            )
        with col2:
            wordcloud = hf.plot_wordcloud(facebook_df, colormap=wc_color)
            st.pyplot(wordcloud)

    adjust_tab_font = """
    <style>
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;
    }
    </style>
    """

    st.write(adjust_tab_font, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["All", "Positive 😊", "Negative ☹️"])
    with tab1:
        facebook_df = st.session_state.df
        make_dashboard(facebook_df, bar_color="#54A24B", wc_color="Greens")
    with tab2:
        facebook_df = st.session_state.df.query("sentiment == 'Positive'")
        make_dashboard(facebook_df, bar_color="#1F77B4", wc_color="Blues")
    with tab3:
        facebook_df = st.session_state.df.query("sentiment == 'Negative'")
        make_dashboard(facebook_df, bar_color="#FF7F0E", wc_color="Oranges")
